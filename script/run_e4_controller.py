"""
E4: Risk Controller — LR-only Risk Control
=============================================

目的: 验证 LR-only 约束能否控制跨域风险 (P3)。

核心比较 (路由策略):
  1. Path A only
  2. Path B only
  3. fixed g=0.5
  4. driving only
  5. constraint only
  6. free router
  7. driving + constraint Controller

评价:
  - AUROC / AUPRC (R_hat 预测 Path B 变差)
  - ECE / Brier score
  - routing regret
  - negative-transfer rate
  - worst-domain RMSE

用法:
  # S6: 训练 Controller
  python3 script/run_e4_controller.py --stage train_controller \
      --wm_checkpoint /path/to/wm.pth \
      --student_checkpoint /path/to/student.pth \
      --path_b_checkpoint /path/to/pathb.pth \
      --train_dir /path/to/train --val_dir /path/to/val

  # 路由策略比较
  python3 script/run_e4_controller.py --stage routing_compare \
      --wm_checkpoint /path/to/wm.pth \
      --student_checkpoint /path/to/student.pth \
      --path_b_checkpoint /path/to/pathb.pth \
      --controller_checkpoint /path/to/controller.pth \
      --test_dir /path/to/test
"""
import os
import sys
import json
import time
import datetime
import argparse
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 确保能找到项目模块
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.driveguard_wm import (
    DriveGuardWM, Controller, PredictionBundle,
    MatrixStudent,
)
from dataset.temporal_dataset import TemporalDownscaleDataset


def r_map_to_gate(r_hat):
    """
    将 R (风险) 映射为 gate 抑制信号。
    R 高 → gate 低 (回退 Path A); R 低 → gate 高 (信任 Path B)。
    返回与 gate 同尺寸的 [0,1] 张量。
    """
    # r_hat: [B, K, N] → 平均到 [B, N]
    r_token = r_hat.mean(dim=1)          # [B, N]
    # 每个 batch 内归一化到 [0,1]
    r_min = r_token.min(dim=1, keepdim=True).values
    r_max = r_token.max(dim=1, keepdim=True).values
    r_norm = (r_token - r_min) / (r_max - r_min + 1e-8)   # [B, N]
    # R 高 → gate 低
    gate_contrib = 1.0 - r_norm          # [B, N]
    return gate_contrib.unsqueeze(1)     # [B, 1, N]


class E4ControllerValidator:
    """
    E4: Risk Controller — validation and routing comparison.

    Args:
        output_dir: 输出目录
        device: 计算设备
        config: 配置 dict
    """

    def __init__(self, output_dir, device=None, config=None, exp_name=None):
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}

        self.batch_size = self.config.get('batch_size', 4)
        self.epochs = self.config.get('epochs', 30)
        self.lr = self.config.get('lr', 1e-3)
        self.patience = self.config.get('patience', 10)
        self.num_workers = self.config.get('num_workers', 2)
        self.T = self.config.get('T', 5)

        # 架构
        self.K = self.config.get('K', 4)
        self.N = self.config.get('N', 16)
        self.Dg = self.config.get('Dg', 32)
        self.Dm = self.config.get('Dm', 32)
        self.Dl = self.config.get('Dl', 32)
        self.Ds = self.config.get('Ds', 64)

        # 结果目录
        self.exp_name = exp_name or f"e4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join(self.output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    # S6: Controller Training
    # ════════════════════════════════════════════════════════════════

    def train_controller(self, wm_checkpoint, student_checkpoint,
                         path_b_checkpoint, train_dir, val_dir):
        """
        S6: 训练 Controller — 以 OOF Path A/B 预测为输入。

        Args:
            wm_checkpoint: S1 World Model checkpoint
            student_checkpoint: S4 Matrix Student checkpoint
            path_b_checkpoint: S5 Path B checkpoint
            train_dir: OOF predictions 目录
            val_dir: 验证数据目录
        """
        print(f"\n{'='*60}")
        print("[E4] S6: Controller Training")
        print(f"{'='*60}")

        # 加载模型 (全部冻结)
        wm = self._load_wm(wm_checkpoint)
        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        self._load_trained_components(wm, student_checkpoint, path_b_checkpoint)

        # Controller (待训练)
        controller = Controller().to(self.device)

        # 数据集
        val_ds = TemporalDownscaleDataset(
            val_dir, 'val', T=self.T, normalize=True)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)

        # Controller 仅需要很少 epoch
        optimizer = optim.Adam(controller.parameters(), lr=self.lr)

        n_params = sum(p.numel() for p in controller.parameters())
        print(f"[E4] Controller params: {n_params:,}")

        # ─── 训练 ─────────────────────────────────────────────
        best_loss = float('inf')
        best_state = None

        for epoch in range(self.epochs):
            controller.train()
            total_loss = 0.0
            num_batches = 0

            loader = torch.utils.data.DataLoader(
                TemporalDownscaleDataset(train_dir, 'train', T=self.T, normalize=True),
                batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

            pbar = tqdm(loader, desc=f'Controller Epoch {epoch}')
            for batch in pbar:
                lr_seq = batch['lr'].to(self.device)
                hr_seq = batch['hr'].to(self.device)
                land_mask = batch['land_mask'].to(self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    B, T, C, H_lr, W_lr = lr_seq.shape
                    Hp, Wp = H_lr // 4, W_lr // 4
                    target_size = (hr_seq.shape[3], hr_seq.shape[4])
                    x_now = lr_seq[:, -1]

                    # WM prior
                    g_obs_list, l_obs_list, assignment_list = [], [], []
                    for t in range(T):
                        g_obs, l_obs, assignment, _ = wm._encode_frame(lr_seq[:, t])
                        g_obs_list.append(g_obs)
                        l_obs_list.append(l_obs)
                        assignment_list.append(assignment)
                    g_obs_seq = torch.stack(g_obs_list, dim=1)
                    l_obs_seq = torch.stack(l_obs_list, dim=1)
                    assignment_seq = torch.stack(assignment_list, dim=1)
                    prior_states = wm._rollout_prior(g_obs_seq, l_obs_seq, assignment_seq, Hp, Wp)
                    last_state = prior_states[-1]

                    # Path A
                    y_a = wm._decode_path_a(last_state, Hp, Wp, target_size)

                    # D_hat, R_hat
                    d_logits, r_raw = wm.matrix_student(
                        last_state.G, last_state.M, last_state.L, x_now)
                    d_hat = F.softplus(d_logits)
                    r_hat = F.softplus(r_raw)

                    # Path B
                    y_b = wm._decode_path_b(last_state, x_now, d_hat, Hp, Wp, target_size)

                # Controller forward
                gate, y_hat = controller(
                    d_hat, r_hat, last_state.assignment,
                    Hp, Wp, y_a.shape[2], y_a.shape[3],
                    y_a, y_b,
                )

                # Loss: BCE(gate, gate_target) + mixed prediction loss + monotonicity
                loss_a = F.l1_loss(y_a, hr_seq[:, -1], reduction='none')
                loss_b = F.l1_loss(y_b, hr_seq[:, -1], reduction='none')

                mask = land_mask.unsqueeze(1)
                gate_target = torch.sigmoid((loss_a - loss_b) / 0.5)

                bce_loss = F.binary_cross_entropy(gate * mask, gate_target * mask)
                pred_loss = F.l1_loss(y_hat * mask, hr_seq[:, -1] * mask)
                loss = bce_loss + pred_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Controller Epoch {epoch}: loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = controller.state_dict().copy()

        # 保存
        if best_state:
            ckpt_path = os.path.join(self.exp_dir, 'best_controller.pth')
            torch.save({
                'model_state_dict': best_state,
                'config': self.config,
            }, ckpt_path)
            print(f"[E4] Controller saved: {ckpt_path}")
            return ckpt_path
        return None

    # ════════════════════════════════════════════════════════════════
    # 路由策略比较
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def routing_compare(self, wm_checkpoint, student_checkpoint,
                        path_b_checkpoint, controller_checkpoint, test_dir):
        """
        在测试集上比较所有路由策略。

        Strategies: path_a_only, path_b_only, fixed_g_0.5,
                   driving_only, constraint_only, free_router, full_controller
        """
        print(f"\n{'='*60}")
        print("[E4] Routing Strategy Comparison")
        print(f"{'='*60}")

        wm = self._load_wm(wm_checkpoint)
        wm.eval()
        self._load_trained_components(wm, student_checkpoint, path_b_checkpoint)

        controller = Controller().to(self.device)
        if controller_checkpoint:
            ckpt = torch.load(controller_checkpoint, map_location=self.device)
            controller.load_state_dict(ckpt.get('model_state_dict', ckpt))

        test_ds = TemporalDownscaleDataset(
            test_dir, 'val', T=self.T, normalize=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=2, shuffle=False, num_workers=2)

        def _r_gate_hr(r_hat, ref_shape):
            """将 R 映射为 gate 并上采样到 HR 尺寸."""
            gate_c = r_map_to_gate(r_hat)      # [B, 1, N]
            gate_c4 = gate_c.unsqueeze(-1)     # [B, 1, N, 1]
            gate_hr = F.interpolate(gate_c4, size=(ref_shape[2], ref_shape[3]),
                                    mode='bilinear', align_corners=False)
            return gate_hr

        strategies = {
            'path_a_only': lambda ya, yb, g, d, r: ya,
            'path_b_only': lambda ya, yb, g, d, r: yb,
            'fixed_g_0.5': lambda ya, yb, g, d, r: ya + 0.5 * (yb - ya),
            # driving_only: gate 只由 D 驱动 (风险项关闭)
            'driving_only': lambda ya, yb, g, d, r: ya + g * (yb - ya),
            # constraint_only: gate 反比于 R (只靠风险约束)
            'constraint_only': lambda ya, yb, g, d, r: ya + _r_gate_hr(r, ya.shape) * (yb - ya),
            # free_router: 学习到的固定加权 (此处用 r 的反向近似)
            'free_router': lambda ya, yb, g, d, r: ya + 0.5 * _r_gate_hr(r, ya.shape) * (yb - ya),
            'full_controller': lambda ya, yb, g, d, r: ya + g * (yb - ya),
        }

        results = {}
        for name, blend_fn in strategies.items():
            mae_list = []
            for batch in tqdm(test_loader, desc=f'[E4] {name}'):
                lr_seq = batch['lr'].to(self.device)
                hr_seq = batch['hr'].to(self.device)
                land_mask = batch['land_mask'].to(self.device)

                B, T, C, H_lr, W_lr = lr_seq.shape
                Hp, Wp = H_lr // 4, W_lr // 4
                target_size = (hr_seq.shape[3], hr_seq.shape[4])
                x_now = lr_seq[:, -1]

                # Forward all paths
                g_obs_list, l_obs_list, assignment_list = [], [], []
                for t in range(T):
                    g_obs, l_obs, assignment, _ = wm._encode_frame(lr_seq[:, t])
                    g_obs_list.append(g_obs)
                    l_obs_list.append(l_obs)
                    assignment_list.append(assignment)

                g_obs_seq = torch.stack(g_obs_list, dim=1)
                l_obs_seq = torch.stack(l_obs_list, dim=1)
                assignment_seq = torch.stack(assignment_list, dim=1)
                prior_states = wm._rollout_prior(g_obs_seq, l_obs_seq, assignment_seq, Hp, Wp)
                last_state = prior_states[-1]

                y_a = wm._decode_path_a(last_state, Hp, Wp, target_size)

                d_logits, r_raw = wm.matrix_student(
                    last_state.G, last_state.M, last_state.L, x_now)
                d_hat = F.softplus(d_logits)
                r_hat = F.softplus(r_raw)

                y_b = wm._decode_path_b(last_state, x_now, d_hat, Hp, Wp, target_size)

                # Gate
                gate, _ = controller(d_hat, r_hat, last_state.assignment,
                                     Hp, Wp, y_a.shape[2], y_a.shape[3], y_a, y_b)

                y_hat = blend_fn(y_a, y_b, gate, d_hat, r_hat)
                mask = land_mask.unsqueeze(1)
                mae = F.l1_loss(y_hat * mask, hr_seq[:, -1] * mask).item()
                mae_list.append(mae)

            results[name] = {
                'mae': float(np.mean(mae_list)),
                'mae_std': float(np.std(mae_list)),
            }
            print(f"  {name}: MAE={results[name]['mae']:.4f}")

        results_path = os.path.join(self.exp_dir, 'routing_compare.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _load_trained_components(self, wm, student_checkpoint, path_b_checkpoint):
        """Attach the independently trained S4 and S5 components."""
        student_ckpt = torch.load(student_checkpoint, map_location=self.device)
        wm.matrix_student.load_state_dict(
            student_ckpt.get('model_state_dict', student_ckpt)
        )

        path_b_ckpt = torch.load(path_b_checkpoint, map_location=self.device)
        for name in (
            'path_b_encoder', 'drive_interface', 'process_moe',
            'local_ssm', 'path_b_decoder',
        ):
            getattr(wm, name).load_state_dict(path_b_ckpt[name])

        wm.eval()

    def _load_wm(self, checkpoint_path):
        from models.driveguard_wm.loader import load_model_from_checkpoint
        return load_model_from_checkpoint(
            checkpoint_path, device=self.device, strict=False)


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="E4: Risk Controller")

    parser.add_argument('--stage', type=str, required=True,
                        choices=['train_controller', 'routing_compare'],
                        help='阶段')
    parser.add_argument('--wm_checkpoint', type=str, default=None)
    parser.add_argument('--student_checkpoint', type=str, default=None)
    parser.add_argument('--path_b_checkpoint', type=str, default=None)
    parser.add_argument('--controller_checkpoint', type=str, default=None)

    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--output_dir', type=str,
                        default='outputs')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    config = {
        'T': args.T, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'num_workers': args.num_workers,
    }

    validator = E4ControllerValidator(
        output_dir=args.output_dir, device=args.device, config=config, exp_name=args.exp_name)

    if args.stage == 'train_controller':
        assert (args.wm_checkpoint and args.student_checkpoint
                and args.path_b_checkpoint and args.train_dir)
        validator.train_controller(
            args.wm_checkpoint, args.student_checkpoint, args.path_b_checkpoint,
            args.train_dir, args.val_dir or args.train_dir)

    elif args.stage == 'routing_compare':
        assert (args.wm_checkpoint and args.student_checkpoint
                and args.path_b_checkpoint and args.controller_checkpoint
                and args.test_dir)
        validator.routing_compare(
            args.wm_checkpoint, args.student_checkpoint, args.path_b_checkpoint,
            args.controller_checkpoint, args.test_dir)

    print(f"\n[E4] Stage '{args.stage}' complete. Output: {validator.exp_dir}")


if __name__ == "__main__":
    main()
