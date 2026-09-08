"""
E3: Dual-Path Ablation — Historical vs. Current Information
=============================================================

目的: 验证历史动力和当前空间信息具有不同迁移性。

变体:
  - 4 个信息因子: Current-Coarse, Current-Full, History-Coarse, History-Full
  - 7 个双路径关键比较: Path A only, local encoder only, Path A+B w/o driving,
    free feature fusion, random driving, full driver-coupled, widened single path

用法:
  # 训练双路径变体
  python3 script/run_e3_dual_path_ablation.py \
      --stage train_variant --variant full_driver_coupled \
      --wm_checkpoint /path/to/wm.pth \
      --student_checkpoint /path/to/student.pth \
      --train_dir /path/to/train

  # 信息因子分析
  python3 script/run_e3_dual_path_ablation.py \
      --stage info_factor --factor History_Full \
      --train_dir /path/to/train --val_dir /path/to/val
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
    DriveGuardWM, PathBEncoder, PathADecoder, PathBDecoder,
    ProcessMoE, LocalSpatialSSM, DriveInterface, Controller,
    MatrixStudent,
)
from dataset.temporal_dataset import TemporalDownscaleDataset


class E3DualPathAblation:
    """
    E3: Dual-Path Ablation — 分析历史 vs 当前信息的迁移性差异。

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
        self.epochs = self.config.get('epochs', 50)
        self.lr = self.config.get('lr', 1e-4)
        self.patience = self.config.get('patience', 10)
        self.num_workers = self.config.get('num_workers', 2)
        self.T = self.config.get('T', 5)
        self.beta = self.config.get('beta', 1.0)

        # 架构参数
        self.K = self.config.get('K', 4)
        self.N = self.config.get('N', 16)
        self.Dg = self.config.get('Dg', 32)
        self.Dm = self.config.get('Dm', 32)
        self.Dl = self.config.get('Dl', 32)
        self.Ds = self.config.get('Ds', 64)

        # 实验目录
        self.exp_name = exp_name or f"e3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join(self.output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0
        self.early_stop_counter = 0
        self.train_losses = []
        self.val_metrics = []

    # ════════════════════════════════════════════════════════════════
    # S5: Path B Training
    # ════════════════════════════════════════════════════════════════

    def train_path_b(self, wm_checkpoint, student_checkpoint,
                     train_dir, val_dir):
        """
        S5: 训练 Path B (MoE + SSM + decoder B)。

        Path A 和 Student 冻结。
        """
        print(f"\n{'='*60}")
        print("[E3] S5: Path B Training (MoE + SSM + Decoder)")
        print(f"{'='*60}")

        # 加载 World Model (冻结)
        wm = self._load_wm(wm_checkpoint)
        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        # 加载 Matrix Student (维度从 WM 结构推断)
        wm_kwargs = getattr(wm, 'model_kwargs', {})
        s_dg = wm_kwargs.get('Dg', self.Dg)
        s_dm = wm_kwargs.get('Dm', self.Dm)
        s_dl = wm_kwargs.get('Dl', self.Dl)
        s_in = wm_kwargs.get('in_channels', 7)
        student = MatrixStudent(
            global_dim=s_dg, message_dim=s_dm, local_dim=s_dl,
            in_channels=s_in, hidden_dim=self.Ds, relation_dim=64,
        ).to(self.device)
        s_ckpt = torch.load(student_checkpoint, map_location=self.device)
        student.load_state_dict(s_ckpt.get('model_state_dict', s_ckpt))
        student.eval()
        for p in student.parameters():
            p.requires_grad = False

        # 数据集
        train_ds = TemporalDownscaleDataset(
            train_dir, 'train', T=self.T, normalize=True)
        val_ds = TemporalDownscaleDataset(
            val_dir, 'val', T=self.T, normalize=True)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)

        # Path B 组件 (从 wm 中提取)
        path_b_encoder = wm.path_b_encoder
        drive_interface = wm.drive_interface
        process_moe = wm.process_moe
        local_ssm = wm.local_ssm
        path_b_decoder = wm.path_b_decoder

        # 冻结 World Model 部分(已冻结), 这里仅训练 Path B 新增参数
        path_b_params = (
            list(path_b_encoder.parameters()) +
            list(drive_interface.parameters()) +
            list(process_moe.parameters()) +
            list(local_ssm.parameters()) +
            list(path_b_decoder.parameters())
        )

        # 重新开启 Path B 参数的梯度 (先前被 wm 全局 freeze 覆盖)
        for p in path_b_params:
            p.requires_grad = True

        optimizer = optim.Adam(path_b_params, lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        n_params = sum(p.numel() for p in path_b_params)
        print(f"[E3] Path B trainable params: {n_params:,}")

        # ─── 训练循环 ─────────────────────────────────────────
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            wm.path_b_encoder.train()
            wm.drive_interface.train()
            wm.process_moe.train()
            wm.local_ssm.train()
            wm.path_b_decoder.train()

            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f'PathB Epoch {epoch}')
            for batch in pbar:
                lr_seq = batch['lr'].to(self.device)
                hr_seq = batch['hr'].to(self.device)
                land_mask = batch['land_mask'].to(self.device)

                optimizer.zero_grad()

                B, T, C, H_lr, W_lr = lr_seq.shape
                Hp, Wp = H_lr // 4, W_lr // 4
                target_size = (hr_seq.shape[3], hr_seq.shape[4])
                x_now = lr_seq[:, -1]

                # WM prior rollout (冻结)
                with torch.no_grad():
                    g_obs_list, l_obs_list, assignment_list = [], [], []
                    for t in range(T):
                        g_obs, l_obs, assignment, _ = wm._encode_frame(lr_seq[:, t])
                        g_obs_list.append(g_obs)
                        l_obs_list.append(l_obs)
                        assignment_list.append(assignment)
                    g_obs_seq = torch.stack(g_obs_list, dim=1)
                    l_obs_seq = torch.stack(l_obs_list, dim=1)
                    assignment_seq = torch.stack(assignment_list, dim=1)
                    prior_states = wm._rollout_prior(
                        g_obs_seq, l_obs_seq, assignment_seq, Hp, Wp)
                    last_state = prior_states[-1]

                    # Student D_hat
                    d_logits, _ = student(
                        last_state.G, last_state.M, last_state.L, x_now)
                    d_hat = F.softplus(d_logits).detach()

                # Path B forward
                y_b = wm._decode_path_b(last_state, x_now, d_hat, Hp, Wp, target_size)

                # Loss
                mask = land_mask.unsqueeze(1)
                diff = (y_b - hr_seq[:, -1]) * mask
                loss = diff.square().sum() / (mask.sum() + 1e-8)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / max(num_batches, 1)
            print(f"  PathB Epoch {epoch}: loss={avg_loss:.4f}")

        path_b_path = os.path.join(self.exp_dir, 'path_b_trained.pth')
        torch.save({
            'path_b_encoder': wm.path_b_encoder.state_dict(),
            'drive_interface': wm.drive_interface.state_dict(),
            'process_moe': wm.process_moe.state_dict(),
            'local_ssm': wm.local_ssm.state_dict(),
            'path_b_decoder': wm.path_b_decoder.state_dict(),
        }, path_b_path)
        print(f"[E3] Path B saved: {path_b_path}")
        return path_b_path

    # ════════════════════════════════════════════════════════════════
    # 信息因子变体
    # ════════════════════════════════════════════════════════════════

    def train_info_factor(self, factor, train_dir, val_dir):
        """
        训练信息因子变体。

        factor: Current_Coarse / Current_Full / History_Coarse / History_Full
        """
        print(f"\n[E3] Training info factor: {factor}")

        # 根据因子构建不同的模型配置
        use_history = 'History' in factor
        use_fine_current = 'Full' in factor

        # 简化版: 使用不同输入配置的 World Model
        config_override = {}
        if not use_history:
            config_override['T'] = 1  # 单帧
        if not use_fine_current:
            config_override['in_channels'] = 4  # 少通道

        trainer_cfg = {**self.config, **config_override}
        print(f"  Config: use_history={use_history}, fine_current={use_fine_current}")

        # 使用基础的 World Model 训练器
        from script.trainer_wm import WorldModelTrainer
        exp_name = f"e3_{factor.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trainer = WorldModelTrainer(
            train_dir=train_dir,
            val_dir=val_dir or train_dir,
            T=trainer_cfg.get('T', self.T),
            output_dir=self.output_dir,
            config=trainer_cfg,
            exp_name=exp_name,
        )
        result = trainer.train()
        return result

    # ════════════════════════════════════════════════════════════════
    # 双路径比较表
    # ════════════════════════════════════════════════════════════════

    def compare_dual_paths(self, wm_checkpoint, student_checkpoint,
                            path_b_checkpoint, test_dir):
        """
        运行所有双路径变体并记录性能。

        Variants:
          1. Path A only
          2. current-LR local encoder only
          3. Path A + Path B w/o driving
          4. free feature fusion
          5. random driving
          6. full driver-coupled dual path
          7. widened single path
        """
        print(f"\n{'='*60}")
        print("[E3] Dual-Path Comparison")
        print(f"{'='*60}")

        wm = self._load_wm(wm_checkpoint)
        wm.eval()

        # 加载 Path B 权重
        if path_b_checkpoint:
            pb_ckpt = torch.load(path_b_checkpoint, map_location=self.device)
            wm.path_b_encoder.load_state_dict(pb_ckpt['path_b_encoder'])
            wm.drive_interface.load_state_dict(pb_ckpt['drive_interface'])
            wm.process_moe.load_state_dict(pb_ckpt['process_moe'])
            wm.local_ssm.load_state_dict(pb_ckpt['local_ssm'])
            wm.path_b_decoder.load_state_dict(pb_ckpt['path_b_decoder'])

        test_ds = TemporalDownscaleDataset(
            test_dir, 'val', T=self.T, normalize=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=2, shuffle=False, num_workers=2)

        results = {}

        @torch.no_grad()
        def forward_variant(variant, wm, lr_seq, hr_seq, target_size):
            """计算单个变体的预测。"""
            B, T, C, H_lr, W_lr = lr_seq.shape
            Hp, Wp = H_lr // 4, W_lr // 4
            x_now = lr_seq[:, -1]

            # 通用 forward: 获取 prior + y_a + y_b + gate
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

            if variant == 'path_a_only':
                return y_a
            if variant == 'current_local_encoder':
                # 只用 Path B encoder + decoder (无 WM prior)
                s = wm.path_b_encoder(x_now, Hp, Wp)
                y_b_only = wm.path_b_decoder(s, target_size)
                return y_b_only
            if variant == 'path_b_only':
                d_logits, _ = wm.matrix_student(last_state.G, last_state.M, last_state.L, x_now)
                d_hat = F.softplus(d_logits)
                y_b = wm._decode_path_b(last_state, x_now, d_hat, Hp, Wp, target_size)
                return y_b
            if variant == 'unguided_b':
                # Path A + Path B 无 driving (d_hat = 0)
                Bsz, K, N = last_state.G.feature.shape[0], last_state.G.feature.shape[1], last_state.L.feature.shape[1]
                d_zero = torch.zeros(Bsz, K, N, device=lr_seq.device)
                y_b = wm._decode_path_b(last_state, x_now, d_zero, Hp, Wp, target_size)
                return 0.5 * (y_a + y_b)
            if variant == 'free_fusion':
                # 固定 0.5 加权融合
                d_logits, _ = wm.matrix_student(last_state.G, last_state.M, last_state.L, x_now)
                d_hat = F.softplus(d_logits)
                y_b = wm._decode_path_b(last_state, x_now, d_hat, Hp, Wp, target_size)
                return 0.5 * (y_a + y_b)
            if variant == 'random_driving':
                # 随机 driving
                Bsz, K, N = last_state.G.feature.shape[0], last_state.G.feature.shape[1], last_state.L.feature.shape[1]
                d_rand = torch.rand(Bsz, K, N, device=lr_seq.device)
                y_b = wm._decode_path_b(last_state, x_now, d_rand, Hp, Wp, target_size)
                return y_b
            # full_driver_coupled (默认)
            d_logits, r_raw = wm.matrix_student(last_state.G, last_state.M, last_state.L, x_now)
            d_hat = F.softplus(d_logits)
            r_hat = F.softplus(r_raw)
            y_b = wm._decode_path_b(last_state, x_now, d_hat, Hp, Wp, target_size)
            gate, y_hat = wm.controller(d_hat, r_hat, last_state.assignment,
                                        Hp, Wp, target_size[0], target_size[1], y_a, y_b)
            return y_hat

        variants = ['path_a_only', 'current_local_encoder', 'path_b_only',
                    'unguided_b', 'free_fusion', 'random_driving', 'full_driver_coupled']
        for variant in variants:
            mae_scores = []
            for batch in tqdm(test_loader, desc=f'[E3] {variant}'):
                lr_seq = batch['lr'].to(self.device)
                hr_seq = batch['hr'].to(self.device)

                target_size = (hr_seq.shape[3], hr_seq.shape[4])
                y_hat = forward_variant(variant, wm, lr_seq, hr_seq, target_size)

                mae = F.l1_loss(y_hat, hr_seq[:, -1]).item()
                mae_scores.append(mae)

            results[variant] = {
                'mae': float(np.mean(mae_scores)),
                'mae_std': float(np.std(mae_scores)),
            }
            print(f"  {variant}: MAE={results[variant]['mae']:.4f}")

        results_path = os.path.join(self.exp_dir, 'dual_path_comparison.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _load_wm(self, checkpoint_path):
        from models.driveguard_wm.loader import load_model_from_checkpoint
        return load_model_from_checkpoint(
            checkpoint_path, device=self.device, strict=False)


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="E3: Dual-Path Ablation")

    parser.add_argument('--stage', type=str, required=True,
                        choices=['train_path_b', 'info_factor', 'compare'],
                        help='阶段')
    parser.add_argument('--variant', type=str, default=None,
                        help='变体名称 (info_factor 时用)')
    parser.add_argument('--factor', type=str, default=None,
                        help='信息因子: Current_Coarse/Current_Full/History_Coarse/History_Full')

    parser.add_argument('--wm_checkpoint', type=str, default=None)
    parser.add_argument('--student_checkpoint', type=str, default=None)
    parser.add_argument('--path_b_checkpoint', type=str, default=None)

    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
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

    ablation = E3DualPathAblation(
        output_dir=args.output_dir, device=args.device, config=config, exp_name=args.exp_name)

    if args.stage == 'train_path_b':
        assert args.wm_checkpoint and args.student_checkpoint and args.train_dir
        ablation.train_path_b(args.wm_checkpoint, args.student_checkpoint,
                              args.train_dir, args.val_dir or args.train_dir)

    elif args.stage == 'info_factor':
        assert args.factor and args.train_dir
        ablation.train_info_factor(args.factor, args.train_dir, args.val_dir)

    elif args.stage == 'compare':
        assert args.wm_checkpoint and args.test_dir
        ablation.compare_dual_paths(args.wm_checkpoint, args.student_checkpoint,
                                    args.path_b_checkpoint, args.test_dir)

    print(f"\n[E3] Stage '{args.stage}' complete.")


if __name__ == "__main__":
    main()
