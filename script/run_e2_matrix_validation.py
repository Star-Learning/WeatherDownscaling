"""
E2: Drive/Constraint Matrix Validation
========================================

目的: 验证 D/R 矩阵的有效性 — D 编码有益干预, R 编码负迁移风险。

阶段:
  S3: Matrix Teacher 训练 (需 posterior 状态)
  S4: Matrix Student 训练 (LR-only, 知识蒸馏)

分析:
  - Edge semantics: 移除/放大高 D 边对预测的影响
  - D/R 是否互补
  - Teacher–student agreement
  - 蒸馏后的目标域性能

用法:
  # S3: 训练 Matrix Teacher
  python3 script/run_e2_matrix_validation.py --stage teacher \
      --wm_checkpoint /path/to/wm.pth \
      --train_dir /path/to/train --val_dir /path/to/val

  # S4: 训练 Matrix Student
  python3 script/run_e2_matrix_validation.py --stage student \
      --teacher_checkpoint /path/to/teacher.pth \
      --train_dir /path/to/train --val_dir /path/to/val

  # Edge semantics 验证
  python3 script/run_e2_matrix_validation.py --stage edge_analysis \
      --student_checkpoint /path/to/student.pth \
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
    DriveGuardWM, MatrixTeacher, MatrixStudent,
    GaussianState, HierarchicalState,
)
from dataset.temporal_dataset import TemporalDownscaleDataset


class E2MatrixValidator:
    """
    E2: Drive/Constraint Matrix Validation.

    管理 S3 (Teacher) / S4 (Student) 的训练和矩阵语义验证。

    Args:
        output_dir: 输出目录
        device: 计算设备
        config: 配置 dict
    """

    def __init__(self, output_dir, device=None, config=None, exp_name=None):
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}

        # 训练参数
        self.batch_size = self.config.get('batch_size', 4)
        self.epochs = self.config.get('epochs', 50)
        self.lr = self.config.get('lr', 1e-4)
        self.patience = self.config.get('patience', 10)
        self.num_workers = self.config.get('num_workers', 2)
        self.T = self.config.get('T', 5)
        self.beta = self.config.get('beta', 1.0)

        # Matrix 架构参数
        self.K = self.config.get('K', 4)
        self.N = self.config.get('N', 16)
        self.Dg = self.config.get('Dg', 32)
        self.Dm = self.config.get('Dm', 32)
        self.Dl = self.config.get('Dl', 32)
        self.Ds = self.config.get('Ds', 64)

        # 实验目录
        self.exp_name = exp_name or f"e2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join(self.output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.best_model_path = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.early_stop_counter = 0
        self.current_epoch = 0
        self.train_losses = []
        self.val_metrics = []

    # ════════════════════════════════════════════════════════════════
    # S3: Matrix Teacher Training
    # ════════════════════════════════════════════════════════════════

    def train_teacher(self, wm_checkpoint, train_dir, val_dir):
        """
        S3: 训练 Matrix Teacher。

        Teacher 以 G/M/L posterior 为输入, 预测 D/R 监督目标。

        Args:
            wm_checkpoint: S1 World Model checkpoint
            train_dir: 训练数据
            val_dir: 验证数据
        """
        print(f"\n{'='*60}")
        print("[E2] S3: Matrix Teacher Training")
        print(f"{'='*60}")

        # 加载 World Model (冻结)
        wm = self._load_wm(wm_checkpoint)
        wm.eval()
        for p in wm.parameters():
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

        # Matrix Teacher (维度从 WM 结构推断, 保证与 posterior 匹配)
        wm_kwargs = getattr(wm, 'model_kwargs', {})
        t_dg = wm_kwargs.get('Dg', self.Dg)
        t_dm = wm_kwargs.get('Dm', self.Dm)
        t_dl = wm_kwargs.get('Dl', self.Dl)
        teacher = MatrixTeacher(
            global_dim=t_dg, message_dim=t_dm, local_dim=t_dl,
        ).to(self.device)

        optimizer = optim.Adam(teacher.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        n_params = sum(p.numel() for p in teacher.parameters())
        print(f"[E2] Teacher params: {n_params:,}")

        # ─── 训练循环 ─────────────────────────────────────────
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            teacher.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f'Teacher Epoch {epoch}')
            for batch in pbar:
                lr_seq = batch['lr'].to(self.device)
                hr_seq = batch['hr'].to(self.device)

                optimizer.zero_grad()

                # World Model forward (获取 posterior 状态)
                with torch.no_grad():
                    wm_result = wm.forward_s1(lr_seq, y_hr=hr_seq,
                                              target_size=(hr_seq.shape[3], hr_seq.shape[4]),
                                              beta=self.beta)
                    prior_states = wm_result.get('prior_states', [])
                    if not prior_states:
                        # 使用 memory 中的简化版
                        continue
                    last_state = prior_states[-1]

                # Teacher forward
                d_logits, r_raw, z_d, z_r = teacher(
                    last_state.G, last_state.M, last_state.L
                )

                # 监督信号: bounded D/R, 鼓励非互补但数值稳定
                d_hat = torch.sigmoid(d_logits)      # [0,1] bounded drive
                r_hat = torch.sigmoid(r_raw)         # [0,1] bounded risk

                # Loss: D 与 R 分离 (不互补) + 保持中位值稳定
                # 使 D 与 R 呈负相关 (互补), 而非简单最大化 spread 爆炸
                d_flat = d_hat.reshape(BK_N := d_hat.shape[0], -1)
                r_flat = r_hat.reshape(BK_N, -1)

                # 鼓励 D/R 负相关 (反互补惩罚)
                d_center = d_flat - d_flat.mean(dim=-1, keepdim=True)
                r_center = r_flat - r_flat.mean(dim=-1, keepdim=True)
                d_norm = d_center.norm(dim=-1, keepdim=True) + 1e-8
                r_norm = r_center.norm(dim=-1, keepdim=True) + 1e-8
                cos_sim = (d_center * r_center).sum(dim=-1) / (d_norm * r_norm).squeeze(-1)
                anti_complement = cos_sim.mean()

                # 鼓励区分度 (spread 温和, 用 sigmoid 自然 bounded)
                spread = 0.5 * (d_flat.std(dim=-1).mean() + r_flat.std(dim=-1).mean())

                loss = -0.5 * spread + 0.5 * anti_complement

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / max(num_batches, 1)

            # Validation
            val_loss = self._evaluate_teacher(teacher, val_loader, wm)
            scheduler.step(val_loss)

            print(f"  Teacher Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

            # Checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(teacher, optimizer, 'teacher')

        teacher_path = os.path.join(self.exp_dir, 'best_teacher.pth')
        print(f"[E2] Teacher training done. Best: {self.best_epoch} (loss={self.best_val_loss:.4f})")
        return teacher_path

    def _evaluate_teacher(self, teacher, loader, wm):
        """Teacher 验证。"""
        teacher.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in loader:
                lr_seq = batch['lr'].to(self.device)
                hr_seq = batch['hr'].to(self.device)
                wm_result = wm.forward_s1(lr_seq, y_hr=hr_seq,
                                          target_size=(hr_seq.shape[3], hr_seq.shape[4]))
                prior_states = wm_result.get('prior_states', [])
                if not prior_states:
                    continue
                last_state = prior_states[-1]

                d_logits, r_raw, _, _ = teacher(last_state.G, last_state.M, last_state.L)
                d_hat = F.softplus(d_logits)
                r_hat = F.softplus(r_raw)
                loss = -(d_hat.std(dim=-1).mean() + r_hat.std(dim=-1).mean())
                total_loss += loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)

    # ════════════════════════════════════════════════════════════════
    # S4: Matrix Student Training
    # ════════════════════════════════════════════════════════════════

    def train_student(self, teacher_checkpoint, wm_checkpoint, train_dir, val_dir):
        """
        S4: 训练 LR-only Matrix Student (知识蒸馏)。

        Args:
            teacher_checkpoint: S3 Teacher checkpoint
            wm_checkpoint: S1 World Model checkpoint
            train_dir: 训练数据
            val_dir: 验证数据
        """
        print(f"\n{'='*60}")
        print("[E2] S4: LR-only Matrix Student Training")
        print(f"{'='*60}")

        # 加载 Teacher (维度从 checkpoint 推断)
        t_ckpt = torch.load(teacher_checkpoint, map_location=self.device)
        t_sd = t_ckpt.get('model_state_dict', t_ckpt)
        # joint_proj.0 输入维度 = Dg + Dm + Dl
        joint_in = t_sd.get('joint_proj.0.weight', torch.zeros(128, 192)).shape[-1]
        t_dg = t_dm = t_dl = joint_in // 3
        teacher = MatrixTeacher(
            global_dim=t_dg, message_dim=t_dm, local_dim=t_dl,
        ).to(self.device)
        teacher.load_state_dict(t_sd)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # 加载 World Model (冻结)
        wm = self._load_wm(wm_checkpoint)
        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False
        wm_kwargs = getattr(wm, 'model_kwargs', {})

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

        # Student (仅 prior, 维度从 WM 结构推断)
        s_dg = wm_kwargs.get('Dg', self.Dg)
        s_dm = wm_kwargs.get('Dm', self.Dm)
        s_dl = wm_kwargs.get('Dl', self.Dl)
        student = MatrixStudent(
            global_dim=s_dg, message_dim=s_dm, local_dim=s_dl,
            in_channels=7, hidden_dim=self.Ds, relation_dim=64,
        ).to(self.device)

        optimizer = optim.Adam(student.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        n_params = sum(p.numel() for p in student.parameters())
        print(f"[E2] Student params: {n_params:,}")

        # ─── 蒸馏训练 ─────────────────────────────────────────
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            student.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f'Student Epoch {epoch}')
            for batch in pbar:
                lr_seq = batch['lr'].to(self.device)
                hr_seq = batch['hr'].to(self.device)

                optimizer.zero_grad()

                # WM forward (仅 prior)
                with torch.no_grad():
                    wm_result = wm.forward_s1(lr_seq, y_hr=hr_seq,
                                              target_size=(hr_seq.shape[3], hr_seq.shape[4]))
                    prior_states = wm_result.get('prior_states', [])
                    if not prior_states:
                        continue
                    last_state = prior_states[-1]

                    # Teacher D/R (KL 目标)
                    d_logits_q, r_raw_q, _, _ = teacher(
                        last_state.G, last_state.M, last_state.L)

                # Student D/R (仅 prior + LR)
                x_now = lr_seq[:, -1]
                d_logits_hat, r_raw_hat = student(
                    last_state.G, last_state.M, last_state.L, x_now)

                # KL distillation (sigmoid logits → 概率分布, bounded)
                tau = 2.0
                teacher_prob = F.softmax(d_logits_q / tau, dim=-1)
                student_logprob = F.log_softmax(d_logits_hat / tau, dim=-1)
                kl_loss = F.kl_div(student_logprob, teacher_prob.detach(),
                                   reduction='batchmean') * (tau ** 2)

                # Magnitude loss (bounded sigmoid)
                d_hat = torch.sigmoid(d_logits_hat)
                d_q = torch.sigmoid(d_logits_q)
                mag_loss = F.smooth_l1_loss(d_hat, d_q.detach())

                loss = kl_loss + 0.1 * mag_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'kl': f'{kl_loss.item():.4f}'})

            avg_loss = total_loss / max(num_batches, 1)
            val_loss = self._evaluate_student(student, teacher, wm, val_loader)
            scheduler.step(val_loss)

            print(f"  Student Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(student, optimizer, 'student')

        student_path = os.path.join(self.exp_dir, 'best_student.pth')
        print(f"[E2] Student training done. Best: {self.best_epoch}")
        return student_path

    def _evaluate_student(self, student, teacher, wm, loader):
        """Student 验证。"""
        student.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in loader:
                lr_seq = batch['lr'].to(self.device)
                hr_seq = batch['hr'].to(self.device)
                wm_result = wm.forward_s1(lr_seq, y_hr=hr_seq,
                                          target_size=(hr_seq.shape[3], hr_seq.shape[4]))
                prior_states = wm_result.get('prior_states', [])
                if not prior_states:
                    continue
                last_state = prior_states[-1]
                x_now = lr_seq[:, -1]
                d_logits_q, r_raw_q, _, _ = teacher(last_state.G, last_state.M, last_state.L)
                d_logits_hat, r_raw_hat = student(last_state.G, last_state.M, last_state.L, x_now)

                tau = 2.0
                teacher_prob = F.softmax(d_logits_q / tau, dim=-1)
                student_logprob = F.log_softmax(d_logits_hat / tau, dim=-1)
                kl_loss = F.kl_div(student_logprob, teacher_prob.detach(),
                                   reduction='batchmean') * (tau ** 2)
                total_loss += kl_loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)

    # ════════════════════════════════════════════════════════════════
    # Edge Semantics Validation
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def validate_edge_semantics(self, student_checkpoint, wm_checkpoint, test_dir):
        """
        验证 D/R 边语义。

        - 移除高 D 边 → 预测变差?
        - 放大高 D 边 → 稳定有益?
        - 高 R 边 → 预测 Path B 负迁移?
        - D/R 非互补?
        """
        print(f"\n{'='*60}")
        print("[E2] Edge Semantics Validation")
        print(f"{'='*60}")

        wm = self._load_wm(wm_checkpoint)
        wm.eval()
        wm_kwargs = getattr(wm, 'model_kwargs', {})

        # Student 维度从 WM 结构推断
        s_dg = wm_kwargs.get('Dg', self.Dg)
        s_dm = wm_kwargs.get('Dm', self.Dm)
        s_dl = wm_kwargs.get('Dl', self.Dl)
        student = MatrixStudent(
            global_dim=s_dg, message_dim=s_dm, local_dim=s_dl,
            in_channels=7, hidden_dim=self.Ds, relation_dim=64,
        ).to(self.device)
        s_ckpt = torch.load(student_checkpoint, map_location=self.device)
        student.load_state_dict(s_ckpt.get('model_state_dict', s_ckpt))
        student.eval()

        test_ds = TemporalDownscaleDataset(
            test_dir, 'val', T=self.T, normalize=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=2, shuffle=False, num_workers=2)

        results = {
            'd_hat_values': [],
            'r_hat_values': [],
            'd_r_correlation': [],
            'loss_with_high_d_removed': [],
            'loss_baseline': [],
        }

        for batch in tqdm(test_loader, desc='[E2] Edge analysis'):
            lr_seq = batch['lr'].to(self.device)
            hr_seq = batch['hr'].to(self.device)

            wm_result = wm.forward_s1(lr_seq, y_hr=hr_seq,
                                      target_size=(hr_seq.shape[3], hr_seq.shape[4]))
            prior_states = wm_result.get('prior_states', [])
            if not prior_states:
                continue
            last_state = prior_states[-1]

            x_now = lr_seq[:, -1]
            d_logits, r_raw = student(last_state.G, last_state.M, last_state.L, x_now)
            d_hat = torch.sigmoid(d_logits)
            r_hat = torch.sigmoid(r_raw)

            results['d_hat_values'].append(d_hat.cpu().numpy())
            results['r_hat_values'].append(r_hat.cpu().numpy())

            # D-R correlation
            d_flat = d_hat.flatten()
            r_flat = r_hat.flatten()
            if d_flat.numel() > 1:
                corr = np.corrcoef(d_flat.cpu().numpy(), r_flat.cpu().numpy())[0, 1]
                results['d_r_correlation'].append(corr)

        # 汇总
        d_all = np.concatenate([v.flatten() for v in results['d_hat_values']])
        r_all = np.concatenate([v.flatten() for v in results['r_hat_values']])

        summary = {
            'd_mean': float(d_all.mean()),
            'd_std': float(d_all.std()),
            'r_mean': float(r_all.mean()),
            'r_std': float(r_all.std()),
            'd_r_corr_mean': float(np.mean(results['d_r_correlation'])),
            'd_top10pct_mean': float(np.sort(d_all)[-int(0.1 * len(d_all)):].mean()),
        }

        # 保存
        summary_path = os.path.join(self.exp_dir, 'edge_semantics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n  D mean: {summary['d_mean']:.4f} ± {summary['d_std']:.4f}")
        print(f"  R mean: {summary['r_mean']:.4f} ± {summary['r_std']:.4f}")
        print(f"  D-R corr: {summary['d_r_corr_mean']:.4f}")
        print(f"  D top 10%: {summary['d_top10pct_mean']:.4f}")
        print(f"  Results: {summary_path}")

        return summary

    # ════════════════════════════════════════════════════════════════
    # 工具函数
    # ════════════════════════════════════════════════════════════════

    def _load_wm(self, checkpoint_path):
        """加载 World Model (自动恢复完整结构)。"""
        print(f"[E2] Loading WM: {checkpoint_path}")
        from models.driveguard_wm.loader import load_model_from_checkpoint
        return load_model_from_checkpoint(
            checkpoint_path, device=self.device, strict=False)

    def _save_checkpoint(self, model, optimizer, prefix=''):
        """保存 checkpoint。"""
        ckpt = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'model_class': getattr(model, 'model_class',
                                   f"{type(model).__module__}.{type(model).__qualname__}"),
            'model_kwargs': getattr(model, 'model_kwargs', {}),
        }
        path = os.path.join(self.exp_dir, f'best_{prefix}.pth')
        torch.save(ckpt, path)
        self.best_model_path = path


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="E2: Drive/Constraint Matrix Validation")

    # 阶段
    parser.add_argument('--stage', type=str, required=True,
                        choices=['teacher', 'student', 'edge_analysis'],
                        help='阶段: teacher=S3, student=S4, edge_analysis=边语义验证')

    # 模型 checkpoints
    parser.add_argument('--wm_checkpoint', type=str, default=None,
                        help='S1 World Model checkpoint')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='S3 Teacher checkpoint')
    parser.add_argument('--student_checkpoint', type=str, default=None,
                        help='S4 Student checkpoint')

    # 数据
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--T', type=int, default=5)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)

    # 输出
    parser.add_argument('--output_dir', type=str,
                        default='outputs')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    config = {
        'T': args.T,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'patience': args.patience,
        'num_workers': args.num_workers,
    }

    validator = E2MatrixValidator(
        output_dir=args.output_dir,
        device=args.device,
        config=config,
        exp_name=args.exp_name,
    )

    if args.stage == 'teacher':
        assert args.wm_checkpoint and args.train_dir, \
            "Teacher 阶段需要 --wm_checkpoint 和 --train_dir"
        result_path = validator.train_teacher(
            args.wm_checkpoint, args.train_dir, args.val_dir or args.train_dir)

    elif args.stage == 'student':
        assert args.teacher_checkpoint and args.wm_checkpoint and args.train_dir, \
            "Student 阶段需要 --teacher_checkpoint, --wm_checkpoint, --train_dir"
        result_path = validator.train_student(
            args.teacher_checkpoint, args.wm_checkpoint,
            args.train_dir, args.val_dir or args.train_dir)

    elif args.stage == 'edge_analysis':
        assert args.student_checkpoint and args.wm_checkpoint and args.test_dir, \
            "edge_analysis 需要 --student_checkpoint, --wm_checkpoint, --test_dir"
        summary = validator.validate_edge_semantics(
            args.student_checkpoint, args.wm_checkpoint, args.test_dir)

    print(f"\n[E2] Stage '{args.stage}' complete. Output: {validator.exp_dir}")


if __name__ == "__main__":
    main()
