"""
DriveGuard-WM Trainer — World Model Stage 1 (S1) 训练器
======================================================

S1 训练: Tokenizer + G/M/L prior/posterior + Path A decoder

损失: recon_loss (MSE) + β · KL divergence

用法:
  python3 script/trainer_driveguard_wm.py \
      --train_dir /path/to/aust_141_183_778_896_7c_wm_T5 \
      --val_dir /path/to/aust_141_183_778_896_7c_wm_T5 \
      --T 5 --epochs 100

迁移学习:
  --train_dir conus_7c ... --test_dir aust_7c ...
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
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 确保能找到项目模块
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.driveguard_wm import DriveGuardWM
from dataset.temporal_dataset import TemporalDownscaleDataset


class DriveGuardWMTrainer:
    """
    DriveGuard-WM Stage 1 训练器。

    S1 训练轮次：
      Tokenizer → G/M/L prior → Posterior → Path A Decoder
      损失: recon + β · KL

    Args:
        train_dir: 训练数据集路径
        val_dir:   验证数据集路径
        T:         时序窗口长度 (default: 5)
        output_dir: 输出目录
        test_dir:  测试数据集路径 (迁移学习, 可选)
        device:    设备
        config:    配置 dict
        exp_name:  实验名称 (可选)
        resume:    checkpoint 路径 (可选)
    """

    def __init__(self, train_dir, val_dir, output_dir, T=5,
                 test_dir=None, device=None, config=None, exp_name=None,
                 resume=None):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.T = T
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # ─── 配置 ───
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 8)
        self.epochs = self.config.get('epochs', 100)
        self.lr = self.config.get('lr', 1e-4)
        self.patience = self.config.get('patience', 10)
        self.num_workers = self.config.get('num_workers', 4)
        self.normalize = self.config.get('normalize', True)
        self.beta = self.config.get('beta', 1.0)
        self.beta_warmup = self.config.get('beta_warmup', 20)
        self.use_single_frame = self.config.get('use_single_frame', False)

        # ─── DriveGuard-WM 架构参数 ───
        self.K = self.config.get('global_tokens', 8)
        self.N = self.config.get('local_tokens', 64)
        self.Dg = self.config.get('global_dim', 64)
        self.Dm = self.config.get('message_dim', 64)
        self.Dl = self.config.get('local_dim', 64)
        self.Ds = self.config.get('spatial_dim', 128)
        self.Dc = self.config.get('condition_dim', 64)
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.num_experts = self.config.get('num_experts', 4)
        self.ssm_dim = self.config.get('ssm_dim', 64)
        self.relation_dim = self.config.get('relation_dim', 64)

        # ─── 实验目录 ───
        if exp_name is None:
            self.exp_name = f"dgwm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.exp_name = exp_name
        self.exp_dir = os.path.join(output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # ─── 数据集 ───
        print(f"[DGWM] Loading datasets (T={T})...")
        self.train_dataset = TemporalDownscaleDataset(
            train_dir, 'train', T=T, normalize=self.normalize)
        self.val_dataset = TemporalDownscaleDataset(
            val_dir, 'val', T=T, normalize=self.normalize)
        self.test_dataset = None
        if test_dir:
            self.test_dataset = TemporalDownscaleDataset(
                test_dir, 'val', T=T, normalize=self.normalize)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True)
        self.test_loader = None
        if self.test_dataset:
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers, pin_memory=True)

        # HR 尺寸
        self.target_h = self.val_dataset.hr_h
        self.target_w = self.val_dataset.hr_w
        print(f"[DGWM] Target HR size: {self.target_h}×{self.target_w}")

        # LR 输入通道数
        self.in_channels = self.train_dataset.n_channels
        print(f"[DGWM] LR input channels: {self.in_channels}")

        # ─── 模型 ───
        self.model = DriveGuardWM(
            in_channels=self.in_channels,
            K=self.K, N=self.N,
            Dg=self.Dg, Dm=self.Dm, Dl=self.Dl,
            Ds=self.Ds, Dc=self.Dc,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            ssm_dim=self.ssm_dim,
            relation_dim=self.relation_dim,
            beta=self.beta,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[DGWM] Model parameters: {n_params:,}")
        print(f"[DGWM]   K={self.K}, N={self.N}, Dg={self.Dg}, Dm={self.Dm}, Dl={self.Dl}")

        # ─── 优化器 + 调度器 ───
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # ─── 训练状态 ───
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.early_stop_counter = 0

        self.train_losses = []
        self.val_metrics = []
        self.test_metrics = []
        self.beta_history = []

        self.best_model_path = None
        self.log_csv_path = os.path.join(self.exp_dir, 'train_log.csv')

        if resume is not None:
            self._load_checkpoint(resume)

        self._save_config()

    def _save_config(self):
        """保存完整配置到 JSON (含所有模型构架参数)。"""
        # 从 model.model_kwargs 获取完整参数，确保与模型构造一致
        model_kwargs = getattr(self.model, 'model_kwargs', {})
        full_config = {
            'train_dir': self.train_dir,
            'val_dir': self.val_dir,
            'test_dir': self.test_dir,
            'T': self.T,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'patience': self.patience,
            'num_workers': self.num_workers,
            'normalize': self.normalize,
            'beta': self.beta,
            'beta_warmup': self.beta_warmup,
            'in_channels': self.in_channels,
            'model': 'DriveGuardWM',
            'model_class': getattr(self.model, 'model_class',
                                   'models.driveguard_wm.DriveGuardWM'),
            'stage': 'S1',
            # 完整架构参数 (与 model_kwargs 同步)
            **model_kwargs,
        }
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
        print(f"[DGWM] Config saved to {config_path}")

    def _load_checkpoint(self, checkpoint_path):
        """从 checkpoint 恢复训练状态。"""
        print(f"[DGWM] Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ckpt_epoch = checkpoint.get('epoch', -1)
            self.current_epoch = ckpt_epoch + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.early_stop_counter = 0
            print(f"[DGWM] Restored epoch={checkpoint.get('epoch','?')}, "
                  f"best_val_loss={self.best_val_loss:.6f}, "
                  f"best_epoch={self.best_epoch}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"[DGWM] Loaded state_dict (no training state)")

    def _get_beta(self):
        """KL annealing: 前 beta_warmup 个 epoch 从 0 线性增加到目标值。"""
        if self.current_epoch < self.beta_warmup:
            return self.beta * (self.current_epoch + 1) / self.beta_warmup
        return self.beta

    # ─── 训练 ──────────────────────────────────────────────────────────

    def train_epoch(self):
        """
        训练一个 epoch。

        数据:
          lr:  [B, T, C, H_lr, W_lr]
          hr:  [B, T, 1, H_hr, W_hr]

        DriveGuardWM.forward_s1 返回:
          hr_pred:     [B, 1, H_hr, W_hr]
          recon_loss:  MSE loss
          kl_loss:     KL divergence
          total_loss:  recon + β·kl
        """
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        beta = self._get_beta()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        for batch in pbar:
            lr_seq = batch['lr'].to(self.device)
            hr_seq = batch['hr'].to(self.device)
            land_mask = batch['land_mask'].to(self.device)

            self.optimizer.zero_grad()

            target_hw = (hr_seq.shape[3], hr_seq.shape[4])

            # 信息因子变体: use_single_frame=True 时只用当前帧 (无历史动力)
            if getattr(self, 'use_single_frame', False):
                lr_seq_f = lr_seq[:, -1:]   # [B, 1, C, H, W]
                hr_seq_f = hr_seq[:, -1:]
            else:
                lr_seq_f = lr_seq
                hr_seq_f = hr_seq

            result = self.model.forward_s1(
                lr_seq_f, y_hr=hr_seq_f,
                target_size=target_hw,
                beta=beta,
            )

            # Land-masked reconstruction loss
            mask = land_mask.unsqueeze(1)  # [B, 1, H_hr, W_hr]
            diff = (result['hr_pred'] - hr_seq[:, -1]) * mask
            recon = diff.square().sum() / (mask.sum() + 1e-8)

            loss = recon + result['kl_loss']
            # Note: result['total_loss'] already includes recon + β·kl,
            # but we recalc recon with land mask for consistency

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += result['kl_loss'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'recon': f'{recon.item():.6f}',
                'kl': f'{result["kl_loss"].item():.6f}',
                'β': f'{beta:.4f}',
            })

        self.beta_history.append(beta)

        return {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'kl': total_kl / num_batches,
        }

    # ─── 评估 ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader, dataset, name="Eval"):
        """
        评估模型。

        - 用训练模式（后验）计算 loss
        - 用推理模式计算 land-only 指标 (MSE, MAE, correlation)
        """
        self.model.eval()

        # Loss 统计
        total_recon = 0.0
        total_kl = 0.0
        total_loss = 0.0
        num_batches = 0

        # 指标统计 (land-only)
        sum_sq = 0.0
        sum_abs = 0.0
        sum_pred = 0.0
        sum_target = 0.0
        sum_pred_sq = 0.0
        sum_target_sq = 0.0
        sum_pred_target = 0.0
        count = 0

        pbar = tqdm(loader, desc=f'Epoch {self.current_epoch} [{name}]')
        for batch in pbar:
            lr_seq = batch['lr'].to(self.device)
            hr_seq = batch['hr'].to(self.device)
            land_mask = batch['land_mask'].to(self.device)

            B = lr_seq.shape[0]
            target_hw = (hr_seq.shape[3], hr_seq.shape[4])

            # 信息因子变体: 单帧时只喂当前帧
            if getattr(self, 'use_single_frame', False):
                lr_seq_f = lr_seq[:, -1:]
                hr_seq_f = hr_seq[:, -1:]
            else:
                lr_seq_f = lr_seq
                hr_seq_f = hr_seq

            # Loss (后验模式)
            result = self.model.forward_s1(
                lr_seq_f, y_hr=hr_seq_f,
                target_size=target_hw,
            )
            mask = land_mask.unsqueeze(1)
            diff = (result['hr_pred'] - hr_seq[:, -1]) * mask
            recon = diff.square().sum() / (mask.sum() + 1e-8)
            loss = recon + result['kl_loss']

            total_recon += recon.item()
            total_kl += result['kl_loss'].item()
            total_loss += loss.item()
            num_batches += 1

            # S1 validation must use the trained historical path only. The
            # Student, Path B, and Controller are trained in later stages.
            hr_pred = self.model.forward_s1(
                lr_seq_f, y_hr=None, target_size=target_hw
            )['hr_pred']
            hr_last = hr_seq[:, -1]

            pred_masked = hr_pred * mask
            target_masked = hr_last * mask

            p_flat = pred_masked.flatten()
            t_flat = target_masked.flatten()
            land_idx = mask.flatten() > 0
            p_land = p_flat[land_idx]
            t_land = t_flat[land_idx]
            n = p_land.numel()

            if n > 0:
                sum_sq += torch.sum((p_land - t_land) ** 2).item()
                sum_abs += torch.sum(torch.abs(p_land - t_land)).item()
                sum_pred += p_land.sum().item()
                sum_target += t_land.sum().item()
                sum_pred_sq += (p_land ** 2).sum().item()
                sum_target_sq += (t_land ** 2).sum().item()
                sum_pred_target += (p_land * t_land).sum().item()
                count += n

            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'recon': f'{recon.item():.6f}',
                'kl': f'{result["kl_loss"].item():.6f}',
            })

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches

        mse = sum_sq / count if count > 0 else 0
        mae = sum_abs / count if count > 0 else 0

        correlation = 0.0
        if count > 0:
            mean_pred = sum_pred / count
            mean_target = sum_target / count
            cov = (sum_pred_target / count) - (mean_pred * mean_target)
            var_pred = (sum_pred_sq / count) - (mean_pred ** 2)
            var_target = (sum_target_sq / count) - (mean_target ** 2)
            std_pred = np.sqrt(max(var_pred, 0))
            std_target = np.sqrt(max(var_target, 0))
            if std_pred > 0 and std_target > 0:
                correlation = cov / (std_pred * std_target)

        return {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'recon': avg_recon,
            'kl': avg_kl,
        }

    # ─── 检查点 ────────────────────────────────────────────────────────

    def save_checkpoint(self, is_best=False):
        """保存 checkpoint。"""
        if is_best:
            if (self.best_model_path is not None
                    and os.path.exists(self.best_model_path)):
                try:
                    os.remove(self.best_model_path)
                except Exception:
                    pass

            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch,
                'config': self.config,
                'model_class': getattr(self.model, 'model_class',
                                       'models.driveguard_wm.DriveGuardWM'),
                'model_kwargs': getattr(self.model, 'model_kwargs', {}),
            }

            if self.val_metrics:
                m = self.val_metrics[-1]
                metrics_str = (
                    f"dgwm_e{self.current_epoch}"
                    f"_loss{m['loss']:.6f}_mae{m['mae']:.6f}"
                )
            else:
                metrics_str = f"dgwm_e{self.current_epoch}"

            best_path = os.path.join(
                self.exp_dir, f'best_model_{metrics_str}.pth')

            def _save():
                torch.save(checkpoint, best_path)
                full_model_path = os.path.join(
                    self.exp_dir, 'model.pt')
                torch.save(self.model, full_model_path)

            threading.Thread(target=_save).start()
            self.best_model_path = best_path

    def save_log_to_csv(self):
        """保存训练日志到 CSV。"""
        data = {
            'epoch': list(range(len(self.train_losses))),
            'train_loss': [v['loss'] for v in self.train_losses],
            'train_recon': [v['recon'] for v in self.train_losses],
            'train_kl': [v['kl'] for v in self.train_losses],
            'val_loss': [v['loss'] for v in self.val_metrics],
            'val_mse': [v['mse'] for v in self.val_metrics],
            'val_mae': [v['mae'] for v in self.val_metrics],
            'val_corr': [v['correlation'] for v in self.val_metrics],
            'val_recon': [v['recon'] for v in self.val_metrics],
            'val_kl': [v['kl'] for v in self.val_metrics],
            'beta': self.beta_history[:len(self.train_losses)],
        }
        if self.test_metrics:
            data['test_loss'] = [v['loss'] for v in self.test_metrics]
            data['test_mse'] = [v['mse'] for v in self.test_metrics]
            data['test_mae'] = [v['mae'] for v in self.test_metrics]
            data['test_corr'] = [v['correlation'] for v in self.test_metrics]
        pd.DataFrame(data).to_csv(self.log_csv_path, index=False)

    # ─── 完整训练循环 ──────────────────────────────────────────────────

    def train(self):
        """完整训练流程。"""
        print(f"\n{'='*60}")
        print(f"[DGWM] DriveGuard-WM Stage 1 Training")
        print(f"[DGWM] Device: {self.device}")
        print(f"[DGWM] Output: {self.exp_dir}")
        print(f"[DGWM] T={self.T}, beta={self.beta}, beta_warmup={self.beta_warmup}")
        print(f"[DGWM] Train: {len(self.train_dataset)} windows")
        print(f"[DGWM] Val:   {len(self.val_dataset)} windows")
        if self.test_dataset:
            print(f"[DGWM] Test:  {len(self.test_dataset)} windows")
        print(f"[DGWM] Patience: {self.patience}")
        if self.current_epoch > 0:
            print(f"[DGWM] Resuming from epoch {self.current_epoch}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            train_result = self.train_epoch()
            self.train_losses.append(train_result)

            val_metrics = self.evaluate(
                self.val_loader, self.val_dataset, name="Val")
            self.val_metrics.append(val_metrics)

            test_metrics = None
            if self.test_loader is not None and self.test_dataset is not None:
                test_metrics = self.evaluate(
                    self.test_loader, self.test_dataset, name="Test")
                self.test_metrics.append(test_metrics)

            self.scheduler.step(val_metrics['loss'])

            # ─── 打印 ───
            print(f"\n[DGWM] Epoch {epoch}/{self.epochs-1}  "
                  f"(β={self.beta_history[-1]:.4f}):")
            print(f"  Train | loss={train_result['loss']:.6f}  "
                  f"recon={train_result['recon']:.6f}  "
                  f"kl={train_result['kl']:.6f}")
            print(f"  Val   | loss={val_metrics['loss']:.6f}  "
                  f"MSE={val_metrics['mse']:.6f}  "
                  f"MAE={val_metrics['mae']:.6f}  "
                  f"Corr={val_metrics['correlation']:.4f}  "
                  f"recon={val_metrics['recon']:.6f}  "
                  f"kl={val_metrics['kl']:.6f}")
            if test_metrics:
                print(f"  Test  | loss={test_metrics['loss']:.6f}  "
                      f"MSE={test_metrics['mse']:.6f}  "
                      f"MAE={test_metrics['mae']:.6f}  "
                      f"Corr={test_metrics['correlation']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # ─── Checkpoint ───
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            self.save_checkpoint(is_best=is_best)
            self.save_log_to_csv()

            if self.early_stop_counter >= self.patience:
                print(f"\n[DGWM] Early stopping triggered after {epoch+1} epochs")
                break

        elapsed = time.time() - start_time
        print(f"\n[DGWM] Training completed in {elapsed/60:.1f} minutes")
        print(f"[DGWM] Best epoch: {self.best_epoch}, "
              f"Best val loss: {self.best_val_loss:.6f}")

        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'exp_dir': self.exp_dir,
        }


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DriveGuard-WM Stage 1 Trainer")

    # 数据
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, default=None,
                        help='迁移学习: 测试集路径')

    # 训练参数
    parser.add_argument('--T', type=int, default=5,
                        help='时序窗口长度 (default: 5)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit training to N windows (few-shot)')

    # DriveGuard-WM 参数
    parser.add_argument('--beta', type=float, default=1.0,
                        help='KL 散度权重')
    parser.add_argument('--beta_warmup', type=int, default=20,
                        help='KL annealing warmup epoch 数')
    parser.add_argument('--global_tokens', type=int, default=8,
                        help='Global/process token 数 K (default: 8)')
    parser.add_argument('--local_tokens', type=int, default=64,
                        help='Local content token 数 N (default: 64)')
    parser.add_argument('--global_dim', type=int, default=64,
                        help='Global state 维度 Dg (default: 64)')
    parser.add_argument('--message_dim', type=int, default=64,
                        help='Message 维度 Dm (default: 64)')
    parser.add_argument('--local_dim', type=int, default=64,
                        help='Local state 维度 Dl (default: 64)')
    parser.add_argument('--spatial_dim', type=int, default=128,
                        help='Spatial feature 维度 Ds (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='GRU hidden 维度 (default: 128)')

    # 信息因子变体: 单帧 (无历史动力)
    parser.add_argument('--use_single_frame', action='store_true',
                        help='只用当前帧 (Current 变体, 无历史序列)')

    # 输出
    parser.add_argument('--output_dir', type=str,
                        default='outputs')
    parser.add_argument('--exp_name', type=str, default=None)

    # 恢复
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint .pth 文件恢复训练')

    args = parser.parse_args()

    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'patience': args.patience,
        'num_workers': args.num_workers,
        'beta': args.beta,
        'beta_warmup': args.beta_warmup,
        'normalize': True,
        'max_train_samples': args.max_train_samples,
        'use_single_frame': args.use_single_frame,
        # Architecture
        'global_tokens': args.global_tokens,
        'local_tokens': args.local_tokens,
        'global_dim': args.global_dim,
        'message_dim': args.message_dim,
        'local_dim': args.local_dim,
        'spatial_dim': args.spatial_dim,
        'hidden_dim': args.hidden_dim,
    }

    trainer = DriveGuardWMTrainer(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        T=args.T,
        output_dir=args.output_dir,
        config=config,
        exp_name=args.exp_name,
        resume=args.resume,
    )

    result = trainer.train()
    print(f"\nTraining result: {result}")


if __name__ == "__main__":
    main()
