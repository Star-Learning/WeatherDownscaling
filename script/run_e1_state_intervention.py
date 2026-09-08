"""
E1: Three-Layer State Propagation Analysis
=============================================

目的: 验证退化沿 G → M → L → y_hat 的三层路径传播。

分析内容 (基于已训练的 World Model):
  1. Posterior-prior gap 逐层计算
  2. Information-matched posterior 替换收益
  3. 随 rollout horizon 的变化
  4. 控制条件: random perturbation / continuous interpolation / matched-state / off-manifold
  5. 三层 gap/干预主图 + 数值表

用法:
  # 分析已训练的 World Model
  python3 script/run_e1_state_intervention.py \
      --wm_checkpoint /path/to/world_model.pth \
      --test_dir /path/to/test_data \
      --output_dir outputs/e1_analysis

  # 完整分析 (三区域)
  python3 script/run_e1_state_intervention.py \
      --wm_checkpoint /path/to/wm.pth --mode full
"""
import os
import sys
import json
import time
import datetime
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 确保能找到项目模块
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.driveguard_wm import (
    DriveGuardWM, GaussianState, HierarchicalState, PredictionBundle,
    GlobalRSSM, LocalRSSM, CrossScalePrior, CrossScalePosterior,
    load_model_from_checkpoint, pool_with_assignment,
)
from dataset.temporal_dataset import TemporalDownscaleDataset


class E1StateIntervention:
    """
    E1: Three-Layer State Propagation Analysis.

    对已训练的 World Model (S1) 进行三层状态传播分析。

    Args:
        wm_checkpoint: .pth checkpoint 路径
        test_dir: 测试数据目录
        output_dir: 输出目录
        device: 计算设备
        config: 配置 dict
    """

    def __init__(self, wm_checkpoint, test_dir, output_dir,
                 device=None, config=None, exp_name=None):
        self.wm_checkpoint = wm_checkpoint
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}

        self.T = self.config.get('T', 5)
        self.batch_size = self.config.get('batch_size', 4)
        self.num_workers = self.config.get('num_workers', 2)
        self.num_intervention_samples = self.config.get('num_samples', 100)

        # 结果目录
        self.exp_name = exp_name or f"e1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join(self.output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # ─── 数据集 ───
        print(f"[E1] Loading test data from {test_dir}...")
        self.test_dataset = TemporalDownscaleDataset(
            test_dir, 'val', T=self.T, normalize=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)

        self.target_h = self.test_dataset.hr_h
        self.target_w = self.test_dataset.hr_w

        # ─── 加载模型 ───
        self.model = self._load_model()
        self.model.eval()

        print(f"[E1] Model loaded. HR size: {self.target_h}×{self.target_w}")

    def _load_model(self):
        """从 checkpoint 加载 DriveGuard-WM (自动恢复完整结构)。"""
        print(f"[E1] Loading checkpoint: {self.wm_checkpoint}")
        model = load_model_from_checkpoint(
            self.wm_checkpoint, device=self.device, strict=False)
        return model

    # ════════════════════════════════════════════════════════════════
    # 核心分析函数
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def extract_states(self, lr_seq):
        """
        对单个 batch 提取 G/M/L 的 prior 和 posterior state。

        Args:
            lr_seq: [B, T, C, H, W]
            hr_seq: [B, T, 1, H_hr, W_hr] (可选, 用于 posterior)

        Returns:
            prior_states: list[T] of HierarchicalState
            h_states: 各层隐状态
        """
        B, T, C, H_lr, W_lr = lr_seq.shape
        Hp, Wp = self.model._get_latent_spatial_size(H_lr, W_lr)

        # Encode all frames
        g_obs_list, l_obs_list, assignment_list = [], [], []
        for t in range(T):
            g_obs, l_obs, assignment, _ = self.model._encode_frame(lr_seq[:, t])
            g_obs_list.append(g_obs)
            l_obs_list.append(l_obs)
            assignment_list.append(assignment)

        g_obs_seq = torch.stack(g_obs_list, dim=1)
        l_obs_seq = torch.stack(l_obs_list, dim=1)
        assignment_seq = torch.stack(assignment_list, dim=1)

        # Prior rollout
        prior_states = self.model._rollout_prior(
            g_obs_seq, l_obs_seq, assignment_seq, Hp, Wp)

        return prior_states, assignment_seq, g_obs_seq, l_obs_seq

    @torch.no_grad()
    def compute_gap(self, prior_states):
        """
        计算 posterior-prior gap 各层的 proxy。

        Returns:
            gap_g: [B, K, T] KL or MSE divergence
            gap_m: [B, K, N, T]
            gap_l: [B, N, T]
        """
        # 由于没有 HR, 这里用 temporal rollout 的变异性作为 gap proxy
        # 实际计算需要 posterior, 此处先返回 placeholder
        B = prior_states[0].G.mean.shape[0]
        K = prior_states[0].G.mean.shape[1]
        N = prior_states[0].L.mean.shape[1]
        T = len(prior_states)

        # Simple gap: variance across rollout steps
        g_means = torch.stack([s.G.mean for s in prior_states], dim=-1)
        m_means = torch.stack([s.M.mean for s in prior_states], dim=-1)
        l_means = torch.stack([s.L.mean for s in prior_states], dim=-1)

        gap_g = g_means.var(dim=-1)      # [B, K, T]
        gap_m = m_means.var(dim=-1)      # [B, K, N, T]
        gap_l = l_means.var(dim=-1)      # [B, N, T]

        return gap_g, gap_m, gap_l

    @torch.no_grad()
    def compute_posterior_gap(self, prior_states, lr_seq, hr_seq):
        """
        用真实 HR posterior 计算三层 posterior-prior gap (KL divergence)。

        Args:
            prior_states: list[T] of HierarchicalState (prior rollout)
            lr_seq: [B, T, C, H, W]
            hr_seq: [B, T, 1, H_hr, W_hr]
        Returns:
            kl_g, kl_m, kl_l: [B] 每样本最后一帧的 KL (逐层)
        """
        B, T, C, H_lr, W_lr = lr_seq.shape
        Hp, Wp = self.model._get_latent_spatial_size(H_lr, W_lr)

        # 对最后一帧做 posterior (论文核心: 最后一帧预测的 gap)
        t = T - 1
        prior = prior_states[t]
        assignment = prior.assignment

        # HR 编码 → local tokens
        y_feat = self.model.hr_encoder(
            hr_seq[:, t], target_size=(Hp, Wp))        # [B, Dl, Hp, Wp]
        y_local = pool_with_assignment(
            y_feat, assignment)                         # [B, N, Dl]
        y_global = y_local.mean(dim=1)                  # [B, Dl]

        # G posterior
        g_post = self.model.global_rssm.cell.forward_posterior(prior.G, y_global)
        # M posterior
        m_post = self.model.cross_scale_posterior(
            prior.M, g_post.feature, y_local)
        # L posterior
        l_post = self.model.local_rssm.cell.forward_posterior(
            prior.L, m_post.feature.mean(dim=1), y_local)

        # KL per layer (dim averaged → [B])
        def _kl_batch(mu_q, lv_q, mu_p, lv_p):
            var_q = torch.exp(lv_q)
            var_p = torch.exp(lv_p)
            kl = 0.5 * (lv_p - lv_q
                        + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)
            return kl.mean(dim=tuple(range(1, kl.ndim)))   # [B]

        kl_g = _kl_batch(g_post.mean, g_post.logvar,
                         prior.G.mean, prior.G.logvar)
        kl_m = _kl_batch(m_post.mean, m_post.logvar,
                         prior.M.mean, prior.M.logvar)
        kl_l = _kl_batch(l_post.mean, l_post.logvar,
                         prior.L.mean, prior.L.logvar)

        return kl_g, kl_m, kl_l

    @torch.no_grad()
    def intervention_effect(self, lr_seq, hr_seq, k_idx=0, n_idx=0):
        """
        测试单个 relation (k,n) 替换的干预效果。

        Args:
            lr_seq: [B, T, C, H, W]
            hr_seq: [B, T, 1, H_hr, W_hr]
            k_idx: 替换哪个 global token
            n_idx: 替换哪个 local token

        Returns:
            delta: 替换前后的 loss 变化
        """
        B, T = lr_seq.shape[:2]
        Hp, Wp = self.model._get_latent_spatial_size(lr_seq.shape[3], lr_seq.shape[4])

        # Get prior and posterior states
        prior_states, assignment_seq, g_obs_seq, l_obs_seq = self.extract_states(lr_seq)
        last_state = prior_states[-1]
        target_size = (hr_seq.shape[3], hr_seq.shape[4])

        # Baseline: decode Path A from prior
        y_a = self.model._decode_path_a(last_state, Hp, Wp, target_size)
        loss_base = F.mse_loss(y_a, hr_seq[:, -1]).item()

        # 这里简化处理 — 完整 intervention 需要重新计算 L
        return {'baseline_loss': loss_base}

    @torch.no_grad()
    def random_perturbation_control(self, lr_seq, hr_seq, noise_scale=0.1):
        """E1 控制条件: random perturbation。"""
        prior_states, _, _, _ = self.extract_states(lr_seq)
        last_state = prior_states[-1]
        Hp, Wp = last_state.assignment.shape[2], last_state.assignment.shape[3]
        target_size = (hr_seq.shape[3], hr_seq.shape[4])

        # Perturb G
        g_noise = torch.randn_like(last_state.G.mean) * noise_scale
        g_pert = GaussianState(
            mean=last_state.G.mean + g_noise,
            logvar=last_state.G.logvar,
            sample=last_state.G.sample + g_noise,
        )
        pert_state = HierarchicalState(
            G=g_pert, M=last_state.M, L=last_state.L,
            assignment=last_state.assignment,
        )
        y_pert = self.model._decode_path_a(pert_state, Hp, Wp, target_size)
        loss_pert = F.mse_loss(y_pert, hr_seq[:, -1]).item()

        return loss_pert

    # ════════════════════════════════════════════════════════════════
    # 完整分析流程
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def analyze(self):
        """
        执行完整 E1 分析。

        Returns:
            dict: 包含 gap 数据, intervention 结果, 控制条件结果
        """
        print(f"\n[E1] Running state propagation analysis...")
        print(f"[E1] Test samples: {len(self.test_dataset)}")
        print(f"[E1] Target size: {self.target_h}×{self.target_w}")

        all_gaps_g = []
        all_gaps_m = []
        all_gaps_l = []
        all_kl_g = []
        all_kl_m = []
        all_kl_l = []
        all_losses_base = []
        all_losses_pert = []

        n_batches = min(len(self.test_loader),
                        self.num_intervention_samples // self.batch_size + 1)

        pbar = tqdm(self.test_loader, desc='[E1] Analyzing', total=n_batches)
        for i, batch in enumerate(pbar):
            if i >= n_batches:
                break

            lr_seq = batch['lr'].to(self.device)
            hr_seq = batch['hr'].to(self.device)

            # Extract states
            prior_states, _, _, _ = self.extract_states(lr_seq)

            # Compute rollout-variance gaps (proxy)
            gap_g, gap_m, gap_l = self.compute_gap(prior_states)
            all_gaps_g.append(gap_g.cpu())
            all_gaps_m.append(gap_m.cpu())
            all_gaps_l.append(gap_l.cpu())

            # Compute real HR posterior-prior gaps (KL)
            kl_g, kl_m, kl_l = self.compute_posterior_gap(
                prior_states, lr_seq, hr_seq)
            all_kl_g.append(kl_g.cpu())
            all_kl_m.append(kl_m.cpu())
            all_kl_l.append(kl_l.cpu())

            # Baseline loss
            last_state = prior_states[-1]
            Hp, Wp = last_state.assignment.shape[2], last_state.assignment.shape[3]
            target_size = (hr_seq.shape[3], hr_seq.shape[4])
            y_a = self.model._decode_path_a(last_state, Hp, Wp, target_size)
            loss_base = F.mse_loss(y_a, hr_seq[:, -1]).item()
            all_losses_base.append(loss_base)

            # Control: random perturbation
            loss_pert = self.random_perturbation_control(lr_seq, hr_seq)
            all_losses_pert.append(loss_pert)

        # ─── 汇总 ────────────────────────────────────────────────
        gaps_g = torch.cat(all_gaps_g, dim=0)
        gaps_m = torch.cat(all_gaps_m, dim=0)
        gaps_l = torch.cat(all_gaps_l, dim=0)
        kl_g = torch.cat(all_kl_g, dim=0)
        kl_m = torch.cat(all_kl_m, dim=0)
        kl_l = torch.cat(all_kl_l, dim=0)

        results = {
            'gap_g_mean': gaps_g.mean().item(),
            'gap_m_mean': gaps_m.mean().item(),
            'gap_l_mean': gaps_l.mean().item(),
            'gap_g_std': gaps_g.std().item(),
            'gap_m_std': gaps_m.std().item(),
            'gap_l_std': gaps_l.std().item(),
            # 真实 posterior-prior KL (论文核心)
            'kl_g_mean': kl_g.mean().item(),
            'kl_m_mean': kl_m.mean().item(),
            'kl_l_mean': kl_l.mean().item(),
            'kl_g_std': kl_g.std().item(),
            'kl_m_std': kl_m.std().item(),
            'kl_l_std': kl_l.std().item(),
            'loss_base_mean': np.mean(all_losses_base),
            'loss_pert_mean': np.mean(all_losses_pert),
            'loss_pert_std': np.std(all_losses_pert),
            'n_samples': gaps_g.shape[0],
        }

        # ─── 保存 ────────────────────────────────────────────────
        results_path = os.path.join(self.exp_dir, 'gap_results.json')
        with open(results_path, 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.floating,)) else v
                      for k, v in results.items()}, f, indent=2)

        # 保存 gap 数组用于绘图
        np.savez(os.path.join(self.exp_dir, 'gaps.npz'),
                 gap_g=gaps_g.numpy(), gap_m=gaps_m.numpy(),
                 gap_l=gaps_l.numpy(),
                 kl_g=kl_g.numpy(), kl_m=kl_m.numpy(), kl_l=kl_l.numpy(),
                 losses_base=np.array(all_losses_base),
                 losses_pert=np.array(all_losses_pert))

        print(f"\n[E1] Analysis complete!")
        print(f"  G gap (proxy):  {results['gap_g_mean']:.4f} ± {results['gap_g_std']:.4f}")
        print(f"  M gap (proxy):  {results['gap_m_mean']:.4f} ± {results['gap_m_std']:.4f}")
        print(f"  L gap (proxy):  {results['gap_l_mean']:.4f} ± {results['gap_l_std']:.4f}")
        print(f"  G KL (posterior-prior): {results['kl_g_mean']:.4f} ± {results['kl_g_std']:.4f}")
        print(f"  M KL (posterior-prior): {results['kl_m_mean']:.4f} ± {results['kl_m_std']:.4f}")
        print(f"  L KL (posterior-prior): {results['kl_l_mean']:.4f} ± {results['kl_l_std']:.4f}")
        print(f"  Base loss:  {results['loss_base_mean']:.4f}")
        print(f"  Pert loss:  {results['loss_pert_mean']:.4f} ± {results['loss_pert_std']:.4f}")
        print(f"  Results: {self.exp_dir}")

        return results


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="E1: Three-Layer State Propagation Analysis")

    # 模型
    parser.add_argument('--wm_checkpoint', type=str, required=True,
                        help='World Model checkpoint .pth 路径')

    # 数据
    parser.add_argument('--test_dir', type=str, required=True,
                        help='测试数据目录')
    parser.add_argument('--T', type=int, default=5,
                        help='时序窗口长度')

    # 分析参数
    parser.add_argument('--num_samples', type=int, default=100,
                        help='分析样本数 (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)

    # 输出
    parser.add_argument('--output_dir', type=str,
                        default='outputs')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    config = {
        'T': args.T,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'num_samples': args.num_samples,
    }

    analyzer = E1StateIntervention(
        wm_checkpoint=args.wm_checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        device=args.device,
        config=config,
        exp_name=args.exp_name,
    )

    results = analyzer.analyze()
    print(f"\n[E1] Results saved. Summary:\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
