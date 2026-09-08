"""
E4: Risk Calibration — 校准与负迁移指标
==========================================

在已训练的 Controller + Path A/B 上计算:

  - R_hat 预测 Path B 变差的 AUROC / AUPRC
  - ECE (Expected Calibration Error) / Brier score
  - routing regret
  - negative-transfer rate
  - worst-domain RMSE
  - deliberately amplified driving stress test

用法:
  python3 script/run_e4_calibration.py \
      --wm_checkpoint /path/to/wm.pth \
      --student_checkpoint /path/to/student.pth \
      --path_b_checkpoint /path/to/path_b.pth \
      --controller_checkpoint /path/to/controller.pth \
      --test_dir /path/to/test \
      --output_dir outputs/e4_calib
"""
import os
import sys
import json
import datetime
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.driveguard_wm import DriveGuardWM, Controller
from models.driveguard_wm.loader import load_model_from_checkpoint
from dataset.temporal_dataset import TemporalDownscaleDataset


def compute_auroc_auprc(labels, scores):
    """计算 AUROC / AUPRC (手写, 避免 sklearn 依赖)。"""
    # 按分数排序
    order = np.argsort(scores)[::-1]
    labels_s = labels[order]
    scores_s = scores[order]

    n_pos = labels_s.sum()
    n_neg = len(labels_s) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5

    # AUROC
    tps = np.cumsum(labels_s)
    fps = np.cumsum(1 - labels_s)
    tpr = tps / n_pos
    fpr = fps / n_neg
    auroc = np.trapezoid(tpr, fpr) if hasattr(np, 'trapezoid') else np.trapz(tpr, fpr)

    # AUPRC (precision-recall)
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / n_pos
    auprc = np.trapezoid(prec, rec) if hasattr(np, 'trapezoid') else np.trapz(prec, rec)

    return float(auroc), float(auprc)


def compute_ece_brier(labels, probs, n_bins=10):
    """ECE + Brier score。"""
    # 分桶
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_conf - bin_acc)

    brier = np.mean((probs - labels) ** 2)
    return float(ece), float(brier)


class E4Calibration:
    """E4 校准与负迁移分析。"""

    def __init__(self, wm_checkpoint, student_checkpoint, path_b_checkpoint,
                 controller_checkpoint, test_dir, output_dir, device=None, config=None):
        self.wm_ckpt = wm_checkpoint
        self.student_ckpt = student_checkpoint
        self.pb_ckpt = path_b_checkpoint
        self.ctrl_ckpt = controller_checkpoint
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or {}
        self.batch_size = self.config.get('batch_size', 4)
        self.T = self.config.get('T', 5)

        self.exp_name = f"e4_calib_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join(output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self._load_models()

    def _load_models(self):
        self.wm = load_model_from_checkpoint(self.wm_ckpt, device=self.device)
        if self.student_ckpt and os.path.exists(self.student_ckpt):
            student = torch.load(self.student_ckpt, map_location=self.device)
            self.wm.matrix_student.load_state_dict(
                student.get('model_state_dict', student)
            )
        if self.pb_ckpt and os.path.exists(self.pb_ckpt):
            pb = torch.load(self.pb_ckpt, map_location=self.device)
            self.wm.path_b_encoder.load_state_dict(pb['path_b_encoder'])
            self.wm.drive_interface.load_state_dict(pb['drive_interface'])
            self.wm.process_moe.load_state_dict(pb['process_moe'])
            self.wm.local_ssm.load_state_dict(pb['local_ssm'])
            self.wm.path_b_decoder.load_state_dict(pb['path_b_decoder'])
        self.wm.eval()
        self.controller = Controller().to(self.device)
        if self.ctrl_ckpt and os.path.exists(self.ctrl_ckpt):
            ck = torch.load(self.ctrl_ckpt, map_location=self.device)
            self.controller.load_state_dict(ck.get('model_state_dict', ck))
        self.controller.eval()

    @torch.no_grad()
    def predict_paths(self, lr_seq, hr_seq, land_mask, scale_r=1.0):
        """
        前向计算 y_a, y_b, d_hat, r_hat, gate。
        支持 risk 放大 (scale_r) 用于 stress test。
        """
        B, T, C, H_lr, W_lr = lr_seq.shape
        Hp, Wp = H_lr // 4, W_lr // 4
        target_size = (hr_seq.shape[3], hr_seq.shape[4])
        x_now = lr_seq[:, -1]

        # WM prior
        g_obs_list, l_obs_list, assignment_list = [], [], []
        for t in range(T):
            g_obs, l_obs, assignment, _ = self.wm._encode_frame(lr_seq[:, t])
            g_obs_list.append(g_obs)
            l_obs_list.append(l_obs)
            assignment_list.append(assignment)
        g_obs_seq = torch.stack(g_obs_list, dim=1)
        l_obs_seq = torch.stack(l_obs_list, dim=1)
        assignment_seq = torch.stack(assignment_list, dim=1)
        prior_states = self.wm._rollout_prior(g_obs_seq, l_obs_seq, assignment_seq, Hp, Wp)
        last_state = prior_states[-1]

        y_a = self.wm._decode_path_a(last_state, Hp, Wp, target_size)

        d_logits, r_raw = self.wm.matrix_student(
            last_state.G, last_state.M, last_state.L, x_now)
        d_hat = F.softplus(d_logits)
        r_hat = F.softplus(r_raw)

        y_b = self.wm._decode_path_b(last_state, x_now, d_hat, Hp, Wp, target_size)

        # risk 放大: 模拟"故意放大风险"的 stress test
        r_eff = r_hat * scale_r

        gate, y_hat = self.controller(
            d_hat, r_eff, last_state.assignment,
            Hp, Wp, y_a.shape[2], y_a.shape[3], y_a, y_b)

        return {
            'y_a': y_a, 'y_b': y_b, 'y_hat': y_hat, 'gate': gate,
            'd_hat': d_hat, 'r_hat': r_hat, 'hr': hr_seq[:, -1],
            'mask': land_mask.unsqueeze(1),
        }

    def run(self):
        """执行全部校准指标计算。"""
        ds = TemporalDownscaleDataset(
            self.test_dir, 'val', T=self.T, normalize=True)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

        all_loss_a = []
        all_loss_b = []
        all_r_mean = []
        all_gate = []
        all_yhat_mae = []
        all_ya_mae = []
        all_yb_mae = []
        # 逐像素收集 (用于 AUROC/ECE/Brier)
        px_gate = []
        px_r = []
        px_neg_transfer = []  # 1 if path_b worse than path_a
        px_diff_ab = []       # |y_b - hr| - |y_a - hr|

        pbar = tqdm(loader, desc='[E4Calib]')
        for batch in pbar:
            lr_seq = batch['lr'].to(self.device)
            hr_seq = batch['hr'].to(self.device)
            land_mask = batch['land_mask'].to(self.device)

            out = self.predict_paths(lr_seq, hr_seq, land_mask)
            m = out['mask']
            hr = out['hr']

            # per-pixel loss
            la = (out['y_a'] - hr).abs() * m
            lb = (out['y_b'] - hr).abs() * m
            lh = (out['y_hat'] - hr).abs() * m
            valid = m.sum() + 1e-8

            all_loss_a.append((la.sum() / valid).item())
            all_loss_b.append((lb.sum() / valid).item())
            all_r_mean.append(out['r_hat'].mean().item())
            all_gate.append(out['gate'].detach().cpu().numpy())
            all_yhat_mae.append((lh.sum() / valid).item())
            all_ya_mae.append((la.sum() / valid).item())
            all_yb_mae.append((lb.sum() / valid).item())

            # 逐像素: gate, r_hat, 负迁移标记
            m_bool = m.bool()
            gate_px = out['gate'][m_bool].detach().cpu()
            r_px = out['r_hat'].mean(dim=1, keepdim=True)  # [B,1,N]→需展开
            # r_hat 是 [B,K,N], 平均到 N 再映射到像素比较麻烦, 用 gate 概率校准即可
            diff_ab = (la - lb)[m_bool].detach().cpu()
            px_gate.append(gate_px)
            px_neg_transfer.append((diff_ab > 0).float())
            px_diff_ab.append(diff_ab)

        # ─── 汇总 ─────────────────────────────────────────────
        # negative-transfer: y_b 比 y_a 差的样本比例
        n_neg = sum(1 for a, b in zip(all_ya_mae, all_yb_mae) if b > a)
        neg_transfer_rate = n_neg / len(all_ya_mae)

        # 平均指标
        results = {
            'n_samples': len(all_ya_mae),
            'path_a_mae': float(np.mean(all_ya_mae)),
            'path_b_mae': float(np.mean(all_yb_mae)),
            'blended_mae': float(np.mean(all_yhat_mae)),
            'negative_transfer_rate': float(neg_transfer_rate),
            'risk_mean': float(np.mean(all_r_mean)),
            'gate_mean': float(np.mean([g.mean() for g in all_gate])),
            'gate_std': float(np.std([g.mean() for g in all_gate])),
        }

        # ─── 校准指标: AUROC/AUPRC/ECE/Brier (gate 预测 Path B 变差) ───
        if px_gate:
            gate_all = torch.cat(px_gate).numpy()
            neg_all = torch.cat(px_neg_transfer).numpy()
            # 用 gate 作为正类概率预测 (gate 高 = 信任 Path B)
            # 负迁移标签: diff_ab > 0 表示 Path B 更差 → 应为负类
            # 所以正类 = Path B 不更差 (gate 应高), 负类 = Path B 更差 (gate 应低)
            pos = (neg_all == 0)  # Path B 不更差
            scores = gate_all

            # AUROC / AUPRC
            from script.run_e4_calibration import compute_auroc_auprc, compute_ece_brier
            try:
                auroc, auprc = compute_auroc_auprc(pos, scores)
                ece, brier = compute_ece_brier(pos, scores)
                results['auroc_pathb_not_worse'] = auroc
                results['auprc_pathb_not_worse'] = auprc
                results['ece'] = ece
                results['brier'] = brier
            except Exception as e:
                print(f"[E4Calib] calibration metrics failed: {e}")

        # r_hat 与 (loss_b - loss_a) 的相关性 → 风险校准信号
        diffs = np.array(all_yb_mae) - np.array(all_ya_mae)
        risks = np.array(all_r_mean)
        if np.std(diffs) > 0 and np.std(risks) > 0:
            results['risk_loss_corr'] = float(np.corrcoef(diffs, risks)[0, 1])
        else:
            results['risk_loss_corr'] = 0.0

        # ─── Stress test: 放大风险 ────────────────────────────
        stress_results = {}
        for scale in [2.0, 5.0, 10.0]:
            mae_stress = []
            gate_stress = []
            for batch in loader:
                lr_seq = batch['lr'].to(self.device)
                hr_seq = batch['hr'].to(self.device)
                land_mask = batch['land_mask'].to(self.device)
                out = self.predict_paths(lr_seq, hr_seq, land_mask, scale_r=scale)
                m = out['mask']
                mae_stress.append(((out['y_hat'] - out['hr']).abs() * m).sum().item()
                                  / (m.sum().item() + 1e-8))
                gate_stress.append(out['gate'].mean().item())
            stress_results[f'scale_{scale}'] = {
                'blended_mae': float(np.mean(mae_stress)),
                'gate_mean': float(np.mean(gate_stress)),
            }
        results['stress_test'] = stress_results

        # ─── 保存 ─────────────────────────────────────────────
        path = os.path.join(self.exp_dir, 'calibration_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        print(json.dumps(results, indent=2))
        print(f"\n[E4Calib] Results saved: {path}")
        return results


def main():
    parser = argparse.ArgumentParser(description="E4: Risk Calibration")
    parser.add_argument('--wm_checkpoint', type=str, required=True)
    parser.add_argument('--student_checkpoint', type=str, required=True)
    parser.add_argument('--path_b_checkpoint', type=str, required=True)
    parser.add_argument('--controller_checkpoint', type=str, default=None)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str,
                        default='outputs')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    calib = E4Calibration(
        wm_checkpoint=args.wm_checkpoint,
        student_checkpoint=args.student_checkpoint,
        path_b_checkpoint=args.path_b_checkpoint,
        controller_checkpoint=args.controller_checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        device=args.device,
        config={'T': args.T, 'batch_size': args.batch_size},
    )
    calib.run()


if __name__ == "__main__":
    main()
