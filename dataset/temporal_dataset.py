"""
Temporal Dataset for World Model training.
============================================

支持两种数据格式:

模式 A: 预打包序列 (pre-packed, 推荐)
  数据由 prepare_wm_* 脚本生成, 直接保存为 [T, C, H, W] 的完整序列。
  目录结构: lr_seq/train/, hr_seq/train/
  每个文件本身就是一组完整的 T 帧序列, 直接加载即可。

模式 B: 滑窗采样 (sliding window, 兼容)
  数据与原始 DownscaleDataset 相同, 文件是单独每天 [C, H, W]。
  目录结构: lr/train/, hr/train/
  在加载时通过滑窗拼接为 T 帧序列。

自动检测: 优先使用 lr_seq/ (预打包), 否则回退到 lr/ (滑窗)。

Output shapes (两种模式一致):
  lr:       [T, C, H_lr, W_lr]    ERA5 多通道序列
  hr:       [T, 1, H_hr, W_hr]    HR 温度序列
  land_mask: [H_hr, W_hr]         陆地掩码
  filename:  str                   最后一帧文件名
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TemporalDownscaleDataset(Dataset):
    """
    时序降尺度数据集 — 按滑窗返回连续 T 天序列。

    Args:
        data_root: 数据集路径 (e.g., .../conus_103_237_567_1050_7c)
        split: 'train' or 'val'
        T: 时间窗口长度 (default: 5)
        normalize: 是否归一化 (default: True)

    Data shapes:
        __getitem__ returns:
          lr:       [T, C, H_lr, W_lr]     — 连续 T 天的 LR
          hr:       [T, 1, H_hr, W_hr]     — 对应 T 天的 HR (全帧重建目标)
          land_mask: [H_hr, W_hr]           — 1=陆地, 0=海洋
          filename: str                      — 最后一帧文件名
    """

    def __init__(self, data_root, split, T=5, normalize=True, max_samples=None):
        assert T >= 2, f"T must be >= 2 (got {T}) — need at least 2 frames for dynamics"

        self.data_root = data_root
        self.split = split
        self.T = T
        self.normalize = normalize
        self.max_samples = max_samples

        # ─── 自动检测数据格式 ───
        # 优先: 预打包序列 (prepare_wm_* 输出)
        lr_seq_dir = os.path.join(data_root, "lr_seq", split)
        hr_seq_dir = os.path.join(data_root, "hr_seq", split)
        lr_dir = os.path.join(data_root, "lr", split)
        hr_dir = os.path.join(data_root, "hr", split)

        if os.path.isdir(lr_seq_dir):
            # 模式 A: 预打包序列
            self.packed = True
            self.lr_dir = lr_seq_dir
            self.hr_dir = hr_seq_dir
            # T 由输出数据自身决定 (但保留用户传入的 T 用于校验)
            print(f"[TemporalDownscaleDataset] Pre-packed mode: "
                  f"lr_seq/{split}, hr_seq/{split}")
        else:
            # 模式 B: 滑窗采样
            self.packed = False
            self.lr_dir = lr_dir
            self.hr_dir = hr_dir
            print(f"[TemporalDownscaleDataset] Sliding-window mode: "
                  f"lr/{split}, hr/{split}")

        # 所有 .npy 文件, 按文件名排序 = 时间顺序
        self.files = sorted([f for f in os.listdir(self.lr_dir) if f.endswith('.npy')])
        assert len(self.files) >= 1, f"No .npy files found in {self.lr_dir}"
        if self.max_samples is not None and self.max_samples < len(self.files):
            self.files = self.files[:self.max_samples]
            print(f"  [Few-shot] Limited to {len(self.files)} samples")

        if self.packed:
            # 预打包: 每个文件已是一组完整序列
            self.num_windows = len(self.files)
            # 检测 T 是否匹配
            sample_check = np.load(os.path.join(self.lr_dir, self.files[0]))
            file_T = sample_check.shape[0]
            assert file_T == T, (
                f"Pre-packed data has T={file_T} but dataset was initialized with T={T}. "
                f"Use the same T value as prepare_wm script."
            )
        else:
            # 滑窗: 需要 T 个连续文件拼接为一个样本
            assert len(self.files) >= T, (
                f"Dataset has {len(self.files)} samples, need at least T={T}"
            )
            self.num_windows = len(self.files) - T + 1

        # ─── Land mask (与原有 Dataset 一致) ───
        self.land_mask = None
        hr_h, hr_w = None, None
        mask_path = os.path.join(data_root, "land_mask.npy")
        if os.path.exists(mask_path):
            mask_data = np.load(mask_path)
            self.land_mask = torch.from_numpy(mask_data).float()
            hr_h, hr_w = mask_data.shape

        if hr_h is None:
            hr_sample = np.load(os.path.join(self.hr_dir, self.files[0]))
            if self.packed:
                hr_h, hr_w = hr_sample.shape[2], hr_sample.shape[3]  # [T, 1, H, W]
            else:
                hr_h, hr_w = hr_sample.shape[1], hr_sample.shape[2]  # [1, H, W]
        self.hr_h, self.hr_w = hr_h, hr_w

        # ─── 归一化参数 ───
        self.lr_norm_mean = None  # None = 不做归一化
        self.lr_norm_std = None
        self.hr_norm_mean = 0.0  # HR 温度专用
        self.hr_norm_std = 1.0
        self.norm_mean = 0.0     # 同 hr_norm_mean (兼容旧代码)
        self.norm_std = 1.0

        if self.normalize:
            lr_var_path = os.path.join(data_root, "lr_variables.json")
            if os.path.exists(lr_var_path):
                with open(lr_var_path) as f:
                    lr_info = json.load(f)
                mean_arr = lr_info['normalization']['mean']
                std_arr = lr_info['normalization']['std']
                self.lr_norm_mean = torch.tensor(mean_arr, dtype=torch.float32)
                self.lr_norm_std = torch.tensor(std_arr, dtype=torch.float32)
                # HR 归一化:
                # 优先使用 hr_normalization (降水数据专用, 由 prepare_wm 脚本写入),
                # 否则回退到 ERA5 t2m (通道0) 统计量 (温度数据, 利于迁移学习)。
                if 'hr_normalization' in lr_info:
                    hmean = lr_info['hr_normalization']['mean'][0]
                    hstd = lr_info['hr_normalization']['std'][0]
                else:
                    hmean = mean_arr[0]
                    hstd = std_arr[0]
                self.hr_norm_mean = hmean
                self.hr_norm_std = hstd
                self.norm_mean = hmean
                self.norm_std = hstd
            else:
                # 单通道兼容模式
                norm_path = os.path.join(data_root, "normalization.json")
                if os.path.exists(norm_path):
                    with open(norm_path) as f:
                        norm = json.load(f)
                    self.hr_norm_mean = norm['lr']['mean']
                    self.hr_norm_std = norm['lr']['std']
                    self.norm_mean = self.hr_norm_mean
                    self.norm_std = self.hr_norm_std

        print(f"[TemporalDownscaleDataset] {split}: {self.num_windows} windows"
              f" (T={T}, files={len(self.files)}, mode={'packed' if self.packed else 'sliding'})"
              f" {self.files[0].replace('.npy','')} ... {self.files[-1].replace('.npy','')}")

    @property
    def n_channels(self):
        """LR 输入通道数 (自动从第一个文件检测)"""
        f = np.load(os.path.join(self.lr_dir, self.files[0]))
        if self.packed:
            return f.shape[1]  # [T, C, H, W] → C
        return f.shape[0]      # [C, H, W] → C

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        """
        返回第 idx 个时序样本。

        模式 A (预打包): 直接加载 [T, C, H, W] / [T, 1, H_hr, W_hr]
        模式 B (滑窗):   从 idx 开始取 T 个连续文件, 拼接为序列

        Args:
            idx: sample index [0, num_windows)
        Returns:
            dict with keys:
              lr:       [T, C, H_lr, W_lr]
              hr:       [T, 1, H_hr, W_hr]
              land_mask: [H_hr, W_hr]
              filename:  str (last frame)
        """
        if self.packed:
            # ─── 模式 A: 预打包, 直接加载 ───
            fname = self.files[idx]
            lr_seq = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)
            hr_seq = np.load(os.path.join(self.hr_dir, fname)).astype(np.float32)
            # lr_seq: [T, C, H, W], hr_seq: [T, 1, H_hr, W_hr]
        else:
            # ─── 模式 B: 滑窗拼接 ───
            window_files = self.files[idx: idx + self.T]
            lr_list, hr_list = [], []
            for fname in window_files:
                lr_list.append(np.load(os.path.join(self.lr_dir, fname)).astype(np.float32))
                hr_list.append(np.load(os.path.join(self.hr_dir, fname)).astype(np.float32))
            lr_seq = np.stack(lr_list, axis=0)  # [T, C, H, W]
            hr_seq = np.stack(hr_list, axis=0)  # [T, 1, H_hr, W_hr]
            fname = window_files[-1]

        # ─── 归一化 ───
        # LR: per-channel 或 scalar
        if self.lr_norm_mean is not None:
            # lr_seq: [T, C, H, W], norm: [C]
            lr_seq = (lr_seq - self.lr_norm_mean.numpy()[None, :, None, None]) \
                     / self.lr_norm_std.numpy()[None, :, None, None]
        else:
            lr_seq = (lr_seq - self.norm_mean) / self.norm_std
        # HR: 使用 HR 温度专用归一化参数 (hr_norm_mean/hr_norm_std)
        hr_seq = (hr_seq - self.hr_norm_mean) / self.hr_norm_std

        # 转为 torch tensor
        lr_t = torch.from_numpy(lr_seq).float()
        hr_t = torch.from_numpy(hr_seq).float()

        return {
            'lr': lr_t,        # [T, C, H_lr, W_lr]
            'hr': hr_t,        # [T, 1, H_hr, W_hr]
            'land_mask': self.land_mask if self.land_mask is not None
                         else torch.ones(self.hr_h, self.hr_w, dtype=torch.float32),
            'filename': fname,
        }


def get_temporal_dataloader(data_root, split, T=5, batch_size=8,
                            num_workers=4, normalize=True):
    """
    创建时序 DataLoader。

    Args:
        data_root: 数据集路径
        split: 'train' or 'val'
        T: 时间窗口长度
        batch_size: 批大小
        num_workers: 数据加载进程数
        normalize: 是否归一化
    Returns:
        DataLoader, 每批返回:
          lr:       [B, T, C, H_lr, W_lr]
          hr:       [B, T, 1, H_hr, W_hr]
          land_mask: [B, H_hr, W_hr]
          filename: list[str]
    """
    dataset = TemporalDownscaleDataset(data_root, split, T=T, normalize=normalize)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# ─── 测试 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_TAR = os.environ.get("WEATHER_DATA_ROOT", "data/data_tar")

    for name, path in [
        ("CONUS", "conus_103_237_567_1050_7c"),
        ("AUST",  "aust_141_183_778_896_7c"),
    ]:
        data_root = os.path.join(DATA_TAR, path)
        print(f"\n{'='*60}")
        print(f"{name}: {path}")
        print(f"{'='*60}")

        ds = TemporalDownscaleDataset(data_root, 'train', T=5)
        print(f"  Samples: {len(ds)}")
        print(f"  Channels: {ds.n_channels}")

        sample = ds[0]
        print(f"  lr_seq: {sample['lr'].shape}")
        print(f"  hr_seq: {sample['hr'].shape}")
        print(f"  land_mask: {sample['land_mask'].shape}")
        print(f"  filename: {sample['filename']}")

        # 验证时间连续性
        files_in_window = sorted(ds.files[:5])
        print(f"  Files in window 0: {[f.replace('.npy','') for f in files_in_window]}")

        # DataLoader 测试
        loader = get_temporal_dataloader(data_root, 'train', T=5, batch_size=4)
        batch = next(iter(loader))
        print(f"  Batch lr: {batch['lr'].shape}")
        print(f"  Batch hr: {batch['hr'].shape}")
        print(f"  Batch land_mask: {batch['land_mask'].shape}")
