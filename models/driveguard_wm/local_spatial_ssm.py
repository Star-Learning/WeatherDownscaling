"""
Windowed Local SSM — spatial scan with window-boundary reset.

Scans the spatial grid in multiple directions within local windows,
resetting state at window boundaries. No full-image continuous state propagation.

Features:
  - Local convolution for near-neighbour compensation
  - Multi-direction scans: left→right, right→left, top→bottom, bottom→top
  - Window boundary reset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMCell(nn.Module):
    """
    Single SSM cell: processes one spatial step within a scan line.

    State update: h_t = A(h_{t-1}) + B(x_t) where A and B are learned.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.A = nn.Linear(dim, dim)
        self.B = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, x, h):
        """
        Args:
            x: [B, D]  input at current step
            h: [B, D]  previous hidden state
        Returns:
            h_new: [B, D]
        """
        h_a = self.A(h)
        h_b = self.B(x)
        g = self.gate(torch.cat([h_a, h_b], dim=-1))
        h_new = (1 - g) * h_a + g * h_b
        return h_new


class LocalSSM(nn.Module):
    """
    Windowed Local SSM — scans within patches; resets at boundaries.

    Operates on a feature map by scanning rows/columns in two directions.
    """
    def __init__(self, dim=64):
        super().__init__()
        self.cell = SSMCell(dim)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def scan_lr(self, x, window_size=8, overlap=2):
        """
        Scan left→right within windows.
        x: [B, D, H, W]
        """
        B, D, H, W = x.shape
        out = torch.zeros_like(x)

        # Process each row independently
        for h in range(H):
            row = x[:, :, h, :]              # [B, D, W]
            row_out = torch.zeros_like(row)

            # Process within windows
            for start in range(0, W, window_size - overlap):
                end = min(start + window_size, W)
                h_state = None
                for w in range(start, end):
                    inp = row[:, :, w]        # [B, D]
                    if h_state is None:
                        h_state = torch.tanh(inp)
                    else:
                        h_state = self.cell(inp, h_state)
                    row_out[:, :, w] = h_state

            out[:, :, h, :] = row_out

        # Add local convolution residual
        local = self.local_conv(x)
        return out + local

    def scan_rl(self, x, window_size=8, overlap=2):
        """
        Scan right→left within windows.
        """
        x_flip = torch.flip(x, dims=[-1])
        out = self.scan_lr(x_flip, window_size, overlap)
        return torch.flip(out, dims=[-1])

    def scan_tb(self, x, window_size=8, overlap=2):
        """
        Scan top→bottom within windows.
        """
        x_t = x.transpose(2, 3)  # [B, D, W, H]
        out = self.scan_lr(x_t, window_size, overlap)
        return out.transpose(2, 3)

    def scan_bt(self, x, window_size=8, overlap=2):
        """
        Scan bottom→top within windows.
        """
        x_t = x.transpose(2, 3)
        out = self.scan_rl(x_t, window_size, overlap)
        return out.transpose(2, 3)


class LocalSpatialSSM(nn.Module):
    """
    Full local spatial SSM with multi-directional scans and conditioning.

    Scans in 4 directions, aggregates results with learned weights.
    Conditioned on `condition` tensor via feature modulation.
    """
    def __init__(self, dim=64, cond_dim=64):
        super().__init__()
        self.ssm_lr = LocalSSM(dim)
        self.ssm_rl = LocalSSM(dim)
        self.ssm_tb = LocalSSM(dim)
        self.ssm_bt = LocalSSM(dim)

        # Learnable fusion weights per direction
        self.fusion_weights = nn.Parameter(torch.ones(4) * 0.25)

        # Condition modulation
        if cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Conv2d(cond_dim, dim, kernel_size=1),
                nn.Sigmoid(),
            )

        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, features, condition=None,
                window_size=8, window_overlap=2):
        """
        Args:
            features:  [B, D, Hp, Wp]  input features from MoE
            condition: [B, Dc, Hp, Wp] conditioning from drive interface
            window_size: SSM window size
            window_overlap: overlap between windows
        Returns:
            z: [B, D, Hp, Wp]  SSM output
        """
        # Multi-direction scans
        z_lr = self.ssm_lr.scan_lr(features, window_size, window_overlap)
        z_rl = self.ssm_rl.scan_rl(features, window_size, window_overlap)
        z_tb = self.ssm_tb.scan_tb(features, window_size, window_overlap)
        z_bt = self.ssm_bt.scan_bt(features, window_size, window_overlap)

        # Learned fusion
        w = F.softmax(self.fusion_weights, dim=0)
        z = (w[0] * z_lr + w[1] * z_rl + w[2] * z_tb + w[3] * z_bt)

        # Condition modulation
        if condition is not None:
            scale = self.cond_proj(condition)
            z = z * scale

        z = self.out_proj(z)
        return z
