"""
HR Posterior — encodes HR observations into latent space for posterior inference.

y_hr → HR_encoder → y_feat → pool_with_assignment → y_local → y_global
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import DoubleConv, DownSample


class HRObservationEncoder(nn.Module):
    """
    Encodes HR temperature field into spatial features at (Hp, Wp) resolution.

    Input:  y_hr [B, 1, H_hr, W_hr]
    Output:      [B, D_out, Hp, Wp]
    """
    def __init__(self, out_dim=128):
        super().__init__()
        D = out_dim
        self.net = nn.Sequential(
            DoubleConv(1, D // 4),
            DownSample(D // 4, D // 2),
            DownSample(D // 2, D),
            nn.Conv2d(D, D, kernel_size=1),
        )

    def forward(self, y_hr, target_size=None):
        x = self.net(y_hr)
        if target_size is not None:
            x = F.adaptive_avg_pool2d(x, target_size)
        return x
