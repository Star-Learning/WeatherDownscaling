"""
Path A Decoder — produces conservative HR prediction from hierarchical state.

Input: concat(g_map, m_map, l_map)  [B, Dg+Dm+Dl, Hp, Wp]
Output: y_a                         [B, 1, H_hr, W_hr]

Uses PixelShuffle upsampling chain, similar to the existing decoder pattern.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import DoubleConv, UpSample


class PathADecoder(nn.Module):
    """
    Decoder for Path A: hierarchical state → HR prediction.

    Takes the concatenated spatial maps (global context + messages + local content)
    and upsamples to HR resolution.
    """
    def __init__(self, state_dim=192, hidden_dim=128):
        """
        Args:
            state_dim:  Dg + Dm + Dl  (total from concatenated maps)
            hidden_dim: decoder hidden dimension
        """
        super().__init__()

        # Initial refine
        self.refine = DoubleConv(state_dim, hidden_dim)

        # Upsample chain: ×2 → ×2 → ×2 (total ×8, then interpolate)
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),       # hidden_dim, ×2
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),       # hidden_dim, ×4
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),       # hidden_dim//2, ×8
        )

        self.final = nn.Sequential(
            DoubleConv(hidden_dim // 2, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, state_map, target_size=None):
        """
        Args:
            state_map:   [B, Dg+Dm+Dl, Hp, Wp]
            target_size: (H_hr, W_hr) or None
        Returns:
            y_a:         [B, 1, H_hr, W_hr]
        """
        x = self.refine(state_map)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        y_a = self.final(x)

        if target_size is not None:
            y_a = F.interpolate(y_a, size=target_size, mode='bilinear', align_corners=False)

        return y_a
