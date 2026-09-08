"""
Path B Decoder — produces locally-enhanced HR prediction from SSM + local features.

Input: concat(z, l_map)  [B, Dh+Dl, Hp, Wp]
Output: y_b              [B, 1, H_hr, W_hr]
"""
import torch.nn as nn
import torch.nn.functional as F

from models.base import DoubleConv


class PathBDecoder(nn.Module):
    """
    Decoder for Path B: SSM features + local content → HR prediction.

    Uses PixelShuffle upsampling chain.
    """
    def __init__(self, in_dim=128, hidden_dim=64):
        """
        Args:
            in_dim:  Dh + Dl  (SSM output channels + local content dim)
            hidden_dim: decoder hidden dimension
        """
        super().__init__()

        self.refine = DoubleConv(in_dim, hidden_dim)

        # Upsample chain: ×2 → ×2 → ×2
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        )

        self.final = nn.Sequential(
            DoubleConv(hidden_dim // 2, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, z, target_size=None):
        """
        Args:
            z:           [B, Dh+Dl, Hp, Wp]
            target_size: (H_hr, W_hr)
        Returns:
            y_b:         [B, 1, H_hr, W_hr]
        """
        x = self.refine(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        y_b = self.final(x)

        if target_size is not None:
            y_b = F.interpolate(y_b, size=target_size, mode='bilinear', align_corners=False)

        return y_b
