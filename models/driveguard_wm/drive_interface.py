"""
Driving Interface — computes Path B conditioning from D_hat-weighted M messages.

weighted_m = d_hat[..., None] * M_prior.feature     [B, K, N, Dm]
drive_tokens = weighted_m.sum(dim=1)                 [B, N, Dm]
drive_map = tokens_to_map(drive_tokens, assignment)  [B, Dm, Hp, Wp]

Combined with G and L maps → condition for Path B.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.driveguard_wm.tokenizers import (
    tokens_to_map, broadcast_global_context,
)
from models.driveguard_wm.state_types import GaussianState


class DriveBottleneck(nn.Module):
    """
    Compresses concatenated conditioning maps into a compact condition tensor.
    """
    def __init__(self, in_dim=192, cond_dim=64):
        """
        Args:
            in_dim: Dg + Dm + Dl (from concatenated g_map, l_map, drive_map)
            cond_dim: output condition channels
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, cond_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cond_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cond_dim * 2, cond_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class DriveInterface(nn.Module):
    """
    Full driving interface: takes prior state + d_hat → condition.
    """
    def __init__(self, global_dim=64, message_dim=64, local_dim=64,
                 cond_dim=64):
        super().__init__()
        total_dim = global_dim + message_dim + local_dim
        self.bottleneck = DriveBottleneck(total_dim, cond_dim)

    def forward(self, prior, d_hat, Hp, Wp, detach=True):
        """
        Args:
            prior:  HierarchicalState
            d_hat:  [B, K, N]
            Hp, Wp: latent spatial size
            detach: whether to detach() the prior (Stage 5 default)
        Returns:
            condition: [B, cond_dim, Hp, Wp]
        """
        g_feat = prior.G.feature
        m_feat = prior.M.feature
        l_feat = prior.L.feature
        assignment = prior.assignment

        if detach:
            g_feat = g_feat.detach()
            m_feat = m_feat.detach()
            l_feat = l_feat.detach()
            d_hat = d_hat.detach()

        # Weighted messages
        weighted_m = d_hat[..., None] * m_feat          # [B, K, N, Dm]
        drive_tokens = weighted_m.sum(dim=1)             # [B, N, Dm]

        # Maps
        drive_map = tokens_to_map(drive_tokens, assignment)   # [B, Dm, Hp, Wp]
        g_map = broadcast_global_context(g_feat, Hp, Wp)       # [B, Dg, Hp, Wp]
        l_map = tokens_to_map(l_feat, assignment)              # [B, Dl, Hp, Wp]

        condition = self.bottleneck(torch.cat([g_map, l_map, drive_map], dim=1))
        return condition
