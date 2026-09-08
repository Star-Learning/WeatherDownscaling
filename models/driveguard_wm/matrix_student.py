"""
LR-only Matrix Student — predicts D_hat and R_hat without HR posterior.

This is the student model that runs during target-domain inference.
It learns to mimic the teacher through knowledge distillation.

d_logits_hat, r_raw_hat = matrix_student(g_prior, m_prior, l_prior, x_now)
d_hat = softplus(d_logits_hat)
r_hat = softplus(r_raw_hat)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import DoubleConv


class MatrixStudent(nn.Module):
    """
    LR-only Matrix Student — predicts D/R from prior states + current LR.

    Operates only on prior (no posterior access).
    """
    def __init__(self, global_dim=64, message_dim=64, local_dim=64,
                 in_channels=7, hidden_dim=64, relation_dim=64):
        super().__init__()

        # Encode current LR for spatial context
        self.spatial_encoder = nn.Sequential(
            DoubleConv(in_channels, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # Global token context
        self.g_proj = nn.Linear(global_dim, relation_dim)
        # Local token context
        self.l_proj = nn.Linear(local_dim, relation_dim)
        # Message context
        self.m_proj = nn.Linear(message_dim, relation_dim)
        # LR context (pooled)
        self.lr_proj = nn.Linear(hidden_dim, relation_dim)

        # Joint processing
        self.joint_net = nn.Sequential(
            nn.Linear(relation_dim * 4, relation_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim * 2, relation_dim),
        )

        # D and R heads
        self.d_head = nn.Linear(relation_dim, 1)
        self.r_head = nn.Linear(relation_dim, 1)

    def forward(self, g_prior, m_prior, l_prior, x_now):
        """
        Args:
            g_prior: GaussianState [B, K, Dg]  (features/sample)
            m_prior: GaussianState [B, K, N, Dm]
            l_prior: GaussianState [B, N, Dl]
            x_now:   [B, C, H_lr, W_lr]  current LR frame
        Returns:
            d_logits_hat: [B, K, N]
            r_raw_hat:    [B, K, N]
        """
        B, K, N = m_prior.mean.shape[:3]

        # Encode current LR
        lr_feat = self.spatial_encoder(x_now)    # [B, Dh, H_lr, W_lr]
        lr_pooled = lr_feat.mean(dim=[2, 3])     # [B, Dh]

        # Project each context
        g_ctx = self.g_proj(g_prior.feature)                   # [B, K, Dr]
        m_ctx = self.m_proj(m_prior.feature)                   # [B, K, N, Dr]
        l_ctx = self.l_proj(l_prior.feature)                   # [B, N, Dr]
        lr_ctx = self.lr_proj(lr_pooled)                       # [B, Dr]

        # Expand to [B, K, N, Dr] and concatenate
        g_exp = g_ctx.unsqueeze(2).expand(-1, -1, N, -1)      # [B, K, N, Dr]
        l_exp = l_ctx.unsqueeze(1).expand(-1, K, -1, -1)      # [B, K, N, Dr]
        lr_exp = lr_ctx.unsqueeze(1).unsqueeze(2).expand(-1, K, N, -1)  # [B, K, N, Dr]

        joint = torch.cat([g_exp, m_ctx, l_exp, lr_exp], dim=-1)  # [B, K, N, 4*Dr]
        h = self.joint_net(joint.view(B * K * N, -1)).view(B, K, N, -1)

        d_logits_hat = self.d_head(h.view(B * K * N, -1)).view(B, K, N)
        r_raw_hat = self.r_head(h.view(B * K * N, -1)).view(B, K, N)

        return d_logits_hat, r_raw_hat
