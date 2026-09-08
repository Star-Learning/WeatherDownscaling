"""
D/R Matrix Teacher — computes target D/R matrices from posterior states.

D and R [B, K, N] are non-negative matrices encoding:
  D: drive (beneficial intervention) — larger = more helpful to use Path B
  R: risk  — larger = more uncertain, avoid Path B

Only trained in source domain with HR posterior available.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixTeacher(nn.Module):
    """
    Posterior teacher: from G/M/L posterior states, predict D and R targets.

    d_logits_q, r_raw_q = teacher(g_post, m_post, l_post)
    d_q = softplus(d_logits_q)
    r_q = softplus(r_raw_q)
    """
    def __init__(self, global_dim=64, message_dim=64, local_dim=64,
                 relation_dim=64, contrastive_dim=32):
        super().__init__()

        total_dim = global_dim + message_dim + local_dim

        self.joint_proj = nn.Sequential(
            nn.Linear(total_dim, relation_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim * 2, relation_dim),
        )

        # Separate heads for D and R
        self.d_head = nn.Linear(relation_dim, 1)   # logits per (k,n)
        self.r_head = nn.Linear(relation_dim, 1)   # raw per (k,n)

        # Contrastive projections
        self.d_proj = nn.Linear(relation_dim, contrastive_dim)
        self.r_proj = nn.Linear(relation_dim, contrastive_dim)

    def forward(self, g_post, m_post, l_post):
        """
        Args:
            g_post: GaussianState [B, K, Dg]
            m_post: GaussianState [B, K, N, Dm]
            l_post: GaussianState [B, N, Dl]
        Returns:
            d_logits_q: [B, K, N]
            r_raw_q:    [B, K, N]
            z_d:        [B, K, N, D_contrast]
            z_r:        [B, K, N, D_contrast]
        """
        B, K, N = m_post.mean.shape[:3]

        # Expand G and L to [B, K, N, D]
        g_feat = g_post.feature.unsqueeze(2).expand(-1, -1, N, -1)  # [B, K, N, Dg]
        l_feat = l_post.feature.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, N, Dl]

        # Concatenate
        joint = torch.cat([g_feat, m_post.feature, l_feat], dim=-1)  # [B, K, N, total]
        joint_flat = joint.view(B * K * N, -1)
        h = self.joint_proj(joint_flat).view(B, K, N, -1)           # [B, K, N, Dr]

        # D and R predictions
        d_logits_q = self.d_head(h.view(B * K * N, -1)).view(B, K, N)
        r_raw_q = self.r_head(h.view(B * K * N, -1)).view(B, K, N)

        # Contrastive embeddings
        z_d = self.d_proj(h)   # [B, K, N, D_contrast]
        z_r = self.r_proj(h)   # [B, K, N, D_contrast]

        return d_logits_q, r_raw_q, z_d, z_r
