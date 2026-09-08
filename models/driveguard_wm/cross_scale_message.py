"""
Cross-Scale Message — M prior and posterior.

M[k,n] encodes how global/process token k affects local content token n.

Prior:
    G_state + local_queries → M_prior  (GaussianState [B, K, N, Dm])

Posterior:
    M_prior + G_post + y_local → M_post (GaussianState [B, K, N, Dm])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.driveguard_wm.state_types import GaussianState


class CrossScalePrior(nn.Module):
    """
    Cross-scale message prior: for each (k,n) pair, compute message from G to L.

    Uses bilinear interaction: G[k] interacts with local query[n].
    """
    def __init__(self, global_dim=64, local_dim=64, message_dim=64):
        super().__init__()
        self.message_dim = message_dim

        # G → message projection (per global token)
        self.g_proj = nn.Linear(global_dim, message_dim)
        # Local query → message projection
        self.l_proj = nn.Linear(local_dim, message_dim)
        # Bilinear mixing
        self.bilinear = nn.Bilinear(message_dim, message_dim, message_dim)
        # Output → mu, logvar
        self.to_params = nn.Linear(message_dim, message_dim * 2)

    def forward(self, g_state, local_queries):
        """
        Args:
            g_state:       [B, K, Dg]  G_prior features
            local_queries: [N, Dl]     learnable local queries (shared)
        Returns:
            m_prior: GaussianState [B, K, N, Dm]
        """
        B, K, Dg = g_state.shape
        N, Dl = local_queries.shape

        # Project G
        g_proj = self.g_proj(g_state)                        # [B, K, Dm]

        # Project local queries
        l_proj = self.l_proj(local_queries.unsqueeze(0))     # [1, N, Dm]

        # Expand and mix
        g_exp = g_proj.unsqueeze(2).expand(-1, -1, N, -1)   # [B, K, N, Dm]
        l_exp = l_proj.unsqueeze(1).expand(B, K, -1, -1)     # [B, K, N, Dm]

        # Mix via bilinear → element-wise interaction
        g_flat = g_exp.reshape(B * K * N, -1)
        l_flat = l_exp.reshape(B * K * N, -1)
        mixed = self.bilinear(g_flat, l_flat)                 # [B*K*N, Dm]
        mixed = mixed.reshape(B, K, N, self.message_dim)

        params = self.to_params(mixed)                       # [B, K, N, 2*Dm]
        mu, logvar = params.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10, 10)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + eps * std

        return GaussianState(mean=mu, logvar=logvar, sample=sample)


class CrossScalePosterior(nn.Module):
    """
    Cross-scale message posterior: refine M_prior with HR observations.

    Adjustment computed from HR-encoded local tokens.
    """
    def __init__(self, global_dim=64, local_dim=64, message_dim=64):
        super().__init__()
        self.message_dim = message_dim

        # Combine prior with G_post and y_local
        self.gate_net = nn.Sequential(
            nn.Linear(message_dim + global_dim + local_dim, message_dim),
            nn.ReLU(inplace=True),
            nn.Linear(message_dim, message_dim),
            nn.Sigmoid(),
        )
        self.residual = nn.Linear(message_dim + local_dim, message_dim)

    def forward(self, m_prior, g_post, y_local):
        """
        Args:
            m_prior: GaussianState [B, K, N, Dm]
            g_post:  [B, K, Dg]   G_post features
            y_local: [B, N, D_l]  HR-pooled local features
        Returns:
            m_post: GaussianState [B, K, N, Dm]
        """
        B, K, N, Dm = m_prior.mean.shape
        Dg = g_post.shape[-1]

        # Calculate residual adjustment
        # Expand g_post to [B, K, N, Dg]
        g_exp = g_post.unsqueeze(2).expand(-1, -1, N, -1)       # [B, K, N, Dg]
        y_exp = y_local.unsqueeze(1).expand(-1, K, -1, -1)      # [B, K, N, Dl]

        gate_in = torch.cat([m_prior.mean, g_exp, y_exp], dim=-1)
        gate = self.gate_net(gate_in.reshape(B * K * N, -1))    # [B*K*N, Dm]
        gate = gate.reshape(B, K, N, Dm)

        residual_in = torch.cat([m_prior.mean, y_exp], dim=-1)
        residual_in = residual_in.reshape(B * K * N, -1)
        delta = self.residual(residual_in).reshape(B, K, N, Dm)

        mean_post = m_prior.mean + gate * delta

        # Keep prior variance (or slightly reduced)
        logvar_post = m_prior.logvar - 0.5  # small reduction

        std = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std)
        sample = mean_post + eps * std

        return GaussianState(mean=mean_post, logvar=logvar_post, sample=sample)
