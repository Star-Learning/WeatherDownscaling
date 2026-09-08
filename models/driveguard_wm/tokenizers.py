"""
Tokenizers and Map Utilities for DriveGuard-WM.

Global tokenizer:  x_t → K global tokens (learned queries, no region embedding)
Local tokenizer:   x_t → N local tokens + assignment map [B,N,Hp,Wp]
Map utilities:     token-to-grid, relation-weight-to-grid, message-to-grid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import DoubleConv, DownSample


# ═══════════════════════════════════════════════════════════════════════
# Global Encoder + Tokenizer
# ═══════════════════════════════════════════════════════════════════════

class GlobalEncoder(nn.Module):
    """
    Encodes LR frame into a spatial feature map for global tokenization.

    Input:  [B, C, H_lr, W_lr]
    Output: [B, Ds, Hp, Wp]   where Hp=H_lr/4, Wp=W_lr/4
    """
    def __init__(self, in_channels=7, spatial_dim=128):
        super().__init__()
        D = spatial_dim
        self.net = nn.Sequential(
            DoubleConv(in_channels, D),
            DownSample(D, D * 2),
            DownSample(D * 2, D * 4),
            nn.Conv2d(D * 4, D, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class GlobalTokenizer(nn.Module):
    """
    Global tokenizer — K learnable queries aggregate spatial features.

    Input:  spatial_feat  [B, Ds, Hp, Wp]
    Output: g_obs         [B, K, Dg_in]

    Uses K learnable queries (shared across regions — no region embedding).
    """
    def __init__(self, K=8, spatial_dim=128, global_dim=64):
        super().__init__()
        self.K = K
        self.global_dim = global_dim

        # Learnable queries [1, K, Ds]
        self.queries = nn.Parameter(torch.randn(1, K, spatial_dim) * 0.02)

        # Cross-attention: queries x spatial_feat
        self.q_proj = nn.Linear(spatial_dim, spatial_dim)
        self.k_proj = nn.Linear(spatial_dim, spatial_dim)
        self.v_proj = nn.Linear(spatial_dim, spatial_dim)
        self.out_proj = nn.Linear(spatial_dim, global_dim)

    def forward(self, spatial_feat):
        B = spatial_feat.shape[0]
        # Flatten spatial dims
        feat_flat = spatial_feat.flatten(2).transpose(1, 2)  # [B, Hp*Wp, Ds]

        Q = self.q_proj(self.queries.expand(B, -1, -1))       # [B, K, Ds]
        K = self.k_proj(feat_flat)                             # [B, Hp*Wp, Ds]
        V = feat_flat                                          # [B, Hp*Wp, Ds]

        attn = torch.matmul(Q, K.transpose(1, 2)) / (self.K ** 0.5)
        attn = F.softmax(attn, dim=-1)                         # [B, K, Hp*Wp]

        g_obs = torch.matmul(attn, V)                          # [B, K, Ds]
        g_obs = self.out_proj(g_obs)                           # [B, K, Dg_in]
        return g_obs


# ═══════════════════════════════════════════════════════════════════════
# Local Encoder + Tokenizer
# ═══════════════════════════════════════════════════════════════════════

class LocalEncoder(nn.Module):
    """
    Encodes LR frame into spatial features for local tokenization.

    Input:  [B, C, H_lr, W_lr]
    Output: [B, Ds, Hp, Wp]   same spatial resolution as GlobalEncoder
    """
    def __init__(self, in_channels=7, spatial_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            DoubleConv(in_channels, spatial_dim),
            DownSample(spatial_dim, spatial_dim * 2),
            DownSample(spatial_dim * 2, spatial_dim * 4),
            nn.Conv2d(spatial_dim * 4, spatial_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class LocalTokenizer(nn.Module):
    """
    Local tokenizer — N learnable queries produce local tokens + assignment.

    Input:  local_feat  [B, Ds, Hp, Wp]
    Output: l_obs       [B, N, Dl_in]
            assignment  [B, N, Hp, Wp]    sum(dim=1) ≈ 1

    Assignment usage regularisation is handled by the trainer.
    """
    def __init__(self, N=64, spatial_dim=128, local_dim=64):
        super().__init__()
        self.N = N
        self.local_dim = local_dim

        # Learnable local queries [1, N, Ds]
        self.queries = nn.Parameter(torch.randn(1, N, spatial_dim) * 0.02)

        # Projections for content extraction
        self.q_proj = nn.Linear(spatial_dim, spatial_dim)
        self.k_proj = nn.Linear(spatial_dim, spatial_dim)
        self.v_proj = nn.Linear(spatial_dim, spatial_dim)
        self.out_proj = nn.Linear(spatial_dim, local_dim)

        # Temperature for softmax
        self.logit_scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, local_feat):
        B = local_feat.shape[0]
        Hp, Wp = local_feat.shape[2], local_feat.shape[3]
        feat_flat = local_feat.flatten(2).transpose(1, 2)     # [B, Hp*Wp, Ds]

        Q = self.q_proj(self.queries.expand(B, -1, -1))       # [B, N, Ds]
        K = self.k_proj(feat_flat)                             # [B, Hp*Wp, Ds]
        V = feat_flat                                          # [B, Hp*Wp, Ds]

        # Assignment: softmax over queries
        logits = torch.matmul(Q, K.transpose(1, 2)) * self.logit_scale.abs()
        attn = F.softmax(logits, dim=1)                        # [B, N, Hp*Wp]
        assignment = attn.view(B, self.N, Hp, Wp)              # [B, N, Hp, Wp]

        # Local tokens: weighted sum over spatial positions
        l_obs = torch.matmul(attn, V)                          # [B, N, Ds]
        l_obs = self.out_proj(l_obs)                           # [B, N, Dl_in]
        return l_obs, assignment


# ═══════════════════════════════════════════════════════════════════════
# Map Utilities
# ═══════════════════════════════════════════════════════════════════════

def tokens_to_map(tokens: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """
    Scatter local tokens back to spatial grid via assignment.

    tokens:     [B, N, D]
    assignment: [B, N, Hp, Wp]   sum(dim=1) ≈ 1
    Returns:    [B, D, Hp, Wp]
    """
    B, N, D = tokens.shape
    Hp, Wp = assignment.shape[2], assignment.shape[3]
    # assignment: [B, N, Hp, Wp] → [B, N, Hp*Wp]
    a = assignment.view(B, N, -1)
    # tokens → weighted sum over N at each spatial location
    # [B, D, N] @ [B, N, Hp*Wp] → [B, D, Hp*Wp]
    out = torch.bmm(tokens.transpose(1, 2), a)   # [B, D, Hp*Wp]
    return out.view(B, D, Hp, Wp)


def relation_weights_to_map(weights: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """
    Scatter relation weights (D_hat or R_hat) to spatial grid.

    weights:    [B, K, N]
    assignment: [B, N, Hp, Wp]   sum(dim=1) ≈ 1
    Returns:    [B, K, Hp, Wp]
    """
    B, K, N = weights.shape
    Hp, Wp = assignment.shape[2], assignment.shape[3]

    # assignment: [B, N, Hp*Wp]
    a = assignment.view(B, N, -1)
    # [B, K, N] @ [B, N, Hp*Wp] → [B, K, Hp*Wp]
    out = torch.bmm(weights, a)
    return out.view(B, K, Hp, Wp)


def messages_to_map(messages: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """
    Scatter cross-scale messages to spatial grid.

    messages:   [B, K, N, Dm]
    assignment: [B, N, Hp, Wp]   sum(dim=1) ≈ 1
    Returns:    [B, Dm, Hp, Wp]
    """
    B, K, N, Dm = messages.shape
    Hp, Wp = assignment.shape[2], assignment.shape[3]

    # Average over K, then scatter
    m = messages.mean(dim=1)        # [B, N, Dm]
    a = assignment.view(B, N, -1)   # [B, N, Hp*Wp]
    out = torch.bmm(m.transpose(1, 2), a)  # [B, Dm, Hp*Wp]
    return out.view(B, Dm, Hp, Wp)


def scalar_tokens_to_map(tokens: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """
    Scatter scalar values per local token to spatial grid.

    tokens:     [B, N]
    assignment: [B, N, Hp, Wp]
    Returns:    [B, 1, Hp, Wp]
    """
    B, N = tokens.shape
    Hp, Wp = assignment.shape[2], assignment.shape[3]
    a = assignment.view(B, N, -1)   # [B, N, Hp*Wp]
    # [B, 1, N] @ [B, N, Hp*Wp] → [B, 1, Hp*Wp]
    out = torch.bmm(tokens.unsqueeze(1), a)
    return out.view(B, 1, Hp, Wp)


def pool_with_assignment(feats: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """
    Pool spatial features into local tokens using assignment weights.

    feats:      [B, D, Hp, Wp]
    assignment: [B, N, Hp, Wp]
    Returns:    [B, N, D]
    """
    B, D, Hp, Wp = feats.shape
    N = assignment.shape[1]

    f_flat = feats.view(B, D, -1)               # [B, D, Hp*Wp]
    a_flat = assignment.view(B, N, -1)          # [B, N, Hp*Wp]
    # [B, N, Hp*Wp] @ [B, D, Hp*Wp]^T → [B, N, D]
    out = torch.bmm(a_flat, f_flat.transpose(1, 2)) / (a_flat.sum(dim=-1, keepdim=True) + 1e-8)
    return out


def broadcast_global_context(g_state: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
    """
    Broadcast global/process tokens to spatial grid.

    g_state:  [B, K, Dg]
    Returns:  [B, Dg, Hp, Wp]  — mean over K, then spatial broadcast
    """
    g = g_state.mean(dim=1, keepdim=True)         # [B, 1, Dg]
    g = g.transpose(1, 2).unsqueeze(-1)            # [B, Dg, 1, 1]
    return g.expand(-1, -1, Hp, Wp)               # [B, Dg, Hp, Wp]
