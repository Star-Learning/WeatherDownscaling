"""
Global RSSM — prior (GRU + Gaussian head) and posterior for global/process tokens.

Prior:
    prev_G + g_obs → G_prior  (GaussianState [B, K, Dg])

Posterior (training only):
    G_prior + HR_global → G_post  (GaussianState [B, K, Dg])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.driveguard_wm.state_types import GaussianState


class GlobalRSSMCell(nn.Module):
    """
    Single-step global RSSM: token-wise GRU + Gaussian prior head.

    For each of the K tokens, runs an independent GRU over the token embedding.
    """
    def __init__(self, global_dim=64, global_hidden=128):
        super().__init__()
        self.global_dim = global_dim
        self.hidden_dim = global_hidden

        # Token-wise GRU each step
        self.gru_cell = nn.GRUCell(
            input_size=global_dim, hidden_size=global_hidden
        )
        # Prior head: hidden → mu, logvar
        self.prior_head = nn.Linear(global_hidden, global_dim * 2)

    def forward_prior(self, g_obs, prev_h=None):
        """
        Args:
            g_obs:  [B, K, Dg_in]  token observations
            prev_h: [B, K, Dh]     previous GRU hidden (or zero)
        Returns:
            prior:  GaussianState [B, K, Dg]
            h:      [B, K, Dh]     updated hidden state
        """
        B, K, _ = g_obs.shape
        Dh = self.hidden_dim

        if prev_h is None:
            prev_h = torch.zeros(B, K, Dh, device=g_obs.device)

        # GRU per token (flatten batch+token dims)
        g_flat = g_obs.reshape(B * K, -1)       # [B*K, Dg_in]
        h_flat = prev_h.reshape(B * K, Dh)       # [B*K, Dh]
        h_new = self.gru_cell(g_flat, h_flat)  # [B*K, Dh]
        h_new = h_new.reshape(B, K, Dh)

        # Prior head
        params = self.prior_head(h_new.reshape(B * K, Dh))   # [B*K, 2*Dg]
        mu, logvar = params.chunk(2, dim=-1)
        mu = mu.reshape(B, K, self.global_dim)
        logvar = logvar.reshape(B, K, self.global_dim)
        logvar = torch.clamp(logvar, -10, 10)

        # Sample (deterministic by default)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + eps * std

        return GaussianState(mean=mu, logvar=logvar, sample=sample), h_new

    def forward_posterior(self, g_prior, y_global):
        """
        Posterior: prior + HR-encoded global observation → G_post

        Args:
            g_prior:  GaussianState [B, K, Dg]
            y_global: [B, Dg]  pooled from HR-encoded features
        Returns:
            g_post:   GaussianState [B, K, Dg]
        """
        mu_p = g_prior.mean
        B, K, Dg = mu_p.shape

        # Expand global observation to all K tokens
        y_exp = y_global.unsqueeze(1).expand(-1, K, -1)   # [B, K, Dg]

        # Simple posterior update: combine prior mean with observation
        # Learnable combination weights
        logvar_p = g_prior.logvar
        precision_p = torch.exp(-logvar_p)
        precision_o = torch.ones_like(precision_p) * 0.1

        mu_post = (mu_p * precision_p + y_exp * precision_o) / (precision_p + precision_o)
        logvar_post = torch.log(1.0 / (precision_p + precision_o) + 1e-8)
        logvar_post = torch.clamp(logvar_post, -10, 10)

        std = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std)
        sample = mu_post + eps * std

        return GaussianState(mean=mu_post, logvar=logvar_post, sample=sample)


class GlobalRSSM(nn.Module):
    """
    Full Global RSSM — runs prior autoregressively over time, supports posterior.
    """
    def __init__(self, global_dim=64, global_hidden=128):
        super().__init__()
        self.cell = GlobalRSSMCell(global_dim, global_hidden)

    def rollout_prior(self, g_obs_seq, prev_h=None):
        """
        Roll out prior over T time steps.

        Args:
            g_obs_seq:  [B, T, K, Dg_in]
            prev_h:     [B, K, Dh] or None
        Returns:
            g_priors:   list[T] of GaussianState [B, K, Dg]
            h_states:   [B, T, K, Dh]
        """
        T = g_obs_seq.shape[1]
        g_priors = []
        h_states = []

        for t in range(T):
            prior, h = self.cell.forward_prior(g_obs_seq[:, t], prev_h)
            g_priors.append(prior)
            h_states.append(h)
            prev_h = h

        h_stack = torch.stack(h_states, dim=1)   # [B, T, K, Dh]
        return g_priors, h_stack

    def forward_prior(self, g_obs, h=None):
        """Single-step convenience."""
        return self.cell.forward_prior(g_obs, h)
