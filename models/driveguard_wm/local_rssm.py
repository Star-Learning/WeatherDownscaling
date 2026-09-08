"""
Local RSSM — prior and posterior for local content tokens.

Prior:
    prev_L + l_obs + message → L_prior  (GaussianState [B, N, Dl])

Posterior (training):
    L_prior + message_post + y_local → L_post  (GaussianState [B, N, Dl])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.driveguard_wm.state_types import GaussianState


class LocalRSSMCell(nn.Module):
    """
    Single-step local RSSM: token-wise GRU + Gaussian prior head,
    conditioned on cross-scale messages.
    """
    def __init__(self, local_dim=64, message_dim=64, hidden_dim=128):
        super().__init__()
        self.local_dim = local_dim
        self.hidden_dim = hidden_dim

        # GRU input: concat(observation, message)
        gru_input = local_dim + message_dim
        self.gru_cell = nn.GRUCell(gru_input, hidden_dim)

        # Prior head: hidden → mu, logvar
        self.prior_head = nn.Linear(hidden_dim, local_dim * 2)

    def forward_prior(self, l_obs, message, prev_h=None):
        """
        Args:
            l_obs:    [B, N, Dl_in]  local token observations
            message:  [B, N, Dm]     mean over K from M_prior
            prev_h:   [B, N, Dh]     or None
        Returns:
            prior:    GaussianState [B, N, Dl]
            h:        [B, N, Dh]
        """
        B, N, _ = l_obs.shape
        Dh = self.hidden_dim

        if prev_h is None:
            prev_h = torch.zeros(B, N, Dh, device=l_obs.device)

        # Flatten for GRU
        gru_in = torch.cat([l_obs, message], dim=-1)              # [B, N, Dl_in+Dm]
        gru_flat = gru_in.reshape(B * N, -1)
        h_flat = prev_h.reshape(B * N, Dh)
        h_new = self.gru_cell(gru_flat, h_flat)                   # [B*N, Dh]
        h_new = h_new.reshape(B, N, Dh)

        params = self.prior_head(h_new.reshape(B * N, Dh))         # [B*N, 2*Dl]
        mu, logvar = params.chunk(2, dim=-1)
        mu = mu.reshape(B, N, self.local_dim)
        logvar = logvar.reshape(B, N, self.local_dim)
        logvar = torch.clamp(logvar, -10, 10)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + eps * std

        return GaussianState(mean=mu, logvar=logvar, sample=sample), h_new

    def forward_posterior(self, l_prior, message_post, y_local):
        """
        Posterior: refine L with HR-encoded local observations.

        Args:
            l_prior:      GaussianState [B, N, Dl]
            message_post: [B, N, Dm]     mean over K from M_post
            y_local:      [B, N, Dl]     HR-pooled local features
        Returns:
            l_post: GaussianState [B, N, Dl]
        """
        mu_p = l_prior.mean
        logvar_p = l_prior.logvar

        # Combine prior with HR observation (simple precision-weighted)
        precision_p = torch.exp(-logvar_p)
        precision_o = torch.ones_like(precision_p) * 0.1

        mu_post = (mu_p * precision_p + y_local * precision_o) / (precision_p + precision_o)
        logvar_post = torch.log(1.0 / (precision_p + precision_o) + 1e-8)
        logvar_post = torch.clamp(logvar_post, -10, 10)

        std = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std)
        sample = mu_post + eps * std

        return GaussianState(mean=mu_post, logvar=logvar_post, sample=sample)


class LocalRSSM(nn.Module):
    """
    Full Local RSSM — runs prior autoregressively over time.
    """
    def __init__(self, local_dim=64, message_dim=64, hidden_dim=128):
        super().__init__()
        self.cell = LocalRSSMCell(local_dim, message_dim, hidden_dim)

    def rollout_prior(self, l_obs_seq, message_seq, prev_h=None):
        """
        Roll out prior over T time steps.

        Args:
            l_obs_seq:    [B, T, N, Dl_in]
            message_seq:  [B, T, N, Dm]
            prev_h:       [B, N, Dh] or None
        Returns:
            l_priors: list[T] of GaussianState [B, N, Dl]
            h_states: [B, T, N, Dh]
        """
        T = l_obs_seq.shape[1]
        l_priors = []
        h_states = []

        for t in range(T):
            prior, h = self.cell.forward_prior(l_obs_seq[:, t], message_seq[:, t], prev_h)
            l_priors.append(prior)
            h_states.append(h)
            prev_h = h

        h_stack = torch.stack(h_states, dim=1)
        return l_priors, h_stack

    def forward_prior(self, l_obs, message, h=None):
        """Single-step convenience."""
        return self.cell.forward_prior(l_obs, message, h)
