"""
Monotone Controller — blends Path A and Path B predictions per-pixel.

Gate = sigmoid(softplus(α) * D_map_hr - softplus(β) * R_map_hr + w_delta * |y_a-y_b| + bias)

D_hat (drive) increases gate → favors Path B
R_hat (risk)  increases gate   → penalised via negative coefficient with softplus(β)
  → ensures increasing R never increases gate (monotonicity)

gate = 0 → Path A only
gate = 1 → Path B only
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.driveguard_wm.tokenizers import scalar_tokens_to_map


class Controller(nn.Module):
    """
    Monotone Controller — blends Path A/B predictions using D and R signals.

    Reference: Section 10 of the spec.
    """
    def __init__(self, use_disagreement=True, gate_temperature=0.5):
        super().__init__()
        self.use_disagreement = use_disagreement

        # Learned parameters
        self.alpha_raw = nn.Parameter(torch.zeros(1))         # drive coefficient
        self.beta_raw = nn.Parameter(torch.zeros(1))          # risk coefficient
        self.w_delta = nn.Parameter(torch.ones(1) * 0.5)      # disagreement weight
        self.bias = nn.Parameter(torch.zeros(1))

        self.tau_gate = gate_temperature

    def forward(self, d_hat, r_hat, assignment, Hp, Wp, H_hr, W_hr, y_a, y_b):
        """
        Args:
            d_hat:      [B, K, N]
            r_hat:      [B, K, N]
            assignment: [B, N, Hp, Wp]
            Hp, Wp:     latent spatial size
            H_hr, W_hr: target HR size
            y_a:        [B, 1, H_hr, W_hr]
            y_b:        [B, 1, H_hr, W_hr]
        Returns:
            gate:   [B, 1, H_hr, W_hr]
            y_hat:  [B, 1, H_hr, W_hr]
        """
        B = d_hat.shape[0]

        # D/R maps
        d_token = d_hat.mean(dim=1)                           # [B, N]
        r_token = r_hat.mean(dim=1)                           # [B, N]

        d_map = scalar_tokens_to_map(d_token, assignment)     # [B, 1, Hp, Wp]
        r_map = scalar_tokens_to_map(r_token, assignment)     # [B, 1, Hp, Wp]

        # Upsample to HR
        d_hr = F.interpolate(d_map, size=(H_hr, W_hr), mode='bilinear', align_corners=False)
        r_hr = F.interpolate(r_map, size=(H_hr, W_hr), mode='bilinear', align_corners=False)

        # Disagreement
        if self.use_disagreement:
            delta = torch.abs(y_a - y_b)
        else:
            delta = torch.zeros_like(y_a)

        # Gate computation (Section 10.2)
        alpha = F.softplus(self.alpha_raw)
        beta = F.softplus(self.beta_raw)  # softplus ensures monotonicity (beta > 0)

        gate_logits = alpha * d_hr - beta * r_hr + self.w_delta * delta + self.bias
        gate = torch.sigmoid(gate_logits)

        # Blended prediction
        y_hat = y_a + gate * (y_b - y_a)

        return gate, y_hat
