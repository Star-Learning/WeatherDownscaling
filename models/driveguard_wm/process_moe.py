"""
Process-guided MoE — variable-guided mixture of experts for Path B.

Process/variable grouping: each expert processes a channel-wise slice of x_now,
guided by the router which uses spatial features + drive relation map.

Expert features are combined via learned router weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import DoubleConv


class ProcessExpert(nn.Module):
    """
    Single expert in the Process MoE.

    Each expert processes its own channel-masked version of the input,
    conditioned on the shared spatial feature s.
    """
    def __init__(self, in_channels=7, spatial_dim=128, hidden_dim=64, top_k=None):
        super().__init__()
        self.top_k = top_k

        self.net = nn.Sequential(
            nn.Conv2d(in_channels + spatial_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_now, s):
        """
        Args:
            x_now: [B, C, H, W]
            s:     [B, Ds, Hp, Wp]  (resized to match x_now)
        Returns:
            feat:  [B, Dh, H_out, W_out]
        """
        # Resize s to match x_now spatial dim if needed
        if s.shape[2:] != x_now.shape[2:]:
            s = F.interpolate(s, size=x_now.shape[2:], mode='bilinear', align_corners=False)

        combined = torch.cat([x_now, s], dim=1)
        return self.net(combined)


class MoERouter(nn.Module):
    """
    Router for Process MoE — determines expert weights per spatial location.

    Uses spatial features + drive relation map + learned expert embeddings.
    """
    def __init__(self, spatial_dim=128, num_experts=4, num_groups=4, hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts

        # Expert embeddings (learnable)
        self.expert_embeddings = nn.Parameter(
            torch.randn(num_experts, hidden_dim) * 0.02
        )

        # Router network
        self.router = nn.Sequential(
            nn.Conv2d(spatial_dim + num_groups, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_experts, kernel_size=1),
        )

    def forward(self, s, drive_relation_map, expert_embeddings=None):
        """
        Args:
            s:                    [B, Ds, Hp, Wp]
            drive_relation_map:   [B, K, Hp, Wp]  (K = num_process_groups)
            expert_embeddings:    ignored (kept for API compat)
        Returns:
            router_logits:  [B, E, Hp, Wp]
        """
        router_in = torch.cat([s, drive_relation_map], dim=1)
        logits = self.router(router_in)                     # [B, E, Hp, Wp]
        return logits


class ProcessMoE(nn.Module):
    """
    Process-guided Mixture of Experts.

    Each expert processes the input independently, then outputs are
    combined via router weights (softmax over experts).
    """
    def __init__(self, in_channels=7, spatial_dim=128, hidden_dim=64,
                 num_experts=4, num_groups=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Experts
        self.experts = nn.ModuleList([
            ProcessExpert(in_channels, spatial_dim, hidden_dim, top_k)
            for _ in range(num_experts)
        ])

        # Router
        self.router = MoERouter(spatial_dim, num_experts, num_groups, hidden_dim)

    def forward(self, x_now, s, drive_relation_map):
        """
        Args:
            x_now:              [B, C, H_lr, W_lr]
            s:                  [B, Ds, Hp, Wp]
            drive_relation_map: [B, K, Hp, Wp]  (K = num_process_groups)
        Returns:
            h_drv:  [B, Dh, Hp, Wp]
            router_weight: [B, E, Hp, Wp]
            aux_loss: dict of auxiliary losses
        """
        # Route
        router_logits = self.router(s, drive_relation_map)
        router_weight = F.softmax(router_logits, dim=1)    # [B, E, Hp, Wp]

        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            feat = expert(x_now, s)
            # Resize to Hp, Wp
            if feat.shape[2:] != (s.shape[2], s.shape[3]):
                feat = F.interpolate(feat, size=(s.shape[2], s.shape[3]),
                                     mode='bilinear', align_corners=False)
            expert_outputs.append(feat)
        expert_stack = torch.stack(expert_outputs, dim=1)  # [B, E, Dh, Hp, Wp]

        # Weighted sum
        w = router_weight.unsqueeze(2)                     # [B, E, 1, Hp, Wp]
        h_drv = (w * expert_stack).sum(dim=1)              # [B, Dh, Hp, Wp]

        # Auxiliary losses
        aux_loss = {}
        # Load balancing loss (encourage even expert usage)
        B, E = router_weight.shape[:2]
        avg_usage = router_weight.mean(dim=[2, 3])         # [B, E]
        target = torch.ones_like(avg_usage) / E
        aux_loss['balance'] = F.mse_loss(avg_usage, target)

        # Sparsity loss (encourage top-k sparsity)
        if self.top_k is not None and self.top_k < E:
            sorted_w, _ = torch.sort(router_weight.view(B, E, -1), dim=1, descending=True)
            topk_sum = sorted_w[:, :self.top_k].sum(dim=1)
            total_sum = sorted_w.sum(dim=1) + 1e-8
            sparsity = 1.0 - (topk_sum / total_sum).mean()
            aux_loss['sparsity'] = sparsity

        return h_drv, router_weight, aux_loss


def relation_weights_to_map_for_moe(weights, assignment):
    """
    Scatter relation weights to spatial grid for MoE routing.

    weights:    [B, K, N]
    assignment: [B, N, Hp, Wp]
    Returns:    [B, K, Hp, Wp]
    """
    B, K, N = weights.shape
    Hp, Wp = assignment.shape[2], assignment.shape[3]
    a = assignment.view(B, N, -1)   # [B, N, Hp*Wp]
    out = torch.bmm(weights, a)     # [B, K, Hp*Wp]
    return out.view(B, K, Hp, Wp)
