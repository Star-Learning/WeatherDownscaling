"""
DriveGuard-WM State Types — GaussianState, HierarchicalState, PredictionBundle.

All states are dataclasses for clean structured access throughout the model.
"""
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class GaussianState:
    """
    Represents a Gaussian distribution with mean, log-variance, and a sample.

    During training, `sample` is drawn via reparameterization.
    During inference/validation, `sample` defaults to `mean` (deterministic).
    """
    mean: Tensor       # [...] any shape
    logvar: Tensor     # [...] same shape as mean
    sample: Tensor     # [...] same shape as mean (reparameterised during training)

    @property
    def feature(self) -> Tensor:
        """Alias for sample — used throughout the model as the "state feature"."""
        return self.sample


@dataclass
class HierarchicalState:
    """
    Full hierarchical world model state.

    G: global/process state tokens          [B, K, Dg]
    M: cross-scale messages (G→L)           [B, K, N, Dm]
    L: local content state tokens           [B, N, Dl]
    assignment: soft assignment map          [B, N, Hp, Wp]   sum(dim=1) ≈ 1
    """
    G: GaussianState         # [B, K, Dg]
    M: GaussianState         # [B, K, N, Dm]
    L: GaussianState         # [B, N, Dl]
    assignment: Tensor       # [B, N, Hp, Wp]


@dataclass
class PredictionBundle:
    """
    Full model output bundle — returned by infer() / forward_source().

    y_a:    Path A conservative prediction  [B, 1, H_hr, W_hr]
    y_b:    Path B locally-enhanced pred     [B, 1, H_hr, W_hr]
    y_hat:  Final blended prediction          [B, 1, H_hr, W_hr]
    gate:   Per-pixel blend weight            [B, 1, H_hr, W_hr]  (0→Path A, 1→Path B)
    d_hat:  Predicted drive matrix            [B, K, N]  (non-negative)
    r_hat:  Predicted risk matrix             [B, K, N]  (non-negative)
    prior:  HierarchicalState from rollout
    """
    y_a: Tensor             # [B, 1, H_hr, W_hr]
    y_b: Tensor             # [B, 1, H_hr, W_hr]
    y_hat: Tensor           # [B, 1, H_hr, W_hr]
    gate: Tensor            # [B, 1, H_hr, W_hr]
    d_hat: Tensor           # [B, K, N]
    r_hat: Tensor           # [B, K, N]
    prior: HierarchicalState
