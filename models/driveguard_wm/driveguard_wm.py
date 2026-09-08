"""
DriveGuard-WM — Top-level model implementing the full architecture.

Architecture (from spec section 2):
  x_hist → Path A (Hierarchical World Model) → y_a
         → Matrix Student → d_hat, r_hat
         → Path B (Process MoE + Local SSM) → y_b
         → Controller → gate → y_hat = y_a + gate * (y_b - y_a)

API (from spec section 11):
  forward_source(x_hist, y_hr, stage) → dict of losses and predictions
  infer(x_hist) → PredictionBundle

Stage-based training (spec section 14):
  S1: Tokenizer + G/M/L prior/posterior + Path A
  S2: Relation intervention (no gradient)
  S3: Similarity heads + D/R Teacher
  S4: LR-only Matrix Student
  S5: Path B + MoE + Local SSM + decoder B
  S6: Controller
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.driveguard_wm.state_types import (
    GaussianState, HierarchicalState, PredictionBundle,
)
from models.driveguard_wm.tokenizers import (
    GlobalEncoder, GlobalTokenizer,
    LocalEncoder, LocalTokenizer,
    tokens_to_map, messages_to_map, broadcast_global_context,
    pool_with_assignment,
)
from models.driveguard_wm.global_rssm import GlobalRSSM
from models.driveguard_wm.cross_scale_message import (
    CrossScalePrior, CrossScalePosterior,
)
from models.driveguard_wm.local_rssm import LocalRSSM
from models.driveguard_wm.posterior import HRObservationEncoder
from models.driveguard_wm.path_a_decoder import PathADecoder
from models.driveguard_wm.matrix_teacher import MatrixTeacher
from models.driveguard_wm.matrix_student import MatrixStudent
from models.driveguard_wm.drive_interface import DriveInterface
from models.driveguard_wm.process_moe import ProcessMoE, relation_weights_to_map_for_moe
from models.driveguard_wm.local_spatial_ssm import LocalSpatialSSM
from models.driveguard_wm.path_b_decoder import PathBDecoder
from models.driveguard_wm.controller import Controller


class PathBEncoder(nn.Module):
    """
    Path B independent spatial encoder (spec section 9.1).

    Encodes current LR frame x_now into spatial features at latent grid resolution.
    Output resolution: (H_lr // patch_stride, W_lr // patch_stride)
    Does NOT read historical sequence.
    """
    def __init__(self, in_channels=7, spatial_dim=128, patch_stride=4):
        super().__init__()
        self.patch_stride = patch_stride
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, spatial_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_dim, spatial_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_dim),
            nn.ReLU(inplace=True),
        )
        # Adaptive pooling to ensure latent grid resolution
        self.pool = nn.AdaptiveAvgPool2d((None, None))  # used dynamically

    def forward(self, x_now, target_hp=None, target_wp=None):
        """
        x_now: [B, C, H_lr, W_lr]
        Returns: [B, Ds, Hp, Wp]
        """
        x = self.net(x_now)
        if target_hp is not None and target_wp is not None:
            x = F.interpolate(x, size=(target_hp, target_wp), mode='bilinear', align_corners=False)
        return x


class DriveGuardWM(nn.Module):
    """
    DriveGuard-WM: Hierarchical World Model with dual-path blending.

    Key hyperparameters:
        K (global_tokens):     number of global/process state tokens
        N (local_tokens):      number of local content tokens
        Dg (global_dim):       global state dimension
        Dm (message_dim):      cross-scale message dimension
        Dl (local_dim):        local state dimension
        Ds (spatial_dim):      spatial feature dimension
        patch_stride:          downsampling factor from LR to latent grid
    """
    def __init__(
        self,
        # Architecture
        in_channels=7,
        K=8,                  # global/process relation tokens
        N=64,                 # local content tokens
        Dg=64,                # global state dim
        Dm=64,                # message dim
        Dl=64,                # local state dim
        Ds=128,               # spatial feature dim
        Dc=64,                # condition dim (drive bottleneck)
        hidden_dim=128,       # GRU hidden dim
        patch_stride=4,
        # Matrix
        relation_dim=64,
        contrastive_dim=32,
        # Path B
        num_experts=4,
        top_k=2,
        window_size=8,
        window_overlap=2,
        ssm_dim=64,
        # Controller
        use_disagreement=True,
        gate_temperature=0.5,
        # Loss weights
        beta=1.0,
    ):
        super().__init__()

        # ── 保存完整构造参数以便 checkpoint 自恢复 ────────────
        self.model_class = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        self.model_kwargs = {
            'in_channels': in_channels,
            'K': K, 'N': N, 'Dg': Dg, 'Dm': Dm, 'Dl': Dl,
            'Ds': Ds, 'Dc': Dc, 'hidden_dim': hidden_dim,
            'patch_stride': patch_stride,
            'relation_dim': relation_dim, 'contrastive_dim': contrastive_dim,
            'num_experts': num_experts, 'top_k': top_k,
            'window_size': window_size, 'window_overlap': window_overlap,
            'ssm_dim': ssm_dim,
            'use_disagreement': use_disagreement,
            'gate_temperature': gate_temperature,
            'beta': beta,
        }

        self.K = K
        self.N = N
        self.patch_stride = patch_stride

        # ── Encoders & Tokenizers ────────────────────────────────
        self.global_encoder = GlobalEncoder(in_channels, Ds)
        self.global_tokenizer = GlobalTokenizer(K, Ds, Dg)

        self.local_encoder = LocalEncoder(in_channels, Ds)
        self.local_tokenizer = LocalTokenizer(N, Ds, Dl)

        # Learnable local queries (shared across all regions)
        self.local_queries = nn.Parameter(torch.randn(N, Dl) * 0.02)

        # ── Path A: Hierarchical World Model ─────────────────────
        self.global_rssm = GlobalRSSM(Dg, hidden_dim)
        self.cross_scale_prior = CrossScalePrior(Dg, Dl, Dm)
        self.cross_scale_posterior = CrossScalePosterior(Dg, Dl, Dm)
        self.local_rssm = LocalRSSM(Dl, Dm, hidden_dim)

        # HR posterior encoder
        self.hr_encoder = HRObservationEncoder(Dl)  # output dim Dl

        # Path A decoder
        path_a_in_dim = Dg + Dm + Dl
        self.path_a_decoder = PathADecoder(path_a_in_dim, hidden_dim)

        # ── Matrix Teacher & Student ─────────────────────────────
        self.matrix_teacher = MatrixTeacher(Dg, Dm, Dl, relation_dim, contrastive_dim)
        self.matrix_student = MatrixStudent(Dg, Dm, Dl, in_channels, hidden_dim, relation_dim)

        # ── Path B ───────────────────────────────────────────────
        self.path_b_encoder = PathBEncoder(in_channels, Ds, patch_stride)
        self.drive_interface = DriveInterface(Dg, Dm, Dl, Dc)
        self.process_moe = ProcessMoE(in_channels, Ds, Dm, num_experts, K, top_k)
        self.local_ssm = LocalSpatialSSM(ssm_dim, Dc)
        path_b_in_dim = ssm_dim + Dl
        self.path_b_decoder = PathBDecoder(path_b_in_dim, hidden_dim)

        # ── Controller ───────────────────────────────────────────
        self.controller = Controller(use_disagreement, gate_temperature)

        # ── Loss weight ──────────────────────────────────────────
        self.beta = beta

    # ════════════════════════════════════════════════════════════════
    # Internal utilities
    # ════════════════════════════════════════════════════════════════

    def _reparameterize(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mu_q, logvar_q, mu_p, logvar_p):
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (logvar_p - logvar_q
                     + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)
        return kl.mean()

    def _get_latent_spatial_size(self, H_lr, W_lr):
        """Compute latent spatial size (Hp, Wp) after patch_stride downsampling."""
        Hp = H_lr // self.patch_stride
        Wp = W_lr // self.patch_stride
        return Hp, Wp

    # ════════════════════════════════════════════════════════════════
    # Core: Encode one frame
    # ════════════════════════════════════════════════════════════════

    def _encode_frame(self, x_t):
        """
        Encode a single LR frame into global and local tokens.

        Args:
            x_t: [B, C, H_lr, W_lr]
        Returns:
            g_obs:      [B, K, Dg]
            l_obs:      [B, N, Dl]
            assignment: [B, N, Hp, Wp]
            spatial_feat: [B, Ds, Hp, Wp]  (from local encoder)
        """
        # Global path
        g_feat = self.global_encoder(x_t)      # [B, Ds, Hp, Wp]
        g_obs = self.global_tokenizer(g_feat)   # [B, K, Dg]

        # Local path
        l_feat = self.local_encoder(x_t)        # [B, Ds, Hp, Wp]
        l_obs, assignment = self.local_tokenizer(l_feat)  # [B,N,Dl], [B,N,Hp,Wp]

        # Use the local encoder features as spatial_feat
        return g_obs, l_obs, assignment, l_feat

    def _rollout_prior(self, g_obs_seq, l_obs_seq, assignment_seq, Hp, Wp):
        """
        Roll out the full hierarchical prior over T time steps.

        Args:
            g_obs_seq:      [B, T, K, Dg]
            l_obs_seq:      [B, T, N, Dl]
            assignment_seq: [B, T, N, Hp, Wp]
            Hp, Wp:         latent spatial size
        Returns:
            prior_states: list[T] of HierarchicalState
        """
        T = g_obs_seq.shape[1]
        B = g_obs_seq.shape[0]

        prior_states = []
        g_h = None
        l_h = None

        for t in range(T):
            # Global prior
            g_prior, g_h = self.global_rssm.forward_prior(g_obs_seq[:, t], g_h)

            # Cross-scale message prior
            m_prior = self.cross_scale_prior(g_prior.feature, self.local_queries)

            # Local prior (with message from G)
            m_local = m_prior.feature.mean(dim=1)    # [B, N, Dm]
            l_prior, l_h = self.local_rssm.forward_prior(
                l_obs_seq[:, t], m_local, l_h
            )

            state = HierarchicalState(
                G=g_prior,
                M=m_prior,
                L=l_prior,
                assignment=assignment_seq[:, t],
            )
            prior_states.append(state)

        return prior_states

    # ════════════════════════════════════════════════════════════════
    # Path A Decode
    # ════════════════════════════════════════════════════════════════

    def _decode_path_a(self, state, Hp, Wp, target_size):
        """
        Decode HR from hierarchical prior state (Path A).

        y_a = path_a_decoder(concat(g_map, m_map, l_map))
        """
        g_map = broadcast_global_context(state.G.feature, Hp, Wp)    # [B, Dg, Hp, Wp]
        m_map = messages_to_map(state.M.feature, state.assignment)   # [B, Dm, Hp, Wp]
        l_map = tokens_to_map(state.L.feature, state.assignment)     # [B, Dl, Hp, Wp]

        state_map = torch.cat([g_map, m_map, l_map], dim=1)          # [B, Dg+Dm+Dl, Hp, Wp]
        y_a = self.path_a_decoder(state_map, target_size)

        return y_a

    # ════════════════════════════════════════════════════════════════
    # Path B Decode
    # ════════════════════════════════════════════════════════════════

    def _decode_path_b(self, prior_state, x_now, d_hat, Hp, Wp, target_size):
        """
        Full Path B forward: encoder → drive → MoE → SSM → decoder.

        Args:
            prior_state: HierarchicalState (last time step)
            x_now:       [B, C, H_lr, W_lr]
            d_hat:       [B, K, N]
            Hp, Wp:      latent spatial size
            target_size: (H_hr, W_hr)
        Returns:
            y_b: [B, 1, H_hr, W_hr]
        """
        # Path B encoder (spec 9.1) — downsample to latent grid
        s = self.path_b_encoder(x_now, Hp, Wp)       # [B, Ds, Hp, Wp]

        # Drive interface (spec 9.2)
        condition = self.drive_interface(
            prior_state, d_hat, Hp, Wp, detach=True
        )                                             # [B, Dc, Hp, Wp]

        # Process MoE (spec 9.3)
        drive_relation_map = relation_weights_to_map_for_moe(
            d_hat, prior_state.assignment
        )                                             # [B, K, Hp, Wp]
        h_drv, router_weight, moe_losses = self.process_moe(
            x_now, s, drive_relation_map
        )                                             # [B, Dh, Hp, Wp]

        # Windowed Local SSM (spec 9.4)
        z = self.local_ssm(
            h_drv, condition,
            window_size=8, window_overlap=2,
        )                                             # [B, Dh, Hp, Wp]

        # Concatenate with local content map
        l_map = tokens_to_map(
            prior_state.L.feature, prior_state.assignment
        )                                             # [B, Dl, Hp, Wp]
        z_cat = torch.cat([z, l_map], dim=1)          # [B, Dh+Dl, Hp, Wp]

        # Path B decoder (spec 9.5)
        y_b = self.path_b_decoder(z_cat, target_size)

        return y_b

    # ════════════════════════════════════════════════════════════════
    # S1 Forward: World Model Training
    # ════════════════════════════════════════════════════════════════

    def forward_s1(self, x_hist, y_hr=None, target_size=None, beta=None):
        """
        Stage 1 training: Tokenizer + G/M/L prior/posterior + Path A.

        Args:
            x_hist:      [B, T, C, H_lr, W_lr]
            y_hr:        [B, T, 1, H_hr, W_hr] or None (inference)
            target_size: (H_hr, W_hr)
            beta:        KL weight override
        Returns:
            dict with keys: hr_pred, mu_prior, logvar_prior, mu_post, logvar_post,
                           recon_loss, kl_loss, total_loss, prior_states
        """
        B, T, C, H_lr, W_lr = x_hist.shape
        Hp, Wp = self._get_latent_spatial_size(H_lr, W_lr)
        if target_size is None and y_hr is not None:
            target_size = (y_hr.shape[3], y_hr.shape[4])

        # ── 1. Encode all frames ────────────────────────────────
        g_obs_list, l_obs_list, assignment_list = [], [], []
        l_feat_first = None
        for t in range(T):
            g_obs, l_obs, assignment, l_feat = self._encode_frame(x_hist[:, t])
            g_obs_list.append(g_obs)
            l_obs_list.append(l_obs)
            assignment_list.append(assignment)
            if l_feat_first is None:
                l_feat_first = l_feat

        g_obs_seq = torch.stack(g_obs_list, dim=1)           # [B, T, K, Dg]
        l_obs_seq = torch.stack(l_obs_list, dim=1)           # [B, T, N, Dl]
        assignment_seq = torch.stack(assignment_list, dim=1)  # [B, T, N, Hp, Wp]

        # ── 2. Prior rollout ───────────────────────────────────
        prior_states = self._rollout_prior(g_obs_seq, l_obs_seq, assignment_seq, Hp, Wp)
        last_state = prior_states[-1]

        # ── Inference only ─────────────────────────────────────
        if y_hr is None:
            y_a = self._decode_path_a(last_state, Hp, Wp, target_size)
            return {
                'hr_pred': y_a,
                'prior_states': prior_states,
            }

        # ── 3. Posterior (training only) ───────────────────────
        mu_post_list, lv_post_list = [], []
        y_local_list = []

        for t in range(T):
            # Encode HR
            y_feat = self.hr_encoder(
                y_hr[:, t], target_size=(Hp, Wp)
            )  # [B, Dl, Hp, Wp]

            # Pool to local tokens
            y_local = pool_with_assignment(y_feat, assignment_seq[:, t])  # [B, N, Dl]
            y_local_list.append(y_local)

            # Global posterior
            y_global = y_local.mean(dim=1)  # [B, Dl]
            g_post = self.global_rssm.cell.forward_posterior(
                prior_states[t].G, y_global
            )

            # Message posterior
            m_post = self.cross_scale_posterior(
                prior_states[t].M,
                g_post.feature,
                y_local,
            )

            # Local posterior
            m_post_local = m_post.feature.mean(dim=1)  # [B, N, Dm]
            l_post = self.local_rssm.cell.forward_posterior(
                prior_states[t].L,
                m_post_local,
                y_local,
            )

            mu_post_list.append(g_post.mean)
            lv_post_list.append(g_post.logvar)

        mu_post = torch.stack(mu_post_list, dim=1)    # [B, T, K, Dg]
        logvar_post = torch.stack(lv_post_list, dim=1)

        # ── 4. Decode Path A (last state, prior) ─────────────────
        y_a = self._decode_path_a(last_state, Hp, Wp, target_size)

        # ── 5. Losses ──────────────────────────────────────────
        # Reconstruction loss (y_a vs y_hr last frame)
        y_hr_last = y_hr[:, -1]
        recon_loss = F.mse_loss(y_a, y_hr_last)

        # KL divergence (global only for S1 simplicity, or full)
        # We track the KL from each level
        kl_g = sum(
            self._kl_divergence(
                s.G.mean, s.G.logvar,
                prior_states[t].G.mean, prior_states[t].G.logvar,
            ) for t, s in enumerate(prior_states)
            if s.G.mean.shape == prior_states[t].G.mean.shape  # simple check
        ) / T

        # For now, only G-level KL is used. M and L KLs can be added in full impl.
        kl_loss = kl_g

        b = beta if beta is not None else self.beta
        total_loss = recon_loss + b * kl_loss

        return {
            'hr_pred': y_a,
            'mu_prior': torch.stack([s.G.mean for s in prior_states], dim=1),
            'logvar_prior': torch.stack([s.G.logvar for s in prior_states], dim=1),
            'mu_post': mu_post,
            'logvar_post': logvar_post,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss,
            'prior_states': prior_states,
        }

    # ════════════════════════════════════════════════════════════════
    # Full Inference
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def infer(self, x_hist, target_size=None, full_bundle=False):
        """
        Target-domain inference (no HR, no posterior).

        Args:
            x_hist:      [B, T, C, H_lr, W_lr]
            target_size: (H_hr, W_hr) — if None, uses decoder's default (×8)
            full_bundle: if True, return PredictionBundle with all intermediates
        Returns:
            y_hat: [B, 1, H_hr, W_hr]  (if full_bundle=False)
            PredictionBundle  (if full_bundle=True)
        """
        B, T, C, H_lr, W_lr = x_hist.shape
        Hp, Wp = self._get_latent_spatial_size(H_lr, W_lr)

        # ── 1. Encode all frames ────────────────────────────────
        g_obs_list, l_obs_list, assignment_list = [], [], []
        for t in range(T):
            g_obs, l_obs, assignment, _ = self._encode_frame(x_hist[:, t])
            g_obs_list.append(g_obs)
            l_obs_list.append(l_obs)
            assignment_list.append(assignment)

        g_obs_seq = torch.stack(g_obs_list, dim=1)
        l_obs_seq = torch.stack(l_obs_list, dim=1)
        assignment_seq = torch.stack(assignment_list, dim=1)

        # ── 2. Prior rollout ───────────────────────────────────
        prior_states = self._rollout_prior(g_obs_seq, l_obs_seq, assignment_seq, Hp, Wp)
        last_state = prior_states[-1]

        # Default target size if not provided (decoder default is ×8 from Hp)
        if target_size is None:
            target_size = (Hp * 8, Wp * 8)

        # ── 3. Decode Path A ───────────────────────────────────
        y_a = self._decode_path_a(last_state, Hp, Wp, target_size)

        # ── 4. Matrix Student → D_hat, R_hat ──────────────────
        x_now = x_hist[:, -1]
        d_logits_hat, r_raw_hat = self.matrix_student(
            last_state.G, last_state.M, last_state.L, x_now
        )
        d_hat = F.softplus(d_logits_hat)
        r_hat = F.softplus(r_raw_hat)

        # ── 5. Path B ─────────────────────────────────────────
        y_b = self._decode_path_b(last_state, x_now, d_hat, Hp, Wp, target_size)

        # ── 6. Controller ─────────────────────────────────────
        gate, y_hat = self.controller(
            d_hat, r_hat, last_state.assignment,
            Hp, Wp,
            y_a.shape[2], y_a.shape[3],
            y_a, y_b,
        )

        if full_bundle:
            return PredictionBundle(
                y_a=y_a,
                y_b=y_b,
                y_hat=y_hat,
                gate=gate,
                d_hat=d_hat,
                r_hat=r_hat,
                prior=last_state,
            )
        return y_hat

    # ════════════════════════════════════════════════════════════════
    # Unified forward (S1 training + inference)
    # ════════════════════════════════════════════════════════════════

    def forward(self, x_hist, y_hr=None, target_size=None, beta=None, stage='s1'):
        """
        Unified forward pass — dispatches based on presence of y_hr.

        Args:
            x_hist:      [B, T, C, H_lr, W_lr]
            y_hr:        [B, T, 1, H_hr, W_hr] or None
            target_size: (H_hr, W_hr) or None
            beta:        KL weight override
            stage:       training stage ('s1', 'infer', etc.)
        Returns:
            If y_hr is not None: dict of losses (training mode)
            If y_hr is None:     PredictionBundle or [B,1,H_hr,W_hr] (inference)
        """
        if stage == 'infer' or y_hr is None:
            return self.infer(x_hist, target_size=target_size, full_bundle=False)
        return self.forward_s1(x_hist, y_hr, target_size, beta)
