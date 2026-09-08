"""
DriveGuard-WM: Hierarchical World Model with Dual-Path Blending.

Architecture (spec section 2):
  Path A:  Hierarchical World Model (G → M → L → y_a)
  Matrix:  LR-only D/R prediction (teacher + student)
  Path B:  Process MoE + Windowed Local SSM (y_b)
  Ctrl:    Monotone Controller blends y_a and y_b via gate

Stage-based training (spec section 14):
  S1: World Model (tokenizer + G/M/L prior/post + Path A)
  S2: Relation intervention
  S3: D/R Teacher
  S4: LR-only Matrix Student
  S5: Path B (MoE + SSM + decoder)
  S6: Controller
"""
from models.driveguard_wm.state_types import (
    GaussianState, HierarchicalState, PredictionBundle,
)
from models.driveguard_wm.tokenizers import (
    GlobalEncoder, GlobalTokenizer,
    LocalEncoder, LocalTokenizer,
    tokens_to_map, relation_weights_to_map, messages_to_map,
    scalar_tokens_to_map, pool_with_assignment, broadcast_global_context,
)
from models.driveguard_wm.global_rssm import GlobalRSSM, GlobalRSSMCell
from models.driveguard_wm.cross_scale_message import CrossScalePrior, CrossScalePosterior
from models.driveguard_wm.local_rssm import LocalRSSM, LocalRSSMCell
from models.driveguard_wm.posterior import HRObservationEncoder
from models.driveguard_wm.path_a_decoder import PathADecoder
from models.driveguard_wm.matrix_teacher import MatrixTeacher
from models.driveguard_wm.matrix_student import MatrixStudent
from models.driveguard_wm.drive_interface import DriveInterface, DriveBottleneck
from models.driveguard_wm.process_moe import ProcessMoE, ProcessExpert, MoERouter
from models.driveguard_wm.local_spatial_ssm import LocalSpatialSSM
from models.driveguard_wm.path_b_decoder import PathBDecoder
from models.driveguard_wm.controller import Controller
from models.driveguard_wm.driveguard_wm import DriveGuardWM, PathBEncoder
from models.driveguard_wm.loader import load_model_from_checkpoint

__all__ = [
    # State types
    'GaussianState', 'HierarchicalState', 'PredictionBundle',
    # Tokenizers
    'GlobalEncoder', 'GlobalTokenizer',
    'LocalEncoder', 'LocalTokenizer',
    'tokens_to_map', 'relation_weights_to_map', 'messages_to_map',
    'scalar_tokens_to_map', 'pool_with_assignment', 'broadcast_global_context',
    # RSSM
    'GlobalRSSM', 'GlobalRSSMCell',
    'CrossScalePrior', 'CrossScalePosterior',
    'LocalRSSM', 'LocalRSSMCell',
    # Posterior
    'HRObservationEncoder',
    # Decoders
    'PathADecoder', 'PathBDecoder',
    # Matrix
    'MatrixTeacher', 'MatrixStudent',
    # Path B components
    'DriveInterface', 'DriveBottleneck',
    'ProcessMoE', 'ProcessExpert', 'MoERouter',
    'LocalSpatialSSM',
    # Controller
    'Controller',
    # Top-level
    'DriveGuardWM', 'PathBEncoder',
    # Loader
    'load_model_from_checkpoint',
]
