"""
DriveGuard-WM 模型加载器 — 从 checkpoint 恢复完整模型。

用法:
    from models.driveguard_wm.loader import load_model_from_checkpoint

    # 方式 1: 从 .pth 文件加载 (推荐)
    model = load_model_from_checkpoint("outputs/s1_conus/best_model_*.pth")

    # 方式 2: 从实验目录加载 (自动找 best checkpoint)
    model = load_model_from_checkpoint("outputs/s1_conus")

    # 方式 3: 直接加载完整模型对象 (model.pt 需要同环境)
    model = torch.load("outputs/s1_conus/model.pt")
"""
import os
import importlib
import torch


def load_model_from_checkpoint(path, device=None, strict=True):
    """
    从 checkpoint 恢复完整 DriveGuard-WM 模型。

    支持:
      - .pth checkpoint 文件 (含 model_class + model_kwargs + state_dict)
      - 实验目录 (自动找 best_model_*.pth)
      - model.pt 完整模型文件

    Args:
        path: checkpoint 路径或实验目录路径
        device: 设备 (default: cpu)
        strict: 是否严格加载 state_dict (default: True)

    Returns:
        model: DriveGuardWM 实例, 已加载权重, eval 模式
    """
    device = device or 'cpu'
    checkpoint = _load_checkpoint(path)

    # ── 完整模型对象 (model.pt) ─────────────────────────────────
    if not isinstance(checkpoint, dict):
        model = checkpoint.to(device)
        model.eval()
        return model

    # ── 从 model_class + model_kwargs 重建 ──────────────────────
    model_class_path = checkpoint.get('model_class',
                                      'models.driveguard_wm.DriveGuardWM')
    model_kwargs = checkpoint.get('model_kwargs', {})
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 动态导入并构造模型
    module_path, class_name = model_class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    model = ModelClass(**model_kwargs)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    return model


def _load_checkpoint(path):
    """统一加载 checkpoint 入口。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # 如果是目录，自动找 checkpoint 文件
    if os.path.isdir(path):
        # 优先 model.pt
        model_pt = os.path.join(path, 'model.pt')
        if os.path.exists(model_pt):
            return torch.load(model_pt, map_location='cpu')

        # 其次 best_model_*.pth
        import glob
        pth_files = sorted(glob.glob(os.path.join(path, 'best_model_*.pth')))
        if pth_files:
            return torch.load(pth_files[-1], map_location='cpu')

        # 最后任意 .pth
        pth_files = sorted(glob.glob(os.path.join(path, '*.pth')))
        if pth_files:
            return torch.load(pth_files[-1], map_location='cpu')

        raise FileNotFoundError(f"No checkpoint found in directory: {path}")

    # 单个文件
    ckpt = torch.load(path, map_location='cpu')

    # 如果 checkpoint 包含嵌套的 state_dict
    if isinstance(ckpt, dict):
        sd = ckpt.get('model_state_dict', ckpt)
        # 检查是否有完整的模型结构信息
        if 'model_class' in ckpt or 'model_kwargs' in ckpt:
            return ckpt
        # 只有 state_dict → 尝试周围目录找 model.pt 或 config.json
        dir_path = os.path.dirname(os.path.abspath(path))
        model_pt = os.path.join(dir_path, 'model.pt')
        if os.path.exists(model_pt):
            return torch.load(model_pt, map_location='cpu')
        # 尝试从 config.json 重建
        config_path = os.path.join(dir_path, 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            model_kwargs = {k: v for k, v in cfg.items()
                          if k in ('in_channels', 'K', 'N', 'Dg', 'Dm', 'Dl',
                                   'Ds', 'Dc', 'hidden_dim', 'patch_stride',
                                   'relation_dim', 'contrastive_dim',
                                   'num_experts', 'top_k', 'window_size',
                                   'window_overlap', 'ssm_dim',
                                   'use_disagreement', 'gate_temperature',
                                   'beta')}
            return {
                'model_class': cfg.get('model_class',
                                       'models.driveguard_wm.DriveGuardWM'),
                'model_kwargs': model_kwargs,
                'model_state_dict': sd,
            }

    return ckpt
