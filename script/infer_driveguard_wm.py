"""Run Path-A or full DriveGuard-WM inference without saving target fields."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset.temporal_dataset import TemporalDownscaleDataset
from models.driveguard_wm import Controller, load_model_from_checkpoint


def _state_dict(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint.get("model_state_dict", checkpoint)


def _load_full_components(model, student_path, path_b_path, controller_path, device):
    if not all((student_path, path_b_path, controller_path)):
        raise ValueError(
            "Full inference requires --student_checkpoint, --path_b_checkpoint, "
            "and --controller_checkpoint"
        )

    model.matrix_student.load_state_dict(_state_dict(student_path, device))

    path_b = torch.load(path_b_path, map_location=device)
    for name in (
        "path_b_encoder",
        "drive_interface",
        "process_moe",
        "local_ssm",
        "path_b_decoder",
    ):
        getattr(model, name).load_state_dict(path_b[name])

    controller = Controller().to(device)
    controller.load_state_dict(_state_dict(controller_path, device))
    model.controller = controller


def main():
    parser = argparse.ArgumentParser(description="DriveGuard-WM inference")
    parser.add_argument("--checkpoint", required=True, help="S1 world-model checkpoint")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--mode", default="path_a", choices=["path_a", "full"])
    parser.add_argument("--student_checkpoint")
    parser.add_argument("--path_b_checkpoint")
    parser.add_argument("--controller_checkpoint")
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Save normalized predictions instead of restoring physical units",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TemporalDownscaleDataset(
        args.data_dir, args.split, T=args.T, normalize=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = load_model_from_checkpoint(
        args.checkpoint, device=device, strict=False
    )
    if args.mode == "full":
        _load_full_components(
            model,
            args.student_checkpoint,
            args.path_b_checkpoint,
            args.controller_checkpoint,
            device,
        )
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = (dataset.hr_h, dataset.hr_w)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"DriveGuard-WM ({args.mode})"):
            lr_seq = batch["lr"].to(device, non_blocking=True)
            if args.mode == "path_a":
                prediction = model.forward_s1(
                    lr_seq, y_hr=None, target_size=target_size
                )["hr_pred"]
            else:
                prediction = model.infer(lr_seq, target_size=target_size)

            prediction = prediction.detach().cpu().numpy()
            if not args.normalized:
                prediction = prediction * dataset.hr_norm_std + dataset.hr_norm_mean

            for array, filename in zip(prediction, batch["filename"]):
                stem = Path(filename).stem
                np.save(output_dir / f"pred_{stem}.npy", array.astype(np.float32))

    print(f"Saved {len(dataset)} predictions to {output_dir}")


if __name__ == "__main__":
    main()
