"""Optional Step 6: fine-tune YOLO pose on the generated MMA dataset."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

from ultralytics import YOLO

from dev_device import default_mps_device

_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_DEFAULT_BATCH = 4 if _APPLE_SILICON else 8


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO pose on mma_pose_dataset (or custom data.yaml)")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "mma_pose_dataset" / "data.yaml",
        help="Pose dataset yaml (from prepare_pose_dataset.py)",
    )
    parser.add_argument("--model", type=str, default="yolo11x-pose.pt", help="Checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=_DEFAULT_BATCH,
        help=f"Batch size (default {_DEFAULT_BATCH} on this machine for Apple Silicon dev)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto = MPS on Apple Silicon when available; or cpu, mps, 0, cuda:0, …",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU (overrides --device)")
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if _APPLE_SILICON else None,
        help="Dataloader workers (default 0 on macOS arm64 to avoid multiprocessing issues)",
    )
    args = parser.parse_args()

    if args.cpu:
        args.device = "cpu"

    dev = default_mps_device() if args.device == "auto" else args.device

    if not args.data.is_file():
        raise SystemExit(
            f"Missing {args.data}. Run prepare_pose_dataset.py first, or pass --data /path/to/data.yaml"
        )

    model = YOLO(args.model)
    train_kw: dict = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
    }
    if dev is not None and dev != "":
        train_kw["device"] = dev
    if args.workers is not None:
        train_kw["workers"] = args.workers
    model.train(**train_kw)


if __name__ == "__main__":
    main()
