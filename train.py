"""Optional Step 6: fine-tune YOLO pose on the generated MMA dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


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
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(
            f"Missing {args.data}. Run prepare_pose_dataset.py first, or pass --data /path/to/data.yaml"
        )

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
