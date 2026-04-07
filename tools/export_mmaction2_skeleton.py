#!/usr/bin/env python3
"""
Build MMAction2 PoseDataset pickle from a CSV manifest of per-clip .npy sequences.

Expected .npy shapes:
  (T, 17, 2) — x,y normalized (e.g. full-image 0..1); scores default to 1.0
  (T, 17, 3) — x, y, score

Output matches MMAction2 docs:
  dict with keys 'split' (train/val lists of clip ids) and 'annotations' (list of dicts
  with frame_dir, total_frames, img_shape, original_shape, label, keypoint [M,T,V,C],
  keypoint_score [M,T,V]).

Usage:
  python tools/export_mmaction2_skeleton.py --manifest clips.csv --out-dir data/skeleton_custom
Writes:
  mmaction_custom.pkl, label_map.txt
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from action_sequence_utils import MANIFEST_COLUMNS_DOC, normalize_sequence_xyv, read_manifest_csv


def load_clip_npy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.load(path)
    if a.ndim != 3 or a.shape[1] != 17:
        raise ValueError(f"{path}: expected (T, 17, C), got {a.shape}")
    if a.shape[2] == 2:
        xy = a.astype(np.float32)
        score = np.ones((xy.shape[0], 17), dtype=np.float32)
    elif a.shape[2] == 3:
        xy = a[:, :, :2].astype(np.float32)
        score = a[:, :, 2].astype(np.float32)
    else:
        raise ValueError(f"{path}: last dim must be 2 or 3, got {a.shape[2]}")
    return xy, score


def main() -> None:
    parser = argparse.ArgumentParser(description="Manifest CSV → MMAction2 skeleton pkl")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest (see --help-columns)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Mid-hip center + shoulder-width scale per clip (xy only; scores unchanged)",
    )
    parser.add_argument(
        "--pkl-name",
        type=str,
        default="mmaction_custom.pkl",
        help="Output pickle filename",
    )
    parser.add_argument("--help-columns", action="store_true", help="Print manifest column docs and exit")
    args = parser.parse_args()
    if args.help_columns:
        print(MANIFEST_COLUMNS_DOC)
        raise SystemExit(0)

    rows = read_manifest_csv(args.manifest)
    if not rows:
        raise SystemExit(f"Empty manifest: {args.manifest}")

    # Resolve string labels → int
    raw_labels = []
    for row in rows:
        lab = row.get("label", "").strip()
        try:
            raw_labels.append(int(lab))
        except ValueError:
            raw_labels.append(lab)
    str_labels = sorted({str(x) for x in raw_labels if isinstance(x, str)})
    int_labels = sorted({x for x in raw_labels if isinstance(x, int)})
    if str_labels and int_labels:
        raise SystemExit("Manifest mixes string and int labels; use one style.")
    if str_labels:
        name_to_id = {n: i for i, n in enumerate(str_labels)}
        label_map_lines = [f"{n}\t{i}" for n, i in name_to_id.items()]
    else:
        name_to_id = {str(i): i for i in int_labels}
        label_map_lines = [f"class_{i}\t{i}" for i in int_labels]

    split_ids: dict[str, list[str]] = defaultdict(list)
    annotations: list[dict] = []

    for row in rows:
        clip_id = row["clip_id"].strip()
        sp = row.get("split", "train").strip().lower()
        if sp not in ("train", "val"):
            raise SystemExit(f"split must be train|val, got {sp!r} for {clip_id}")
        npy_path = Path(row["npy_path"]).expanduser().resolve()
        lab_raw = row["label"].strip()
        try:
            lab_int = int(lab_raw)
        except ValueError:
            lab_int = name_to_id[lab_raw]

        xy, score = load_clip_npy(npy_path)
        if args.normalize:
            xy, score = normalize_sequence_xyv(xy, score)

        t, v, _ = xy.shape
        h = int(row.get("img_h") or 720)
        w = int(row.get("img_w") or 1280)
        kp = xy[None, ...]  # (1, T, 17, 2)
        kps = score[None, ...]  # (1, T, 17)

        annotations.append(
            {
                "frame_dir": clip_id,
                "total_frames": int(t),
                "img_shape": (h, w),
                "original_shape": (h, w),
                "label": int(lab_int),
                "keypoint": kp.astype(np.float32),
                "keypoint_score": kps.astype(np.float32),
            }
        )
        split_ids[sp].append(clip_id)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / args.pkl_name
    blob = {"split": dict(split_ids), "annotations": annotations}
    with open(pkl_path, "wb") as f:
        pickle.dump(blob, f, protocol=4)

    map_path = out_dir / "label_map.txt"
    map_path.write_text("\n".join(label_map_lines) + "\n", encoding="utf-8")

    print(f"Wrote {pkl_path} ({len(annotations)} clips)")
    print(f"Wrote {map_path}")
    print("Splits:", {k: len(v) for k, v in split_ids.items()})


if __name__ == "__main__":
    main()
