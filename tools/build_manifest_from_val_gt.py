#!/usr/bin/env python3
"""
Build per-fight (or flat) GT pose sequences from a YOLO pose val split + manifest CSV template.

For each fight group (stem before `_mp4-<digits>`), sorts frames by index, takes the **first**
labeled person per frame, stacks to (T, 17, 3) with visibility as third channel (0/1/2 → /2.0).

All rows get label `unlabeled` and split by hash — **edit the CSV** before training.

Usage:
  python tools/build_manifest_from_val_gt.py --data-yaml /path/to/data.yaml --out-dir ./action_clips
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from action_sequence_utils import K, parse_fight_and_frame

_RF = re.compile(r"^(.*)\.rf\.[a-f0-9]+$", re.I)


def read_pose_labels(label_path: Path) -> list[dict]:
    if not label_path.is_file():
        return []
    out: list[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5 + K * 3:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        rest = list(map(float, parts[5 : 5 + K * 3]))
        xy = np.array(rest[0::3], dtype=np.float64).reshape(-1, 1)
        yy = np.array(rest[1::3], dtype=np.float64).reshape(-1, 1)
        vis = np.array(rest[2::3], dtype=np.int64)
        xy = np.hstack([xy, yy])
        out.append({"bbox": (cx, cy, w, h), "xy": xy, "vis": vis})
    return out


def _dataset_root(data_yaml: Path) -> Path:
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    p = cfg.get("path")
    if p:
        root = Path(p)
        return root if root.is_absolute() else (data_yaml.parent / root).resolve()
    return data_yaml.parent.resolve()


def _val_images_root(data_yaml: Path) -> Path:
    root = _dataset_root(data_yaml)
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rel = cfg.get("val") or cfg.get("valid")
    if not rel:
        raise SystemExit("data.yaml needs val or valid")
    p = Path(str(rel))
    return p if p.is_absolute() else (root / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-frames", type=int, default=8)
    args = ap.parse_args()

    img_root = _val_images_root(args.data_yaml)
    lbl_root = img_root.parent / "labels"
    if not lbl_root.is_dir():
        raise SystemExit(f"Missing labels dir: {lbl_root}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = sorted([p for p in img_root.rglob("*") if p.suffix.lower() in exts])
    by_fight: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    orphans: list[Path] = []
    for p in images:
        fid, fidx = parse_fight_and_frame(p.stem)
        if fidx < 0:
            orphans.append(p)
        else:
            by_fight[fid].append((fidx, p))
    for fid in by_fight:
        by_fight[fid].sort(key=lambda x: x[0])

    out_dir = args.out_dir.resolve()
    seq_dir = out_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def process_group(gid: str, pairs: list[tuple[int, Path]]) -> None:
        pairs = sorted(pairs, key=lambda x: (x[0], x[1].name))
        frames: list[np.ndarray] = []
        for _fi, imp in pairs:
            stem = imp.stem
            m = _RF.match(stem)
            stem_alt = m.group(1) if m else stem
            lp = lbl_root / f"{stem}.txt"
            if not lp.is_file():
                lp = lbl_root / f"{stem_alt}.txt"
            insts = read_pose_labels(lp)
            if not insts:
                continue
            ins = insts[0]
            xy = ins["xy"].astype(np.float32)
            vis = ins["vis"].astype(np.float32)
            triple = np.concatenate([xy, (vis / 2.0)[:, None]], axis=1)
            frames.append(triple)
        if len(frames) < args.min_frames:
            return
        arr = np.stack(frames, axis=0)
        safe = re.sub(r"[^\w\-]+", "_", gid)[:80]
        clip_id = f"{safe}_{len(arr)}f"
        npy_path = seq_dir / f"{clip_id}.npy"
        np.save(npy_path, arr)
        h = hash(gid) % 2
        split = "val" if h == 0 else "train"
        rows.append(
            {
                "clip_id": clip_id,
                "label": "unlabeled",
                "npy_path": str(npy_path.resolve()),
                "split": split,
                "img_h": "720",
                "img_w": "1280",
            }
        )

    for fid, pairs in sorted(by_fight.items()):
        process_group(fid, pairs)

    if orphans:
        orphans.sort(key=lambda x: x.name)
        process_group("__flat_orphans__", [(-1, p) for p in orphans])

    manifest = out_dir / "manifest_template.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "label", "npy_path", "split", "img_h", "img_w"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} sequences under {seq_dir}")
    print(f"Manifest template: {manifest}")
    print("Edit `label` column (e.g. jab, hook, other) before LSTM / MMAction2 export.")


if __name__ == "__main__":
    main()
