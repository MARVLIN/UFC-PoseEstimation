#!/usr/bin/env python3
"""
Build a YOLO pose dataset from MMA Fighter Detection V1 (DOI 10.17632/c456bnk8bm.1):

1. Load pretrained YOLO11x-pose
2. Run pose inference on every image
3. Match each V1 bbox to a pose detection (IoU), keep GT box, zero visibility for keypoints outside that box
4. Copy images unchanged into the output layout
5. Optionally verify label files and counts

Expects the unzipped Mendeley layout with data.yaml and per-split folders (train/valid/test),
each with images/ and labels/ (YOLO detection lines: class cx cy w h, normalized).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from ultralytics import YOLO

# COCO pose keypoint flip order used by Ultralytics pretrained pose models
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
K = 17
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _dataset_root(cfg_path: Path, cfg: dict) -> Path:
    p = cfg.get("path")
    if p is None or p == "":
        return cfg_path.parent
    root = Path(p)
    return root if root.is_absolute() else (cfg_path.parent / root).resolve()


def _resolve_split_dir(root: Path, rel: str) -> Path:
    return (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()


def images_to_labels_dir(image_dir: Path) -> Path:
    if image_dir.name == "images":
        return image_dir.parent / "labels"
    sibling = image_dir.parent / "labels"
    if sibling.is_dir():
        return sibling
    raise FileNotFoundError(f"Cannot infer labels directory next to {image_dir}")


def discover_splits(data_yaml: Path) -> dict[str, Path]:
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = _dataset_root(data_yaml, cfg)
    out: dict[str, Path] = {}
    if "train" in cfg and cfg["train"]:
        out["train"] = _resolve_split_dir(root, str(cfg["train"]))
    val_key = "val" if cfg.get("val") else ("valid" if cfg.get("valid") else None)
    if val_key:
        out["val"] = _resolve_split_dir(root, str(cfg[val_key]))
    if cfg.get("test"):
        out["test"] = _resolve_split_dir(root, str(cfg["test"]))
    if "train" not in out:
        raise ValueError(f"No train split in {data_yaml}")
    return out


def iter_images(image_root: Path) -> Iterator[Path]:
    for p in sorted(image_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def read_detection_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    """Return list of (cx, cy, w, h) normalized; ignores class id."""
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h, *_ = parts
        boxes.append((float(cx), float(cy), float(w), float(h)))
    return boxes


def cxcywhn_to_xyxyn(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def iou_xyxyn(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    gt_xyxyn: list[tuple[float, float, float, float]],
    pose_xyxyn: np.ndarray,
    iou_min: float,
) -> list[int | None]:
    """Return list aligned with gt: index of matched pose row or None."""
    if not gt_xyxyn or pose_xyxyn.size == 0:
        return [None] * len(gt_xyxyn)
    n_g, n_p = len(gt_xyxyn), pose_xyxyn.shape[0]
    ious = np.zeros((n_g, n_p), dtype=np.float64)
    for i, g in enumerate(gt_xyxyn):
        for j in range(n_p):
            b = pose_xyxyn[j]
            ious[i, j] = iou_xyxyn(g, tuple(b.tolist()))
    used_p = set()
    used_g = set()
    pairs: list[tuple[int, int, float]] = []
    for gi in range(n_g):
        for pj in range(n_p):
            if ious[gi, pj] >= iou_min:
                pairs.append((gi, pj, ious[gi, pj]))
    pairs.sort(key=lambda t: t[2], reverse=True)
    match_g_to_p: dict[int, int] = {}
    for gi, pj, _ in pairs:
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
        match_g_to_p[gi] = pj
    return [match_g_to_p.get(i) for i in range(n_g)]


def keypoint_visibilities(
    xyn: np.ndarray,
    kpt_conf: np.ndarray | None,
    gt_cx: float,
    gt_cy: float,
    gt_w: float,
    gt_h: float,
    kpt_conf_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    xyn: (17, 2) normalized; returns vis int [17] and xyn (unchanged copy).
    vis 2 if inside GT box and (conf ok); 0 otherwise.
    """
    x1, y1, x2, y2 = cxcywhn_to_xyxyn(gt_cx, gt_cy, gt_w, gt_h)
    vis = np.zeros(K, dtype=np.int64)
    xy = xyn.copy()
    for i in range(K):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        inside = x1 <= x <= x2 and y1 <= y <= y2
        conf_ok = True
        if kpt_conf is not None and kpt_conf.size > i:
            conf_ok = float(kpt_conf[i]) >= kpt_conf_thr
        if inside and conf_ok:
            vis[i] = 2
    return vis, xy


def format_pose_line(
    cls: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    xyn: np.ndarray,
    vis: np.ndarray,
) -> str:
    parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for i in range(K):
        parts.append(f"{float(xyn[i, 0]):.6f}")
        parts.append(f"{float(xyn[i, 1]):.6f}")
        parts.append(str(int(vis[i])))
    return " ".join(parts)


def process_batch(
    model: YOLO,
    items: list[tuple[Path, Path, Path, Path]],
    iou_min: float,
    kpt_conf_thr: float,
) -> None:
    """items: (src_image, dst_image, dst_label, v1_label_path)"""
    paths = [str(t[0]) for t in items]
    results = model.predict(paths, verbose=False, imgsz=640)
    for (src_img, dst_img, dst_lbl, label_src), r in zip(items, results, strict=True):
        gts = read_detection_boxes(label_src)
        lines: list[str] = []
        if gts and r.boxes is not None and len(r.boxes) and r.keypoints is not None and len(r.keypoints):
            pose_xyxyn = r.boxes.xyxyn.cpu().numpy()
            xyn_all = r.keypoints.xyn.cpu().numpy()
            conf_all = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
            gt_xy = [cxcywhn_to_xyxyn(*g) for g in gts]
            matches = greedy_match(gt_xy, pose_xyxyn, iou_min)
            for gi, gt in enumerate(gts):
                pj = matches[gi]
                if pj is None:
                    continue
                xyn = xyn_all[pj]
                kc = conf_all[pj] if conf_all is not None else None
                vis, xy = keypoint_visibilities(xyn, kc, gt[0], gt[1], gt[2], gt[3], kpt_conf_thr)
                lines.append(format_pose_line(0, gt[0], gt[1], gt[2], gt[3], xy, vis))
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        shutil.copy2(src_img, dst_img)


def write_output_data_yaml(out_root: Path, has_test: bool) -> None:
    test_line = "test: test/images\n" if has_test else ""
    text = f"""# Generated from MMA Fighter Detection V1 + YOLO11x-pose pseudo-labels
path: {out_root.resolve()}
train: train/images
val: val/images
{test_line}
names:
  0: fighter
kpt_shape: [17, 3]
flip_idx: {FLIP_IDX}
"""
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def verify_dataset(out_root: Path) -> int:
    yaml_path = out_root / "data.yaml"
    if not yaml_path.is_file():
        print("VERIFY FAIL: missing data.yaml", file=sys.stderr)
        return 1
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    expected = 5 + K * 3
    rc = 0
    for split in ("train", "val", "test"):
        img_dir = out_root / split / "images"
        lbl_dir = out_root / split / "labels"
        if not img_dir.is_dir():
            if split == "test":
                continue
            print(f"VERIFY WARN: missing image dir {img_dir}")
            continue
        imgs = {p.stem for p in iter_images(img_dir)}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else set()
        missing_lbl = imgs - lbls
        orphan_lbl = lbls - imgs
        if missing_lbl:
            print(f"VERIFY WARN: {split} {len(missing_lbl)} images without .txt (first 5): {list(sorted(missing_lbl))[:5]}")
        if orphan_lbl:
            print(f"VERIFY WARN: {split} {len(orphan_lbl)} labels without image")
        bad_lines = 0
        for stem in sorted(lbls & imgs):
            txt = (lbl_dir / f"{stem}.txt").read_text(encoding="utf-8")
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                n = len(line.split())
                if n != expected:
                    bad_lines += 1
                    print(f"VERIFY FAIL: {split}/{stem}.txt wrong field count {n} != {expected}")
                    rc = 1
        if bad_lines == 0:
            print(f"VERIFY OK: {split} images={len(imgs)} labels={len(lbls)} fields_per_line={expected}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="MMA V1 bbox + YOLO11x-pose → YOLO pose dataset")
    ap.add_argument(
        "--data-yaml",
        type=Path,
        required=True,
        help="Path to Version 1 data.yaml inside the unzipped Mendeley dataset",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("mma_pose_dataset"),
        help="Output dataset root (receives data.yaml, train|val|test/images|labels)",
    )
    ap.add_argument("--model", type=str, default="yolo11x-pose.pt", help="Ultralytics pose weights")
    ap.add_argument("--batch", type=int, default=4, help="Inference batch size (reduce if OOM)")
    ap.add_argument("--iou-min", type=float, default=0.25, help="Min IoU to assign pose det to a V1 box")
    ap.add_argument("--kpt-conf", type=float, default=0.25, help="Min per-keypoint conf when visible")
    ap.add_argument("--verify-only", action="store_true", help="Only run Step 5 on existing --output")
    ap.add_argument("--skip-verify", action="store_true", help="Skip verification after export")
    args = ap.parse_args()

    out_root = args.output.resolve()
    if args.verify_only:
        return verify_dataset(out_root)

    if not args.data_yaml.is_file():
        print(f"Missing {args.data_yaml}", file=sys.stderr)
        return 1

    splits = discover_splits(args.data_yaml.resolve())
    print("Splits (image roots):", {k: str(v) for k, v in splits.items()})
    resolved_roots = [str(v.resolve()) for v in splits.values()]
    if len(resolved_roots) != len(set(resolved_roots)):
        print(
            "WARN: two or more splits use the same image folder; those images are inferred twice.",
            file=sys.stderr,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    split_alias = {"train": "train", "val": "val", "test": "test"}

    print(f"Loading model {args.model} …")
    model = YOLO(args.model)

    total = 0
    has_test_out = False
    for split_name, img_root in splits.items():
        out_split = split_alias.get(split_name, split_name)
        if out_split == "test":
            has_test_out = True
        labels_root = images_to_labels_dir(img_root)
        dst_img_root = out_root / out_split / "images"
        dst_lbl_root = out_root / out_split / "labels"
        dst_img_root.mkdir(parents=True, exist_ok=True)
        dst_lbl_root.mkdir(parents=True, exist_ok=True)

        n_split = 0
        batch: list[tuple[Path, Path, Path, Path]] = []
        for src_img in iter_images(img_root):
            rel = src_img.relative_to(img_root)
            dst_img = dst_img_root / rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl = dst_lbl_root / rel.with_suffix(".txt")
            label_src = labels_root / f"{src_img.stem}.txt"
            batch.append((src_img, dst_img, dst_lbl, label_src))
            if len(batch) >= args.batch:
                process_batch(model, batch, args.iou_min, args.kpt_conf)
                n_split += len(batch)
                total += len(batch)
                batch.clear()
                print(f"  {out_split}: {n_split} images …", flush=True)
        if batch:
            process_batch(model, batch, args.iou_min, args.kpt_conf)
            n_split += len(batch)
            total += len(batch)
            batch.clear()
        print(f"  {out_split} done: {n_split} images")

    write_output_data_yaml(out_root, has_test_out)
    print(f"Done. Wrote dataset under {out_root} ({total} images processed).")

    if not args.skip_verify:
        print("--- Verification ---")
        v = verify_dataset(out_root)
        if v != 0:
            return v
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
