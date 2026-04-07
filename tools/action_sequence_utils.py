"""Shared helpers for pose sequences (T, 17, C) used by LSTM export and MMAction2 conversion."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

K = 17
LH, RH, LS, RS = 11, 12, 5, 6

_MP4_FRAME_RE = re.compile(r"_mp4-(\d+)")


def label_stem_from_image_stem(stem: str) -> str:
    m = re.match(r"^(.*)\.rf\.[a-f0-9]+$", stem, re.I)
    return m.group(1) if m else stem


def parse_fight_and_frame(stem: str) -> tuple[str, int]:
    base = label_stem_from_image_stem(stem)
    if "_mp4-" not in base:
        return base, -1
    m = _MP4_FRAME_RE.search(base)
    if not m:
        left, _, _ = base.partition("_mp4-")
        return left, -1
    return base[: m.start()], int(m.group(1))


def normalize_sequence_xyv(
    xy: np.ndarray,
    vis: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Center on mid-hip per frame, scale by median shoulder width. Returns xy_norm (T,17,2), score (T,17)."""
    t = xy.shape[0]
    if vis is None:
        vis = np.ones((t, K), dtype=np.float64)
    else:
        vis = np.asarray(vis, dtype=np.float64)
        if vis.ndim == 1:
            vis = np.broadcast_to(vis[None, :], (t, K))
    xy = np.asarray(xy, dtype=np.float64)
    out = np.zeros_like(xy)
    mid = (xy[:, LH, :] + xy[:, RH, :]) / 2.0
    xy_c = xy - mid[:, None, :]
    ls, rs = xy_c[:, LS, :], xy_c[:, RS, :]
    d = np.linalg.norm(ls - rs, axis=-1)
    good = d > eps
    s = float(np.nanmedian(d[good])) if np.any(good) else 1.0
    if not np.isfinite(s) or s < eps:
        s = 1.0
    out = xy_c / s
    score = np.clip(vis / 2.0, 0.0, 1.0)
    return out, score


def read_manifest_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


MANIFEST_COLUMNS_DOC = """
CSV columns (header required):
  clip_id     — unique string id (becomes MMAction2 frame_dir)
  label       — class name (string) or integer class id
  npy_path    — path to .npy array shaped (T, 17, 2) or (T, 17, 3); if 3 channels, last is treated as score
  split       — train | val
  img_h, img_w — optional ints for img_shape in MMAction2 pickle (default 720 1280)
"""
