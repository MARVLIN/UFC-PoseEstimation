#!/usr/bin/env python3
"""
Skeleton-only motion report for heuristic punch / kick / tactic_proxy sequences.

Uses the same pipeline as ``tactic_interactive_gallery`` (YOLO pose on val, then
``analyze_recording_keypoints`` + ``cluster_strike_combos``), but renders **only**
normalized stick figures (SVG) so you can inspect whether limb geometry separates
punches, kicks, and multi-strike clusters — without RGB distraction.

This does **not** prove classifier accuracy; it visualizes the **pose proxy** the
heuristics (and optional sequence models) see.
"""

from __future__ import annotations

import argparse
import html as html_mod
import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from val_pose_insights import (  # noqa: E402
    COCO_EDGES,
    cluster_strike_combos,
    collect_fight_frames,
    fight_ids_for_predict,
    predict_largest_kpts,
)
from webcam_pose import analyze_recording_keypoints  # noqa: E402

TACTIC_ORDER = [
    "punch_kick_chain_proxy",
    "multi_punch_proxy",
    "punch_proxy",
    "kick_proxy",
    "mixed_strike_proxy",
    "strike_proxy",
]


def _normalize_pose_xy(
    kpts: np.ndarray | None,
    conf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hip-centered, torso-scaled 2D coords; SVG-friendly (y increases downward like image)."""
    xy = np.zeros((17, 2), dtype=np.float64)
    vis = np.zeros(17, dtype=bool)
    if kpts is None or kpts.shape[0] < 17:
        return xy, vis
    k = kpts.astype(np.float64)
    for i in range(17):
        vis[i] = k[i, 2] >= conf
        xy[i] = k[i, :2]

    lh, rh, ls, rs = 11, 12, 5, 6
    hip = None
    if vis[lh] and vis[rh]:
        hip = 0.5 * (xy[lh] + xy[rh])
    elif vis[lh]:
        hip = xy[lh].copy()
    elif vis[rh]:
        hip = xy[rh].copy()
    else:
        hip = np.nanmedian(xy[:17], axis=0)

    sh = None
    if vis[ls] and vis[rs]:
        sh = 0.5 * (xy[ls] + xy[rs])
    elif vis[ls]:
        sh = xy[ls].copy()
    elif vis[rs]:
        sh = xy[rs].copy()
    else:
        sh = hip + np.array([0.0, -1.0], dtype=np.float64)

    scale = float(np.linalg.norm(sh - hip))
    if not np.isfinite(scale) or scale < 1e-3:
        bb = np.ptp(xy[vis], axis=0) if np.any(vis) else np.array([100.0, 100.0])
        scale = float(max(bb[0], bb[1], 50.0))

    out = (xy - hip) / scale
    return out, vis


def _svg_skeleton_panel(
    kpts: np.ndarray | None,
    conf: float,
    *,
    width_px: int = 200,
    height_px: int = 260,
    stroke: str = "#6ec8ff",
    joint_fill: str = "#ff6b9d",
    stroke_w: float = 0.07,
    joint_r: float = 0.055,
) -> str:
    xy, vis = _normalize_pose_xy(kpts, conf)
    vb = "-1.35 -2.15 2.7 4.0"
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" '
        f'width="{width_px}" height="{height_px}" '
        f'style="background:#16161e;border-radius:8px;border:1px solid #333">'
    ]
    for a, b in COCO_EDGES:
        if vis[a] and vis[b]:
            parts.append(
                f'<line x1="{xy[a, 0]:.4f}" y1="{xy[a, 1]:.4f}" '
                f'x2="{xy[b, 0]:.4f}" y2="{xy[b, 1]:.4f}" '
                f'stroke="{stroke}" stroke-width="{stroke_w}" stroke-linecap="round"/>'
            )
    for i in range(17):
        if vis[i]:
            parts.append(
                f'<circle cx="{xy[i, 0]:.4f}" cy="{xy[i, 1]:.4f}" r="{joint_r}" '
                f'fill="{joint_fill}"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _svg_overlay_panels(
    frames: list[np.ndarray | None],
    conf: float,
    *,
    width_px: int = 220,
    height_px: int = 260,
) -> str:
    """Several skeletons in the same coordinates with increasing opacity (motion ghost)."""
    vb = "-1.35 -2.15 2.7 4.0"
    n = max(1, len(frames))
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" '
        f'width="{width_px}" height="{height_px}" '
        f'style="background:#0d1117;border-radius:8px;border:1px solid #444">'
    ]
    for fi, kpts in enumerate(frames):
        alpha = 0.18 + 0.82 * (fi / max(n - 1, 1))
        stroke = f"rgba(110,200,255,{alpha:.2f})"
        joint = f"rgba(255,107,157,{min(1.0, alpha + 0.15):.2f})"
        xy, vis = _normalize_pose_xy(kpts, conf)
        g_op = 0.35 + 0.65 * (fi / max(n - 1, 1))
        parts.append(f'<g opacity="{g_op:.2f}">')
        for a, b in COCO_EDGES:
            if vis[a] and vis[b]:
                parts.append(
                    f'<line x1="{xy[a, 0]:.4f}" y1="{xy[a, 1]:.4f}" '
                    f'x2="{xy[b, 0]:.4f}" y2="{xy[b, 1]:.4f}" '
                    f'stroke="{stroke}" stroke-width="0.06" stroke-linecap="round"/>'
                )
        for i in range(17):
            if vis[i]:
                parts.append(
                    f'<circle cx="{xy[i, 0]:.4f}" cy="{xy[i, 1]:.4f}" r="0.05" fill="{joint}"/>'
                )
        parts.append("</g>")
    parts.append(
        '<text x="-1.25" y="-1.85" fill="#8ab4f8" font-size="0.18" font-family="system-ui,sans-serif">'
        "overlay</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _frame_indices_for_span(fs: int, fe: int, T: int, sequence_len: int) -> np.ndarray:
    fs = max(0, min(fs, T - 1))
    fe = max(0, min(fe, T - 1))
    if fe < fs:
        fs, fe = fe, fs
    span = fe - fs + 1
    if span >= sequence_len:
        return np.linspace(fs, fe, num=sequence_len, dtype=int)
    idxs = np.linspace(fs, fe, num=max(1, span), dtype=int)
    while len(idxs) < sequence_len:
        idxs = np.append(idxs, idxs[-1])
    return idxs[:sequence_len]


def _section_html(title: str, rows: list[str], intro: str) -> str:
    body = "\n".join(rows) if rows else '<p style="color:#888;">No examples in this section.</p>'
    return f"""
<section style="margin-bottom:36px;">
  <h2 style="color:#e8eaed;border-bottom:1px solid #333;padding-bottom:8px;">{title}</h2>
  <p style="color:#9aa0a6;max-width:920px;line-height:1.5;">{intro}</p>
  {body}
</section>
"""


def _example_row(
    label: str,
    idxs: np.ndarray,
    kpts_seq: list[np.ndarray | None],
    conf: float,
    *,
    show_overlay: bool,
) -> str:
    caps = " → ".join(f"f{int(i)}" for i in idxs)
    panels = []
    for ii in idxs:
        panels.append(_svg_skeleton_panel(kpts_seq[int(ii)], conf))
    overlay = ""
    if show_overlay:
        frames = [kpts_seq[int(ii)] for ii in idxs]
        overlay = _svg_overlay_panels(frames, conf)
    lab_esc = html_mod.escape(label)
    caps_esc = html_mod.escape(caps)
    flex = '<div style="display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;">'
    flex += "".join(panels)
    if overlay:
        flex += f'<div style="margin-left:8px;">{overlay}</div>'
    flex += "</div>"
    return f"""
<div style="margin-bottom:28px;padding:14px;background:#111318;border-radius:10px;">
  <div style="font-size:14px;color:#fdd663;margin-bottom:10px;"><b>{lab_esc}</b>
    <span style="color:#80868b;font-weight:normal;"> — {caps_esc}</span></div>
  {flex}
</div>
"""


def build_skeleton_motion_html(
    model: YOLO,
    data_yaml: Path,
    out_html: Path,
    *,
    split: str = "val",
    imgsz: int = 640,
    device: str | None = None,
    max_fights: int = 12,
    max_frames_per_fight: int = 150,
    max_predict_total: int = 600,
    max_examples_per_tactic: int = 3,
    max_peak_examples_per_fight: int = 3,
    max_peak_rows_total: int = 24,
    sequence_len: int = 7,
    kpt_conf: float = 0.35,
    fps_assumed: float = 30.0,
    combo_gap_frames: int = 12,
    peak_radius: int = 3,
    show_overlay: bool = True,
) -> Path:
    by_fight, _ = collect_fight_frames(Path(data_yaml), split)
    fight_ids, group_mode = fight_ids_for_predict(by_fight)
    fight_ids = fight_ids[:max_fights]

    by_tactic: dict[str, list[str]] = {t: [] for t in TACTIC_ORDER}
    other_tactic_rows: list[str] = []
    punch_rows: list[str] = []
    kick_rows: list[str] = []

    n_pred = 0
    n_punch_peak = 0
    n_kick_peak = 0
    for fid in fight_ids:
        pairs = by_fight[fid]
        if max_frames_per_fight is not None:
            pairs = pairs[:max_frames_per_fight]
        kpts_seq: list[np.ndarray | None] = []
        for _fi, path in pairs:
            if n_pred >= max_predict_total:
                break
            sk, _h, _w = predict_largest_kpts(model, path, device, imgsz)
            n_pred += 1
            kpts_seq.append(sk)

        if not kpts_seq:
            continue
        T = len(kpts_seq)

        events, _summary = analyze_recording_keypoints(
            kpts_seq,
            fps=fps_assumed,
            kpt_conf=kpt_conf,
            strikes_only=True,
        )
        combos = cluster_strike_combos(events, gap_frames=combo_gap_frames)

        for c in combos:
            fs, fe = int(c["frame_start"]), int(c["frame_end"])
            idxs = _frame_indices_for_span(fs, fe, T, sequence_len)
            label = f"{c['tactic_proxy']} | fight {fid[:40]} | events={c['n_events']}"
            row = _example_row(label, idxs, kpts_seq, kpt_conf, show_overlay=show_overlay)
            tag = c["tactic_proxy"]
            if tag in by_tactic:
                if len(by_tactic[tag]) < max_examples_per_tactic:
                    by_tactic[tag].append(row)
            else:
                if len(other_tactic_rows) < max_examples_per_tactic * 2:
                    other_tactic_rows.append(row)

        punch_evs = [e for e in events if e.get("kind") == "punch_candidate"]
        kick_evs = [e for e in events if e.get("kind") == "kick_candidate"]
        for e in punch_evs[:max_peak_examples_per_fight]:
            if n_punch_peak >= max_peak_rows_total:
                break
            t0 = int(e["frame_index"])
            side = e.get("side", "")
            idxs = np.array(
                [max(0, min(T - 1, t0 + d)) for d in range(-peak_radius, peak_radius + 1)],
                dtype=int,
            )
            if len(idxs) > sequence_len:
                idxs = np.linspace(idxs[0], idxs[-1], num=sequence_len, dtype=int)
            label = f"punch_candidate peak | fight {fid[:36]} | side {side} | t={t0}"
            punch_rows.append(_example_row(label, idxs, kpts_seq, kpt_conf, show_overlay=show_overlay))
            n_punch_peak += 1
        for e in kick_evs[:max_peak_examples_per_fight]:
            if n_kick_peak >= max_peak_rows_total:
                break
            t0 = int(e["frame_index"])
            side = e.get("side", "")
            idxs = np.array(
                [max(0, min(T - 1, t0 + d)) for d in range(-peak_radius, peak_radius + 1)],
                dtype=int,
            )
            if len(idxs) > sequence_len:
                idxs = np.linspace(idxs[0], idxs[-1], num=sequence_len, dtype=int)
            label = f"kick_candidate peak | fight {fid[:36]} | side {side} | t={t0}"
            kick_rows.append(_example_row(label, idxs, kpts_seq, kpt_conf, show_overlay=show_overlay))
            n_kick_peak += 1

    tactic_rows: list[str] = []
    for tac in TACTIC_ORDER:
        tactic_rows.extend(by_tactic[tac])
    tactic_rows.extend(other_tactic_rows)

    disclaimer = (
        "Normalized view: hips centered, scale ≈ shoulder–hip distance. "
        "COCO-17 stick figure from <b>predicted</b> keypoints. "
        "Sections reflect pose <b>heuristics</b> (extension + angles), not ground-truth strike counts."
    )
    meta = (
        f"split={html_mod.escape(split)} | fight grouping={html_mod.escape(group_mode)} | "
        f"frames predicted (cap)={n_pred} | kpt_conf={kpt_conf}"
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Skeleton motion — punch / kick / tactic proxy</title>
  <style>
    body {{ background:#0e0e10;color:#e8eaed;font-family:system-ui,sans-serif;margin:0;padding:20px 24px 48px; }}
    code {{ background:#2d2d30;padding:2px 6px;border-radius:4px; }}
  </style>
</head>
<body>
  <h1 style="margin-top:0;">Skeleton-only sequences</h1>
  <p style="color:#9aa0a6;max-width:960px;">{disclaimer}</p>
  <p style="color:#80868b;font-size:13px;">{meta}</p>

  {_section_html(
      "A. Tactic proxy clusters (multi-strike windows)",
      tactic_rows,
      "Each row: sampled frames across a clustered window (same <code>tactic_proxy</code> tags as "
      "<code>cluster_strike_combos</code>). Last panel (if present) superimposes all frames as a motion ghost.",
  )}
  {_section_html(
      "B. Punch candidate peaks (local windows)",
      punch_rows,
      "Short temporal window around each <code>punch_candidate</code> event (arm extension + elbow angle heuristic). "
      "Inspect whether arms extend in a consistent pattern.",
  )}
  {_section_html(
      "C. Kick candidate peaks (local windows)",
      kick_rows,
      "Short temporal window around each <code>kick_candidate</code> event (leg extension + knee angle heuristic).",
  )}
</body>
</html>
"""

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")
    print("Wrote", out_html)
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Skeleton-only HTML report for strike/tactic heuristics")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("skeleton_motion_report.html"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--no-overlay", action="store_true", help="Disable motion-ghost overlay panel")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    dev = args.device or None
    build_skeleton_motion_html(
        model,
        args.data.resolve(),
        args.out.resolve(),
        split=args.split,
        imgsz=args.imgsz,
        device=dev,
        show_overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    main()
