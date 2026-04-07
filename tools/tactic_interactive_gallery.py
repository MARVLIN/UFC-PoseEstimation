#!/usr/bin/env python3
"""
Build an interactive HTML report:
  • Plotly timeline of heuristic punch/kick candidates on val predictions (per fight).
  • Tactic sequences as embedded JPEG ``<img>`` data-URIs (reliable in Colab; Plotly Image subplots often blank there).

Heuristics match ``webcam_pose.analyze_recording_keypoints`` + ``cluster_strike_combos`` — not ground-truth tactics.
"""

from __future__ import annotations

import argparse
import base64
import html
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from val_pose_insights import (  # noqa: E402
    cluster_strike_combos,
    collect_fight_frames,
    fight_ids_for_predict,
    predict_largest_kpts,
)
from webcam_pose import analyze_recording_keypoints  # noqa: E402

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError as e:
    raise ImportError("pip install plotly") from e

TACTIC_ORDER = [
    "punch_kick_chain_proxy",
    "multi_punch_proxy",
    "punch_proxy",
    "kick_proxy",
    "mixed_strike_proxy",
    "strike_proxy",
]

TACTIC_COLORS = {
    "punch_kick_chain_proxy": "#e377c2",
    "multi_punch_proxy": "#ff7f0e",
    "punch_proxy": "#d62728",
    "kick_proxy": "#9467bd",
    "mixed_strike_proxy": "#17becf",
    "strike_proxy": "#7f7f7f",
    "punch_candidate": "#d62728",
    "kick_candidate": "#9467bd",
}


def _resize_bgr(bgr: np.ndarray, max_w: int = 360) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr
    s = max_w / w
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _draw_skeleton(bgr: np.ndarray, sk: np.ndarray | None, kpt_conf: float) -> np.ndarray:
    canvas = bgr.copy()
    if sk is None:
        return canvas
    h, w = canvas.shape[:2]
    lw = max(2, int(round((h + w) / 400)))
    ann = Annotator(canvas, line_width=lw)
    ann.kpts(sk, shape=(h, w), kpt_line=True, conf_thres=kpt_conf, radius=max(2, lw))
    return ann.result()


def _bgr_to_data_uri_jpeg(bgr: np.ndarray, *, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _placeholder_bgr(msg: str, w: int = 360, h: int = 203) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (45, 45, 48)
    cv2.putText(
        img,
        msg[:42],
        (10, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return img


def _build_strips_html(
    capped: list[dict[str, Any]],
    *,
    jpeg_quality: int = 82,
    max_img_css_px: int = 420,
) -> str:
    """Plain HTML + data-URI images (reliable in Colab; Plotly Image subplots often fail there)."""
    if not capped:
        return (
            '<p style="color:#fa0;">No tactic combos found (try more val frames or lower '
            "kpt confidence in analyze_recording_keypoints).</p>"
        )
    chunks: list[str] = [
        '<div id="strips" style="max-width:100%;">',
        "<h3 style=\"margin:16px 0 8px;\">Tactic sequences (left → right in time)</h3>",
        "<p style=\"font-size:13px;color:#888;margin-top:0;\">"
        "Embedded JPEGs (no Plotly) so frames show in Colab and offline viewers.</p>",
    ]
    for s in capped:
        title = html.escape(
            f"{s['tactic']} | {str(s['fight_id'])[:48]} | "
            f"f{s['frame_start']}-{s['frame_end']} | n={s['n_events']}"
        )
        chunks.append(
            '<div class="strip-row" style="margin-bottom:22px;padding-bottom:14px;'
            'border-bottom:1px solid #333;">'
        )
        chunks.append(f'<div style="font-size:13px;margin-bottom:8px;color:#9cf;">{title}</div>')
        chunks.append(
            '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end;">'
        )
        for bi, im in enumerate(s["images_bgr"]):
            uri = _bgr_to_data_uri_jpeg(im, quality=jpeg_quality)
            if not uri:
                continue
            mw = min(int(im.shape[1]), max_img_css_px)
            chunks.append(
                f'<figure style="margin:0;">'
                f'<img src="{uri}" alt="frame {bi}" '
                f'style="display:block;height:auto;width:auto;max-width:{mw}px;'
                f'max-height:360px;border:1px solid #444;border-radius:4px;"/>'
                f'<figcaption style="font-size:11px;color:#666;margin-top:4px;">t + {bi}</figcaption>'
                f"</figure>"
            )
        chunks.append("</div></div>")
    chunks.append("</div>")
    return "\n".join(chunks)


def build_interactive_tactic_report(
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
    max_examples_per_tactic: int = 4,
    sequence_len: int = 5,
    kpt_conf_draw: float = 0.35,
    fps_assumed: float = 30.0,
    combo_gap_frames: int = 12,
    thumb_max_w: int = 340,
    strip_jpeg_quality: int = 82,
) -> Path:
    """Run val predictions, heuristic strikes, write single self-contained HTML (Plotly CDN)."""
    by_fight, _all_imgs = collect_fight_frames(Path(data_yaml), split)
    fight_ids, _group_mode = fight_ids_for_predict(by_fight)
    fight_ids = fight_ids[:max_fights]

    # Collect per-fight: paths, kpts_seq, events, combos, time_base
    strips: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    time_cursor = 0

    n_pred = 0
    n_imread_fail = 0
    for fid in fight_ids:
        pairs = by_fight[fid]
        if max_frames_per_fight is not None:
            pairs = pairs[:max_frames_per_fight]
        kpts_seq: list[np.ndarray | None] = []
        paths: list[Path] = []

        for _fi, path in pairs:
            if n_pred >= max_predict_total:
                break
            sk, _h, _w = predict_largest_kpts(model, path, device, imgsz)
            n_pred += 1
            kpts_seq.append(sk)
            paths.append(path)

        if not kpts_seq:
            continue

        events, _summary = analyze_recording_keypoints(
            kpts_seq,
            fps=fps_assumed,
            kpt_conf=kpt_conf_draw,
            strikes_only=True,
        )
        combos = cluster_strike_combos(events, gap_frames=combo_gap_frames)

        for e in events:
            if e.get("kind") not in ("punch_candidate", "kick_candidate"):
                continue
            timeline_rows.append(
                {
                    "x": time_cursor + int(e["frame_index"]),
                    "y": e["kind"],
                    "side": e.get("side", ""),
                    "fight": fid[:48],
                    "frame_local": int(e["frame_index"]),
                }
            )

        T = len(kpts_seq)
        for c in combos:
            fs, fe = int(c["frame_start"]), int(c["frame_end"])
            fs = max(0, min(fs, T - 1))
            fe = max(0, min(fe, T - 1))
            if fe < fs:
                fs, fe = fe, fs
            span = fe - fs + 1
            if span >= sequence_len:
                idxs = np.linspace(fs, fe, num=sequence_len, dtype=int)
            else:
                idxs = np.linspace(fs, fe, num=max(1, span), dtype=int)
                while len(idxs) < sequence_len:
                    idxs = np.append(idxs, idxs[-1])
                idxs = idxs[:sequence_len]
            row_images: list[np.ndarray] = []
            for ii in idxs:
                p = paths[int(ii)]
                bgr = cv2.imread(str(p))
                if bgr is None:
                    n_imread_fail += 1
                    bgr = _placeholder_bgr(f"cv2.imread failed: {p.name}")
                bgr = _draw_skeleton(bgr, kpts_seq[int(ii)], kpt_conf_draw)
                cap = f"{c['tactic_proxy']} f{int(ii)}"
                cv2.putText(
                    bgr,
                    cap[:40],
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                row_images.append(_resize_bgr(bgr, thumb_max_w))
            if row_images:
                strips.append(
                    {
                        "tactic": c["tactic_proxy"],
                        "fight_id": fid,
                        "frame_start": fs,
                        "frame_end": fe,
                        "n_events": c["n_events"],
                        "images_bgr": row_images,
                    }
                )

        time_cursor += T

    # Cap strips per tactic
    by_tactic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in strips:
        by_tactic[s["tactic"]].append(s)
    capped: list[dict[str, Any]] = []
    for tac in TACTIC_ORDER:
        for s in by_tactic.get(tac, [])[:max_examples_per_tactic]:
            capped.append(s)
    for tac, lst in by_tactic.items():
        if tac not in TACTIC_ORDER:
            for s in lst[:max_examples_per_tactic]:
                capped.append(s)

    # --- Figure 1: timeline ---
    fig_timeline = go.Figure()
    kinds = sorted({r["y"] for r in timeline_rows})
    for k in kinds:
        pts = [r for r in timeline_rows if r["y"] == k]
        fig_timeline.add_trace(
            go.Scatter(
                x=[p["x"] for p in pts],
                y=[k] * len(pts),
                mode="markers",
                name=k,
                marker=dict(size=10, color=TACTIC_COLORS.get(k, "#333")),
                text=[
                    f"fight={html.escape(p['fight'])}<br>local_f={p['frame_local']}<br>side={p['side']}"
                    for p in pts
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig_timeline.update_layout(
        title="Heuristic strike candidates over val (concatenated fights; x = global frame index)",
        xaxis_title="frame index (concatenated)",
        yaxis_title="event kind",
        height=360,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    strips_html = _build_strips_html(capped, jpeg_quality=strip_jpeg_quality)

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    tl_json = pio.to_json(fig_timeline)
    warn_reads = ""
    if n_imread_fail:
        warn_reads = (
            f'<p style="color:#fa0;">Warning: <code>cv2.imread</code> failed {n_imread_fail} time(s). '
            "Check image paths (Drive mount, symlinks). Placeholder tiles show where reads failed.</p>"
        )
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Tactic preview (heuristic)</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body style="background:#111;color:#ddd;font-family:system-ui,sans-serif;padding:16px;">
  <p style="max-width:900px;">
    <b>Disclaimer:</b> <code>tactic_proxy</code> and strike dots are <b>pose heuristics</b> (extension + joint angles), not labeled MMA tactics.
    Each row is one detected combo window; thumbnails are in time order left→right (embedded images work in Colab).
  </p>
  {warn_reads}
  <div id="tl"></div>
  {strips_html}
  <script>
    var tl = {tl_json};
    Plotly.newPlot('tl', tl.data, tl.layout, {{responsive:true}});
  </script>
</body>
</html>
"""

    out_html.write_text(html_doc, encoding="utf-8")
    print("Wrote", out_html)
    if n_imread_fail:
        print(
            f"Note: cv2.imread failed {n_imread_fail} time(s); "
            "gray tiles in HTML mark missing files (check Drive paths)."
        )
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("tactics_interactive.html"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    dev = args.device or None
    build_interactive_tactic_report(
        model,
        args.data.resolve(),
        args.out.resolve(),
        split=args.split,
        imgsz=args.imgsz,
        device=dev,
    )


if __name__ == "__main__":
    main()
