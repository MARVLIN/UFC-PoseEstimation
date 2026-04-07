#!/usr/bin/env python3
"""
Run validation insights: official ``model.val()`` metrics, predicted pose sequences on the
val split grouped by fight (``_mp4-<frame>`` stems), heuristic punch/kick candidates and
simple tactic-proxy clusters (same spirit as Colab §7b / ``webcam_pose``), and an optional
Plotly HTML gallery of skeletons.

**Honesty boundaries**
  • **mAP / box metrics** — what the checkpoint achieves on the labeled val set (Ultralytics).
  • **Punch/kick/tactic_proxy** — geometry + extension heuristics on **predicted** keypoints;
    not official strikes, not ground-truth tactic labels.

Usage:
  python val_pose_insights.py --weights runs/pose/train/weights/best.pt --data mma_pose_dataset/data.yaml
  python val_pose_insights.py --weights best.pt --data data.yaml --out-dir insights_out --skip-val
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO

from dev_device import default_mps_device
from webcam_pose import _primary_detection_index, analyze_recording_keypoints, raw_kpts_17x3

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# COCO-17 edges (same as typical Ultralytics pose viz)
COCO_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (5, 11),
    (6, 8),
    (6, 12),
    (7, 9),
    (8, 10),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]


_MP4_FRAME_RE = re.compile(r"_mp4-(\d+)")


def parse_fight_and_frame(stem: str) -> tuple[str, int]:
    """fight_id = text before ``_mp4-``; frame index = first digit run after ``_mp4-`` (Roboflow-safe)."""
    base = stem
    if "_mp4-" not in base:
        return base, -1
    m = _MP4_FRAME_RE.search(base)
    if not m:
        left, _, _ = base.partition("_mp4-")
        return left, -1
    return base[: m.start()], int(m.group(1))


def _dataset_root(data_yaml: Path, cfg: dict) -> Path:
    root = data_yaml.parent
    p = cfg.get("path") or ""
    if not p:
        return root.resolve()
    base = Path(p)
    if not base.is_absolute():
        base = root / base
    return base.resolve()


def _resolve_split_path(data_yaml: Path, split_key: str) -> Path:
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = _dataset_root(data_yaml, cfg)
    rel = cfg.get(split_key)
    if rel is None:
        raise SystemExit(f"data.yaml missing key {split_key!r}")
    p = Path(rel)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def collect_split_images(split_path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[Path] = []
    if split_path.is_file():
        if split_path.suffix.lower() == ".txt":
            parent = split_path.parent
            for line in split_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                im = Path(line)
                if not im.is_absolute():
                    im = parent / im
                if im.suffix.lower() in exts and im.is_file():
                    out.append(im.resolve())
        return sorted(set(out))
    if not split_path.is_dir():
        raise SystemExit(f"Val path is not a file or directory: {split_path}")
    for ext in exts:
        out.extend(split_path.rglob(f"*{ext}"))
    return sorted({p.resolve() for p in out if p.is_file()})


def collect_fight_frames(
    data_yaml: Path, split_key: str
) -> tuple[dict[str, list[tuple[int, Path]]], list[Path]]:
    split_path = _resolve_split_path(data_yaml, split_key)
    images = collect_split_images(split_path)
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
    if orphans:
        by_fight["__orphan_lex__"] = [(-1, x) for x in sorted(orphans, key=lambda x: x.name)]
    return dict(by_fight), images


def fight_ids_for_predict(by_fight: dict[str, list[tuple[int, Path]]]) -> tuple[list[str], str]:
    """Return sorted fight keys to run prediction on, and how images were grouped.

    If no stems match ``_mp4-<frame>``, all val images live under ``__orphan_lex__``; we then
    treat the whole split as one synthetic sequence ``__flat_val__`` so predict + heuristics still run.
    """
    fight_ids = sorted(k for k in by_fight if k not in ("__orphan_lex__", "__flat_val__"))
    if fight_ids:
        return fight_ids, "fight_stem_mp4"
    orphans = by_fight.get("__orphan_lex__") or []
    paths = [p for _fi, p in orphans]
    if not paths:
        return [], "empty_split"
    by_fight["__flat_val__"] = [(i, p) for i, p in enumerate(sorted(paths, key=lambda x: x.as_posix()))]
    return ["__flat_val__"], "flat_all_images"


def predict_largest_kpts(model: YOLO, path: Path, device: str | None, imgsz: int) -> tuple[np.ndarray | None, int, int]:
    """One image → 17×3 keypoints for largest box, plus original H×W."""
    results = model.predict(
        source=str(path),
        imgsz=imgsz,
        device=device or "",
        verbose=False,
    )
    if not results:
        return None, 0, 0
    r = results[0]
    h, w = (int(r.orig_shape[0]), int(r.orig_shape[1])) if r.orig_shape is not None else (0, 0)
    if r.boxes is None or len(r.boxes) == 0:
        return None, h, w
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    det_i = _primary_detection_index(boxes, confs, (h, w), "largest")
    sk = raw_kpts_17x3(r, det_i)
    return sk, h, w


def cluster_strike_combos(events: list[dict], gap_frames: int = 12) -> list[dict]:
    """Merge nearby punch/kick candidates into tactic_proxy clusters (heuristic)."""
    strike_kinds = frozenset({"punch_candidate", "kick_candidate"})
    evs = sorted((e for e in events if e.get("kind") in strike_kinds), key=lambda e: e["frame_index"])
    if not evs:
        return []
    clusters: list[list[dict]] = []
    cur: list[dict] = []
    for e in evs:
        if not cur:
            cur = [e]
        elif e["frame_index"] - cur[-1]["frame_index"] <= gap_frames:
            cur.append(e)
        else:
            clusters.append(cur)
            cur = [e]
    if cur:
        clusters.append(cur)
    out: list[dict] = []
    for c in clusters:
        kinds = [x["kind"] for x in c]
        n_p = sum(1 for k in kinds if k == "punch_candidate")
        n_k = sum(1 for k in kinds if k == "kick_candidate")
        if n_p and n_k:
            tag = "punch_kick_chain_proxy"
        elif n_p >= 2:
            tag = "multi_punch_proxy"
        elif n_k:
            tag = "kick_proxy"
        elif n_p:
            tag = "punch_proxy"
        else:
            tag = "strike_proxy"
        out.append(
            {
                "frame_start": int(c[0]["frame_index"]),
                "frame_end": int(c[-1]["frame_index"]),
                "tactic_proxy": tag,
                "n_events": len(c),
            }
        )
    return out


def _metrics_to_jsonable(model: YOLO, data_yaml: Path, device: str | None, imgsz: int) -> dict[str, Any]:
    """Run official val; return a small JSON-safe dict."""
    kw: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": imgsz,
        "plots": False,
        "verbose": False,
    }
    if device is not None and device != "":
        kw["device"] = device
    m = model.val(**kw)
    payload: dict[str, Any] = {"ok": True}
    if hasattr(m, "box") and m.box is not None:
        b = m.box
        for name in ("map50", "map75", "map"):
            if hasattr(b, name):
                try:
                    payload[f"box_{name}"] = float(getattr(b, name))
                except (TypeError, ValueError):
                    pass
    if hasattr(m, "pose") and m.pose is not None:
        p = m.pose
        for name in ("map50", "map75", "map"):
            if hasattr(p, name):
                try:
                    payload[f"pose_{name}"] = float(getattr(p, name))
                except (TypeError, ValueError):
                    pass
    if hasattr(m, "speed") and m.speed:
        payload["speed_ms"] = {k: float(v) for k, v in m.speed.items() if isinstance(v, (int, float))}
    return payload


def _kpts_traces_3d(
    kpts: np.ndarray,
    w: int,
    h: int,
    conf_draw: float,
    line_color: str = "#00d4ff",
    point_color: str = "#ff4d6d",
) -> list[Any]:
    """2D keypoints in a 3D viewer: X/Y normalized, Z=0 (monocular; no metric depth)."""
    traces: list[Any] = []
    xs, ys, zs, text = [], [], [], []
    for i in range(17):
        if kpts.shape[0] <= i or kpts[i, 2] < conf_draw:
            continue
        x = float(kpts[i, 0]) / max(w, 1) - 0.5
        y = 0.5 - float(kpts[i, 1]) / max(h, 1)
        xs.append(x)
        ys.append(y)
        zs.append(0.0)
        text.append(str(i))
    traces.append(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            text=text,
            textposition="top center",
            textfont=dict(size=8, color="#e0e0e0"),
            marker=dict(size=4, color=point_color),
            name="kpts",
            showlegend=False,
        )
    )
    for a, b in COCO_EDGES:
        if (
            kpts.shape[0] <= max(a, b)
            or kpts[a, 2] < conf_draw
            or kpts[b, 2] < conf_draw
        ):
            continue
        xa = float(kpts[a, 0]) / max(w, 1) - 0.5
        ya = 0.5 - float(kpts[a, 1]) / max(h, 1)
        xb = float(kpts[b, 0]) / max(w, 1) - 0.5
        yb = 0.5 - float(kpts[b, 1]) / max(h, 1)
        traces.append(
            go.Scatter3d(
                x=[xa, xb],
                y=[ya, yb],
                z=[0.0, 0.0],
                mode="lines",
                line=dict(color=line_color, width=6),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return traces


def build_plotly_gallery(highlights: list[dict], out_html: Path) -> None:
    if not _HAS_PLOTLY:
        raise RuntimeError("plotly is not installed; pip install plotly")
    n = min(len(highlights), 6)
    if n == 0:
        if out_html.is_file():
            out_html.unlink()
        return
    rows, cols = 2, 3
    n_cells = rows * cols
    titles = [h["title"] for h in highlights[:n]] + [""] * max(0, n_cells - n)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scatter3d"}] * cols for _ in range(rows)],
        subplot_titles=titles[:n_cells],
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
    )
    for i, hl in enumerate(highlights[:n]):
        r = i // cols + 1
        c = i % cols + 1
        for tr in _kpts_traces_3d(hl["kpts"], hl["w"], hl["h"], hl.get("conf_draw", 0.25)):
            fig.add_trace(tr, row=r, col=c)
    fig.update_layout(
        title_text="Val split — predicted pose (2D keypoints in 3D viewer; Z=0, no depth)",
        height=780,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="#111",
        font=dict(color="#ddd"),
    )
    fig.update_scenes(
        xaxis=dict(range=[-0.55, 0.55], showbackground=False, gridcolor="#444"),
        yaxis=dict(range=[-0.55, 0.55], showbackground=False, gridcolor="#444"),
        zaxis=dict(range=[-0.05, 0.05], showbackground=False, showticklabels=False),
        aspectmode="cube",
        bgcolor="#1a1a1a",
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Val metrics + pose strike heuristics + 3D skeleton HTML")
    parser.add_argument("--weights", type=Path, required=True, help="Trained pose weights (.pt)")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "mma_pose_dataset" / "data.yaml",
        help="Dataset data.yaml",
    )
    parser.add_argument("--split", type=str, default="val", help="YAML key for split (usually val)")
    parser.add_argument("--out-dir", type=Path, default=Path("insights_out"), help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | mps | cuda:0 | …")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fps-assumed", type=float, default=30.0, help="FPS for time_sec in heuristics")
    parser.add_argument("--kpt-conf", type=float, default=0.35, help="Keypoint conf for heuristics / drawing")
    parser.add_argument("--max-fights", type=int, default=None, help="Cap number of fight groups (after sort)")
    parser.add_argument("--max-frames-per-fight", type=int, default=None, help="Cap frames per fight")
    parser.add_argument("--max-predict-images", type=int, default=500, help="Stop after this many predicted images")
    parser.add_argument("--skip-val", action="store_true", help="Skip slow model.val() metrics")
    parser.add_argument("--no-html", action="store_true", help="Skip Plotly HTML gallery")
    parser.add_argument(
        "--no-neutral-gallery",
        action="store_true",
        help="If no punch/kick peaks, do not fill the HTML gallery with confident neutral poses",
    )
    args = parser.parse_args()

    if args.cpu:
        args.device = "cpu"
    dev = default_mps_device() if args.device == "auto" else args.device

    if not args.data.is_file():
        raise SystemExit(f"Missing data yaml: {args.data}")
    if not args.weights.is_file():
        raise SystemExit(f"Missing weights: {args.weights}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))

    val_metrics: dict[str, Any] = {"skipped": True}
    if not args.skip_val:
        try:
            val_metrics = _metrics_to_jsonable(model, args.data.resolve(), dev, args.imgsz)
            val_metrics["skipped"] = False
        except Exception as e:  # noqa: BLE001 — user-facing aggregate
            val_metrics = {"skipped": False, "ok": False, "error": str(e)}

    by_fight, all_images = collect_fight_frames(args.data.resolve(), args.split)
    fight_ids, grouping_mode = fight_ids_for_predict(by_fight)
    n_orphan_lex = len(by_fight.get("__orphan_lex__") or [])
    if args.max_fights is not None:
        fight_ids = fight_ids[: args.max_fights]

    per_fight: dict[str, Any] = {}
    global_events: list[dict] = []
    highlights: list[dict] = []
    pose_preview_pool: list[dict] = []
    n_predicted = 0

    stop_predict = False
    for fid in fight_ids:
        if stop_predict:
            break
        pairs = by_fight[fid]
        if args.max_frames_per_fight is not None:
            pairs = pairs[: args.max_frames_per_fight]
        kpts_seq: list[np.ndarray | None] = []
        meta: list[tuple[Path, int, int]] = []
        for _fidx, path in pairs:
            if args.max_predict_images is not None and n_predicted >= args.max_predict_images:
                stop_predict = True
                break
            sk, h, w = predict_largest_kpts(model, path, dev, args.imgsz)
            n_predicted += 1
            kpts_seq.append(sk)
            meta.append((path, h, w))
            if sk is not None and h > 0 and w > 0:
                mc = float(np.nanmean(sk[:, 2]))
                if mc >= 0.25 and len(pose_preview_pool) < 600:
                    pose_preview_pool.append(
                        {
                            "title": f'{fid[:32]}{"…" if len(fid) > 32 else ""}  neutral  conf~{mc:.2f}  {path.name}',
                            "kpts": sk,
                            "w": w,
                            "h": h,
                            "conf_draw": args.kpt_conf,
                            "fight_id": fid,
                            "frame_index": len(kpts_seq) - 1,
                            "path": str(path),
                        }
                    )

        events, summary = analyze_recording_keypoints(
            kpts_seq,
            fps=args.fps_assumed,
            kpt_conf=args.kpt_conf,
            strikes_only=True,
        )
        combos = cluster_strike_combos(events)
        for e in events:
            e2 = dict(e)
            e2["fight_id"] = fid
            global_events.append(e2)
        per_fight[fid] = {
            "summary": summary,
            "n_combos": len(combos),
            "tactic_combos_proxy": combos,
        }

        for e in events:
            if e.get("kind") not in ("punch_candidate", "kick_candidate"):
                continue
            fi = int(e["frame_index"])
            if fi < 0 or fi >= len(meta):
                continue
            path, h, w = meta[fi]
            sk = kpts_seq[fi]
            if sk is None:
                continue
            highlights.append(
                {
                    "title": f'{fid[:40]}{"…" if len(fid) > 40 else ""}  f{fi}  {e["kind"]} {e.get("side", "")}',
                    "kpts": sk,
                    "w": w,
                    "h": h,
                    "conf_draw": args.kpt_conf,
                    "fight_id": fid,
                    "frame_index": fi,
                    "path": str(path),
                }
            )

    highlights.sort(
        key=lambda x: (
            -float(np.nanmean(x["kpts"][:, 2])) if x.get("kpts") is not None else 0.0,
            x["fight_id"],
        )
    )

    gallery_used_neutral = False
    if not highlights and pose_preview_pool and not args.no_neutral_gallery:
        n = min(6, len(pose_preview_pool))
        if n > 0:
            idxs = np.linspace(0, len(pose_preview_pool) - 1, num=n, dtype=int)
            highlights = [dict(pose_preview_pool[int(i)]) for i in idxs]
            gallery_used_neutral = True

    summary_blob: dict[str, Any] = {
        "weights": str(args.weights.resolve()),
        "data_yaml": str(args.data.resolve()),
        "split": args.split,
        "n_val_images": len(all_images),
        "grouping_mode": grouping_mode,
        "n_orphan_lex_entries": n_orphan_lex,
        "n_fight_groups": len(fight_ids),
        "n_images_predicted": n_predicted,
        "gallery_used_neutral_fallback": gallery_used_neutral,
        "val_metrics": val_metrics,
        "disclaimer": (
            "punch_candidate/kick_candidate/tactic_proxy are pose heuristics on model predictions, "
            "not labeled strikes or official tactics."
        ),
        "per_fight": {k: {"summary": v["summary"], "n_combos": v["n_combos"], "tactic_combos_proxy": v["tactic_combos_proxy"]} for k, v in per_fight.items()},
        "strike_events": global_events,
        "highlights_for_html": [
            {"fight_id": h["fight_id"], "frame_index": h["frame_index"], "path": h["path"], "title": h["title"]}
            for h in highlights[:12]
        ],
    }

    json_path = args.out_dir / "val_insights_summary.json"
    html_path = args.out_dir / "val_3d_gallery.html"
    html_skip_reason: str | None = None
    if not args.no_html and _HAS_PLOTLY:
        try:
            build_plotly_gallery(highlights, html_path)
        except Exception as e:  # noqa: BLE001
            summary_blob["html_error"] = str(e)
            html_skip_reason = f"plotly_error:{e}"
    elif not args.no_html and not _HAS_PLOTLY:
        summary_blob["html_note"] = "install plotly to generate val_3d_gallery.html"
        html_skip_reason = "missing_plotly_package"

    if (
        not args.no_html
        and _HAS_PLOTLY
        and not summary_blob.get("html_error")
        and not html_path.is_file()
    ):
        html_skip_reason = html_skip_reason or "no_frames_for_gallery"

    if not args.no_html and html_skip_reason and not html_path.is_file():
        summary_blob["html_skip_reason"] = html_skip_reason

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_blob, f, indent=2)

    print(f"Wrote {json_path}")
    print(
        f"Grouping: {grouping_mode} | fights/sequences: {len(fight_ids)} | predicted images: {n_predicted}"
    )
    if grouping_mode == "flat_all_images" and n_orphan_lex:
        print(
            f"Note: no '_mp4-<frame>' stems in val — ran as one flat sequence ({n_orphan_lex} images)."
        )
    if not args.no_html and _HAS_PLOTLY and html_path.is_file():
        print(f"Wrote {html_path}" + (" (neutral pose fallback; no strike peaks)" if gallery_used_neutral else ""))
    elif not args.no_html:
        if not _HAS_PLOTLY:
            print("Plotly HTML skipped: install plotly (`pip install plotly`).")
        elif html_skip_reason == "no_frames_for_gallery":
            print("Plotly HTML skipped: no valid poses in sampled frames.")
        else:
            print(f"Plotly HTML skipped: {html_skip_reason}")


if __name__ == "__main__":
    main()
