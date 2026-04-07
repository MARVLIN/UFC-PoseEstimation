#!/usr/bin/env python3
"""
Webcam pose overlay (best.pt) with optional recording and education panel.

Why it can feel “late”
  • Inference takes time. On M1/M2/M3, device defaults to mps (Metal). Smaller --imgsz helps.
  • Temporal smoothing (EMA) trades jitter for lag — default ema is 1.0 (no blend); use --ema-new 0.35 or --smooth-display to smooth.
  • Webcams buffer frames: CAP_PROP_BUFFERSIZE=1 plus a background grab thread (default) so you process the newest frame, not a stale one.

Can delayed overlay teach tactics/posture?
  • Live motor learning needs the skeleton close to real time; heavy smoothing works
    against that. Use --direct (or high --ema-new) for “am I doing it now?” feedback.
  • The rotating fundamentals are **general concepts**, not tied to the delayed skeleton;
    treat them like flashcards while you drill.
  • **Recorded sessions** (--record) are better for review: pause, compare posture to cues,
    and repeat movements without relying on instant overlay accuracy.

Usage (Apple Silicon dev defaults to MPS):
  python webcam_pose.py --mirror                       # threaded capture + center person + no EMA by default
  python webcam_pose.py --mirror --imgsz 640           # more accurate keypoints, a bit slower
  python webcam_pose.py --mirror --sync-capture        # if threaded grab misbehaves on your OS
  python webcam_pose.py --mirror --select largest      # track biggest box (e.g. full-body shot)
  python webcam_pose.py --mirror --half                # fp16 on MPS when stable
  python webcam_pose.py --cpu                          # force CPU for debugging
  python webcam_pose.py --mirror --record out/session.mp4   # after stop: punch/kick counts + snapshot PNGs

After --record, the script writes a JSON summary and saves PNG stills only at heuristic **strike**
moments (punch_candidate / kick_candidate). Softer motion peaks are omitted unless you pass
`--include-soft-motion-peaks` (JSON only; snapshots stay strike-only).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from dev_device import default_mps_device

# COCO keypoints (Ultralytics)
LS, RS = 5, 6
LE, RE = 7, 8
LW, RW = 9, 10
LH, RH = 11, 12
LK, RK = 13, 14
LA, RA = 15, 16
NOSE = 0

# General education (rotating); not personalized coaching.
FUNDAMENTALS: list[tuple[str, str]] = [
    (
        "Stance & balance",
        "Weight mostly on the balls of the feet; knees soft so you can move in any direction.",
    ),
    (
        "Guard",
        "Elbows stay close to the ribs; hands protect the center line, not the hips.",
    ),
    (
        "Head position",
        "Chin slightly down; eyes up — you see threats through your brow, not by lifting the chin.",
    ),
    (
        "Footwork",
        "Small steps beat long lunges; reset your stance after every angle change.",
    ),
    (
        "Breathing",
        "Steady nose breathing when possible; sharp exhale on sharp effort to keep tension from freezing you.",
    ),
    (
        "Distance",
        "Control range with feints and rhythm — striking starts from timing, not only speed.",
    ),
    (
        "Defense layers",
        "Movement first, then blocks or parries; the last line is your guard, not your face.",
    ),
    (
        "Recovery",
        "After combos, return hands to guard before you admire the work — habits decide real sparring.",
    ),
]


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = aa + ba - inter
    return inter / u if u > 0 else 0.0


def _primary_detection_index(
    boxes: np.ndarray,
    confs: np.ndarray,
    frame_hw: tuple[int, int],
    policy: str,
) -> int:
    """Pick one person when there is no temporal prior (first frame or after long loss)."""
    h, w = frame_hw
    cx0, cy0 = 0.5 * w, 0.5 * h
    n = len(boxes)
    if n == 1:
        return 0
    if policy == "conf":
        return int(np.argmax(confs))
    if policy == "largest":
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        return int(np.argmax(areas))
    # center — best default for selfie webcam (subject near middle)
    xc = 0.5 * (boxes[:, 0] + boxes[:, 2])
    yc = 0.5 * (boxes[:, 1] + boxes[:, 3])
    d2 = (xc - cx0) ** 2 + (yc - cy0) ** 2
    return int(np.argmin(d2))


def pick_detection_index(
    result,
    prev_xyxy: np.ndarray | None,
    frame_hw: tuple[int, int],
    policy: str,
) -> int | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    if prev_xyxy is None:
        return _primary_detection_index(boxes, confs, frame_hw, policy)
    ious = [iou_xyxy(prev_xyxy, boxes[i]) for i in range(len(boxes))]
    j = int(np.argmax(ious))
    if ious[j] >= 0.08:
        return j
    return _primary_detection_index(boxes, confs, frame_hw, policy)


def raw_kpts_17x3(result, det_i: int) -> np.ndarray | None:
    k = result.keypoints
    if k is None or len(k) == 0 or det_i >= len(k):
        return None
    xy = k.xy[det_i].cpu().numpy()
    if k.conf is None:
        conf = np.ones(17, dtype=np.float32)
    else:
        conf = k.conf[det_i].cpu().numpy().astype(np.float32)
    return np.concatenate([xy, conf[:, np.newaxis]], axis=1).astype(np.float32)


class SmoothPoseTracker:
    """EMA on (x, y, conf) to stabilize limbs and reduce flicker."""

    def __init__(
        self,
        ema_new: float,
        ema_conf: float,
        hold_frames: int,
        select_policy: str,
    ):
        self.ema_new = ema_new
        self.ema_conf = ema_conf
        self.hold_frames = hold_frames
        self.select_policy = select_policy
        self.state: np.ndarray | None = None
        self.prev_box: np.ndarray | None = None
        self.miss = 0

    def update(self, result, frame_hw: tuple[int, int]) -> np.ndarray | None:
        det_i = pick_detection_index(result, self.prev_box, frame_hw, self.select_policy)
        if det_i is None:
            self.miss += 1
            self.prev_box = None
            if self.state is not None and self.miss <= self.hold_frames:
                return self.state
            if self.miss > self.hold_frames:
                self.state = None
            return None

        raw = raw_kpts_17x3(result, det_i)
        if raw is None:
            self.miss += 1
            return self.state if self.state is not None and self.miss <= self.hold_frames else None

        self.miss = 0
        self.prev_box = result.boxes.xyxy[det_i].cpu().numpy()

        if self.state is None:
            self.state = raw.copy()
        else:
            a, b = self.ema_new, 1.0 - self.ema_new
            self.state[:, :2] = a * raw[:, :2] + b * self.state[:, :2]
            ac, bc = self.ema_conf, 1.0 - self.ema_conf
            self.state[:, 2] = ac * raw[:, 2] + bc * self.state[:, 2]
        return self.state


class LatestFrameThread:
    """Background read loop: main thread always gets the newest frame (cuts USB buffer lag)."""

    def __init__(self, cap: cv2.VideoCapture, mirror: bool):
        self.cap = cap
        self.mirror = mirror
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="webcam-grab", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.002)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


def drain_capture_buffer(cap: cv2.VideoCapture, max_grabs: int) -> None:
    """Drop queued frames on the driver (best-effort; not all backends honor grab())."""
    for _ in range(max(0, max_grabs)):
        if not cap.grab():
            break


def posture_note_from_kpts(kpts: np.ndarray, w: int, h: int, vis_thr: float) -> str | None:
    """Single short line from geometry; optional when keypoints confident enough."""
    xyn = kpts.copy()
    xyn[:, 0] /= w
    xyn[:, 1] /= h

    def ok(i: int) -> bool:
        return kpts[i, 2] >= vis_thr

    if ok(LW) and ok(LS) and xyn[LW, 1] > xyn[LS, 1] + 0.1:
        return "Posture cue: bring hands closer to shoulder height for a tighter guard."
    if ok(RW) and ok(RS) and xyn[RW, 1] > xyn[RS, 1] + 0.1:
        return "Posture cue: bring hands closer to shoulder height for a tighter guard."
    if ok(LA) and ok(RA) and ok(LS) and ok(RS):
        base = abs(xyn[LA, 0] - xyn[RA, 0]) / (abs(xyn[LS, 0] - xyn[RS, 0]) + 1e-6)
        if base < 0.85:
            return "Base cue: a slightly wider stance helps lateral balance when you move."
    if ok(LW) and ok(RW) and ok(LH) and ok(RH):
        hip_y = 0.5 * (xyn[LH, 1] + xyn[RH, 1])
        if xyn[LW, 1] < hip_y - 0.04 and xyn[RW, 1] < hip_y - 0.04:
            return "Guard height: hands are above the hip line — a solid default in stand-up."
    return None


def wrap_lines(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    n = 0
    for w in words:
        if n + len(w) + (1 if cur else 0) > max_chars:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
            n = len(w)
        else:
            cur.append(w)
            n += len(w) + (1 if len(cur) > 1 else 0)
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def draw_insights_panel(
    frame: np.ndarray,
    title: str,
    body: str,
    extra: str | None,
    panel_h: int = 150,
    alpha: float = 0.55,
) -> None:
    h, w = frame.shape[:2]
    ph = min(panel_h, h // 3)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - ph), (w, h), (24, 24, 28), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    y = h - ph + 22
    cv2.putText(
        frame,
        title,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 230, 240),
        2,
        cv2.LINE_AA,
    )
    y += 26
    for line in wrap_lines(body, max(40, w // 11)):
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 205, 210),
            1,
            cv2.LINE_AA,
        )
        y += 20
    if extra:
        y += 6
        cv2.putText(
            frame,
            extra,
            (16, min(y, h - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (140, 200, 255),
            1,
            cv2.LINE_AA,
        )


def _interior_angle_deg_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Angle ABC at vertex B (degrees)."""
    ba = a.astype(np.float64) - b.astype(np.float64)
    bc = c.astype(np.float64) - b.astype(np.float64)
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosv = float(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def _mov_mean_nan(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if win <= 1:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        seg = xp[i : i + win]
        m = np.isfinite(seg)
        if np.any(m):
            out[i] = float(np.mean(seg[m]))
    return out


def _arm_ext_dist(K: np.ndarray, t: int, sh: int, el: int, wr: int, conf_min: float) -> float | None:
    if float(K[t, sh, 2]) < conf_min or float(K[t, el, 2]) < conf_min or float(K[t, wr, 2]) < conf_min:
        return None
    p0 = K[t, sh, :2].astype(np.float64)
    p2 = K[t, wr, :2].astype(np.float64)
    return float(np.linalg.norm(p2 - p0))


def _leg_ext_dist(K: np.ndarray, t: int, hip: int, knee: int, ank: int, conf_min: float) -> float | None:
    if float(K[t, hip, 2]) < conf_min or float(K[t, knee, 2]) < conf_min or float(K[t, ank, 2]) < conf_min:
        return None
    p0 = K[t, hip, :2].astype(np.float64)
    p2 = K[t, ank, :2].astype(np.float64)
    return float(np.linalg.norm(p2 - p0))


def _series_arm_ext(K: np.ndarray, sh: int, el: int, wr: int, conf_min: float) -> np.ndarray:
    T = K.shape[0]
    d = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        v = _arm_ext_dist(K, t, sh, el, wr, conf_min)
        if v is not None:
            d[t] = v
    return d


def _series_leg_ext(K: np.ndarray, hip: int, knee: int, ank: int, conf_min: float) -> np.ndarray:
    T = K.shape[0]
    d = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        v = _leg_ext_dist(K, t, hip, knee, ank, conf_min)
        if v is not None:
            d[t] = v
    return d


def _find_peaks_simple(y: np.ndarray, min_height: float, min_dist: int) -> list[int]:
    y = np.asarray(y, dtype=np.float64)
    cand: list[int] = []
    for i in range(1, len(y) - 1):
        if not np.isfinite(y[i]) or y[i] < min_height:
            continue
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            cand.append(i)
    cand.sort(key=lambda i: float(y[i]), reverse=True)
    kept: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_dist for j in kept):
            kept.append(i)
    return sorted(kept)


def analyze_recording_keypoints(
    kpts_per_frame: list[np.ndarray | None],
    fps: float,
    *,
    kpt_conf: float,
    vel_smooth: int = 5,
    peak_min_dist: int = 6,
    peak_height_pct: float = 72.0,
    punch_min_elbow_deg: float = 125.0,
    kick_min_knee_deg: float = 128.0,
    strikes_only: bool = True,
) -> tuple[list[dict], dict]:
    """
    Heuristic punch/kick *candidates* from a sequence of 17x3 keypoints (x, y, conf).
    Not official strikes — pose proxy only (same spirit as the Colab §7b notebook).

    When strikes_only=True (default), only extension peaks that pass elbow/knee straightness
    are kept — no arm_motion_peak / leg_motion_peak rows or snapshots.
    """
    T = len(kpts_per_frame)
    K = np.full((T, 17, 3), np.nan, dtype=np.float64)
    for t, sk in enumerate(kpts_per_frame):
        if sk is not None and sk.shape[0] >= 17:
            K[t] = sk.astype(np.float64)

    events: list[dict] = []
    min_dist = max(3, int(round(fps * 0.18)))
    min_dist = max(min_dist, peak_min_dist)

    def process_arm(side: str, sh: int, el: int, wr: int) -> None:
        d = _series_arm_ext(K, sh, el, wr, kpt_conf)
        ds = _mov_mean_nan(d, vel_smooth)
        pos = ds[np.isfinite(ds) & (ds > 0)]
        thr = float(np.percentile(pos, peak_height_pct)) if len(pos) > 8 else (float(np.nanmax(ds)) * 0.55 if np.any(np.isfinite(ds)) else 0.0)
        thr = max(thr, 1e-3)
        for t in _find_peaks_simple(ds, thr, min_dist):
            ang = None
            if (
                np.isfinite(K[t, sh, 2])
                and np.isfinite(K[t, el, 2])
                and np.isfinite(K[t, wr, 2])
                and min(float(K[t, sh, 2]), float(K[t, el, 2]), float(K[t, wr, 2])) >= kpt_conf
            ):
                ang = _interior_angle_deg_2d(K[t, sh, :2], K[t, el, :2], K[t, wr, :2])
            kind = "punch_candidate" if ang is not None and ang >= punch_min_elbow_deg else "arm_motion_peak"
            if strikes_only and kind != "punch_candidate":
                continue
            events.append(
                {
                    "frame_index": int(t),
                    "time_sec": float(t / max(fps, 1e-6)),
                    "limb": "arm",
                    "side": side,
                    "kind": kind,
                    "elbow_or_knee_deg": float(ang) if ang is not None else None,
                }
            )

    def process_leg(side: str, hip: int, knee: int, ank: int) -> None:
        d = _series_leg_ext(K, hip, knee, ank, kpt_conf)
        ds = _mov_mean_nan(d, vel_smooth)
        pos = ds[np.isfinite(ds) & (ds > 0)]
        thr = float(np.percentile(pos, peak_height_pct)) if len(pos) > 8 else (float(np.nanmax(ds)) * 0.55 if np.any(np.isfinite(ds)) else 0.0)
        thr = max(thr, 1e-3)
        for t in _find_peaks_simple(ds, thr, min_dist):
            ang = None
            if (
                np.isfinite(K[t, hip, 2])
                and np.isfinite(K[t, knee, 2])
                and np.isfinite(K[t, ank, 2])
                and min(float(K[t, hip, 2]), float(K[t, knee, 2]), float(K[t, ank, 2])) >= kpt_conf
            ):
                ang = _interior_angle_deg_2d(K[t, hip, :2], K[t, knee, :2], K[t, ank, :2])
            kind = "kick_candidate" if ang is not None and ang >= kick_min_knee_deg else "leg_motion_peak"
            if strikes_only and kind != "kick_candidate":
                continue
            events.append(
                {
                    "frame_index": int(t),
                    "time_sec": float(t / max(fps, 1e-6)),
                    "limb": "leg",
                    "side": side,
                    "kind": kind,
                    "elbow_or_knee_deg": float(ang) if ang is not None else None,
                }
            )

    process_arm("L", LS, LE, LW)
    process_arm("R", RS, RE, RW)
    process_leg("L", LH, LK, LA)
    process_leg("R", RH, RK, RA)

    events.sort(key=lambda e: (e["frame_index"], e["limb"], e["side"]))

    n_punch = sum(1 for e in events if e["kind"] == "punch_candidate")
    n_kick = sum(1 for e in events if e["kind"] == "kick_candidate")
    summary = {
        "frames": T,
        "fps_assumed": float(fps),
        "n_punch_candidate": n_punch,
        "n_kick_candidate": n_kick,
        "strikes_only": strikes_only,
        "disclaimer": "Pose-only heuristics (extension peaks + joint angle). Not UFC / Fight Metric strikes landed.",
    }
    if not strikes_only:
        summary["n_arm_motion_peak"] = sum(1 for e in events if e["kind"] == "arm_motion_peak")
        summary["n_leg_motion_peak"] = sum(1 for e in events if e["kind"] == "leg_motion_peak")
    return events, summary


def save_strike_snapshots(
    video_path: Path,
    kpts_per_frame: list[np.ndarray | None],
    events: list[dict],
    out_dir: Path,
    *,
    kinds_to_snap: frozenset[str] | None = None,
    redraw_skeleton: bool = False,
    kpt_draw_conf: float = 0.35,
) -> list[Path]:
    """Extract frames from the recorded video at strike-tagged moments.

    By default uses the frame as stored in the MP4 (already includes skeleton overlay from the session).
    Pass redraw_skeleton=True to draw from stored keypoints again (e.g. if you recorded without overlay).
    """
    if kinds_to_snap is None:
        kinds_to_snap = frozenset({"punch_candidate", "kick_candidate"})
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved: list[Path] = []
    try:
        for ev in events:
            if ev["kind"] not in kinds_to_snap:
                continue
            fi = int(ev["frame_index"])
            if fi < 0 or fi >= len(kpts_per_frame):
                continue
            sk = kpts_per_frame[fi]
            if sk is None:
                continue
            if n_frames > 0 and fi >= n_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            canvas = frame.copy()
            if redraw_skeleton:
                lw = max(2, int(round((h + w) / 420)))
                ann = Annotator(canvas, line_width=lw)
                ann.kpts(sk, shape=(h, w), kpt_line=True, conf_thres=kpt_draw_conf, radius=max(2, lw))
            label = f'{ev["kind"]}_{ev["side"]}_f{fi:06d}_t{ev["time_sec"]:.2f}s'
            fp = out_dir / f"{label}.png"
            tag = f'{ev["kind"].replace("_", " ")} {ev["side"]}  t={ev["time_sec"]:.2f}s  f={fi}'
            cv2.putText(
                canvas,
                tag,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(fp), canvas)
            saved.append(fp)
    finally:
        cap.release()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Webcam pose overlay + optional recording + insights")
    ap.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent / "best.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument(
        "--imgsz",
        type=int,
        default=480,
        help="Inference size (higher = tighter skeleton, slower; 416–640 typical)",
    )
    ap.add_argument("--conf", type=float, default=0.45, help="Person detection confidence (higher = less clutter)")
    ap.add_argument("--kpt-conf", type=float, default=0.35, help="Min keypoint confidence to draw joint/limb")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument(
        "--max-det",
        type=int,
        default=5,
        help="Detect up to N people, then pick one via --select (use 1 if frame is always solo)",
    )
    ap.add_argument(
        "--select",
        type=str,
        choices=("center", "largest", "conf"),
        default="center",
        help="Which person to track when multiple are visible (center = best for webcam selfie)",
    )
    ap.add_argument(
        "--sync-capture",
        action="store_true",
        help="Use cap.read() on main thread instead of a grab thread (higher latency on some cameras)",
    )
    ap.add_argument(
        "--buffer-drain",
        type=int,
        default=2,
        help="With --sync-capture: cap.grab() this many times before read (drop stale frames; 0 to disable)",
    )
    ap.add_argument(
        "--direct",
        action="store_true",
        help="No EMA smoothing: skeleton follows the model on this frame (best for live posture feedback)",
    )
    ap.add_argument(
        "--smooth-display",
        action="store_true",
        help="Heavy temporal smoothing: calmer lines, noticeably more lag",
    )
    ap.add_argument(
        "--ema-new",
        type=float,
        default=1.0,
        help="Blend weight for new frame (1.0 = no smoothing, best body sync). Lower = smoother, laggier",
    )
    ap.add_argument(
        "--ema-conf",
        type=float,
        default=1.0,
        help="Smoothing for per-keypoint confidence (1.0 = no smoothing). Ignored if --direct",
    )
    ap.add_argument("--hold", type=int, default=3, help="Frames to keep last pose after lost detection")
    ap.add_argument("--record", type=Path, default=None, help="Save MP4 to this path (e.g. session.mp4)")
    ap.add_argument(
        "--no-record-report",
        action="store_true",
        help="After recording, skip punch/kick heuristic summary and snapshot PNGs",
    )
    ap.add_argument(
        "--snapshots-dir",
        type=Path,
        default=None,
        help="Where to save strike snapshot images (default: <recording_stem>_strike_snapshots/)",
    )
    ap.add_argument(
        "--analyze-kpt-conf",
        type=float,
        default=None,
        help="Min keypoint confidence for strike heuristics (default: same as --kpt-conf)",
    )
    ap.add_argument(
        "--punch-elbow-deg",
        type=float,
        default=125.0,
        help="Arm peak counts as punch_candidate if elbow angle >= this (degrees)",
    )
    ap.add_argument(
        "--kick-knee-deg",
        type=float,
        default=128.0,
        help="Leg peak counts as kick_candidate if knee angle >= this (degrees)",
    )
    ap.add_argument(
        "--snapshot-redraw-skeleton",
        action="store_true",
        help="Redraw skeleton on snapshots from stored keypoints; also on by default with --no-insights",
    )
    ap.add_argument(
        "--include-soft-motion-peaks",
        action="store_true",
        help="Also record arm_motion_peak / leg_motion_peak in JSON (default: strikes-only punch/kick candidates)",
    )
    ap.add_argument("--no-insights", action="store_true", help="Skeleton only, no text panel")
    ap.add_argument("--tip-interval", type=float, default=12.0, help="Seconds between rotating fundamentals")
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto = MPS on Apple Silicon when available; or cpu, mps, 0, cuda:0, …",
    )
    ap.add_argument("--cpu", action="store_true", help="Force CPU (overrides --device)")
    ap.add_argument(
        "--half",
        action="store_true",
        help="FP16 inference (often faster on MPS; disable if you see errors)",
    )
    args = ap.parse_args()

    if args.cpu:
        args.device = "cpu"

    if args.smooth_display and args.direct:
        raise SystemExit("Use only one of --direct and --smooth-display")
    if args.smooth_display:
        args.ema_new = 0.12
        args.ema_conf = 0.22
        args.hold = max(args.hold, 6)
    if args.direct:
        args.ema_new = 1.0
        args.ema_conf = 1.0
        args.hold = min(args.hold, 2)
    elif not args.smooth_display and args.ema_new >= 0.999:
        args.hold = min(args.hold, 3)

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    grab_thread: LatestFrameThread | None = None
    if not args.sync_capture:
        grab_thread = LatestFrameThread(cap, mirror=args.mirror)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps < 1:
        fps = 30.0

    writer: cv2.VideoWriter | None = None
    recording_kpts: list[np.ndarray | None] = []
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)

    tracker = SmoothPoseTracker(
        ema_new=args.ema_new,
        ema_conf=args.ema_conf,
        hold_frames=args.hold,
        select_policy=args.select,
    )

    t0 = time.time()
    tip_index = 0
    last_tip_switch = t0

    if args.smooth_display:
        mode = "smooth (heavy EMA)"
    elif args.direct or args.ema_new >= 0.999:
        mode = "snap (no EMA lag)"
    else:
        mode = f"balanced (ema_new={args.ema_new})"
    dev = default_mps_device() if args.device == "auto" else args.device
    if dev is not None and dev != "":
        dev_label = dev
    else:
        dev_label = "auto (Ultralytics default)"
    cap_mode = "threaded grab" if grab_thread is not None else "sync"
    print(
        f"q / ESC — quit   |   mode: {mode}   |   capture: {cap_mode}   |   select: {args.select}   |   device: {dev_label}",
        f"   |   imgsz={args.imgsz}   |   max_det={args.max_det}   |   recording:",
        args.record or "off",
    )

    pred_kw: dict = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
        "max_det": max(1, args.max_det),
    }
    if dev is not None and dev != "":
        pred_kw["device"] = dev
    if args.half:
        pred_kw["half"] = True

    # Warm up device / kernels so first seconds are not extra laggy
    _warm = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    for _ in range(2):
        model.predict(_warm, **pred_kw)

    while True:
        if grab_thread is not None:
            ok, frame = grab_thread.read()
            if not ok or frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                time.sleep(0.005)
                continue
        else:
            drain_capture_buffer(cap, args.buffer_drain)
            ok, frame = cap.read()
            if ok and args.mirror:
                frame = cv2.flip(frame, 1)
            if not ok or frame is None:
                break

        h, w = frame.shape[:2]
        results = model.predict(frame, **pred_kw)
        r0 = results[0]

        skel = tracker.update(r0, (h, w))

        out = frame.copy()
        if skel is not None:
            lw = max(2, int(round((h + w) / 420)))
            ann = Annotator(out, line_width=lw)
            ann.kpts(skel, shape=(h, w), kpt_line=True, conf_thres=args.kpt_conf, radius=max(2, lw))

        if not args.no_insights:
            now = time.time()
            if now - last_tip_switch >= args.tip_interval:
                tip_index = (tip_index + 1) % len(FUNDAMENTALS)
                last_tip_switch = now
            title, body = FUNDAMENTALS[tip_index]
            extra = posture_note_from_kpts(skel, w, h, args.kpt_conf) if skel is not None else None
            if extra is None and skel is None:
                extra = "Step into frame — we’ll overlay cues once a pose is tracked."
            draw_insights_panel(out, title, body, extra)

        if args.record:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.record), fourcc, fps, (w, h))
            writer.write(out)
            recording_kpts.append(skel.copy() if skel is not None else None)

        cv2.imshow("Pose — q / ESC quit", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    if grab_thread is not None:
        grab_thread.stop()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if (
        args.record
        and not args.no_record_report
        and len(recording_kpts) > 0
        and args.record.is_file()
    ):
        ak = args.analyze_kpt_conf if args.analyze_kpt_conf is not None else args.kpt_conf
        events, summary = analyze_recording_keypoints(
            recording_kpts,
            fps,
            kpt_conf=ak,
            punch_min_elbow_deg=args.punch_elbow_deg,
            kick_min_knee_deg=args.kick_knee_deg,
            strikes_only=not args.include_soft_motion_peaks,
        )
        snap_dir = args.snapshots_dir
        if snap_dir is None:
            snap_dir = args.record.parent / f"{args.record.stem}_strike_snapshots"
        summary_path = args.record.with_name(f"{args.record.stem}_strike_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "events": events,
                    "recording": str(args.record.resolve()),
                    "snapshots_dir": str(snap_dir.resolve()),
                },
                f,
                indent=2,
            )
        redraw_snaps = args.snapshot_redraw_skeleton or args.no_insights
        saved = save_strike_snapshots(
            args.record,
            recording_kpts,
            events,
            snap_dir,
            kinds_to_snap=frozenset({"punch_candidate", "kick_candidate"}),
            redraw_skeleton=redraw_snaps,
            kpt_draw_conf=args.kpt_conf,
        )
        print()
        print("--- Recording strike report (pose heuristics, not official stats) ---")
        extra = " (strikes only)" if not args.include_soft_motion_peaks else " (+ soft motion peaks)"
        print(
            f"  punch_candidate: {summary['n_punch_candidate']}   "
            f"kick_candidate: {summary['n_kick_candidate']}{extra}"
        )
        print(f"  Summary JSON: {summary_path}")
        print(f"  Snapshot PNGs: {len(saved)} saved under {snap_dir}")


if __name__ == "__main__":
    main()
