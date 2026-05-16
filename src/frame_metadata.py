"""
Per-frame flight metadata computation for NVD drone videos.

Provides two public entry points:

  load_video_csv()
      Reads data/video_data.csv and returns a dict keyed by video name
      (matching the zip stem) with fields: h_max, altitude_str, snow_cover,
      cloud_cover.

  compute_frame_metadata(annotations, img_w, img_h, video_meta)
      Derives per-frame measurements from annotation data + video_meta and
      returns {frame_id: {"mean_diag_px": float, "altitude_m": float, ...}}.

Extension points
----------------
- To swap in a different altitude algorithm, replace estimate_altitudes().
- To add new per-frame signals (tilt, pitch, GSD, …), extend
  compute_frame_metadata() — its return dict is spread directly into each
  frame's metadata.json entry, so new keys appear automatically.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import globals as g

# ── CSV loading ─────────────────────────────────────────────────────────────

def _parse_h_max(altitude_str: str) -> float:
    """Parse maximum altitude (m) from
    '130-200 m' → 200.0 or '250 m' → 250.0."""
    s = altitude_str.lower().replace(" m", "").strip()
    parts = s.split("-")
    return float(parts[-1])


def load_video_csv() -> Dict[str, Dict[str, Any]]:
    """Load per-video metadata from data/video_data.csv.

    Returns a dict keyed by video name (CSV 'Video' column, which matches the
    zip stem without extension). Only rows where Annotated=TRUE are included.

    Each entry contains:
        h_max        : float  – maximum flight altitude in metres
        altitude_str : str    – raw string from CSV (e.g. '130-200 m')
        snow_cover   : str    – e.g. 'Minimal (0-1 cm)'
        cloud_cover  : str    – e.g. 'Overcast'
    """
    csv_path = g.DATA_DIR / "video_data.csv"
    result: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Annotated", "").strip().upper() != "TRUE":
                continue
            name = row["Video"].strip()
            result[name] = {
                "h_max":        _parse_h_max(row["Flight Altitude"].strip()),
                "altitude_str": row["Flight Altitude"].strip(),
                "snow_cover":   row["Snow Cover"].strip(),
                "cloud_cover":  row["Cloud Cover"].strip(),
            }
    return result


# ── per-frame computation ───────────────────────────────────────────────────

def compute_frame_diagonals(
    annotations: Dict[int, List[Tuple[float, float, float, float, float]]],
    img_w: int,
    img_h: int,
) -> Dict[int, float]:
    """Per-frame mean bounding-box diagonal length, in pixels.

    Rotation does not change a rectangle's diagonal, so the OBB angle is
    ignored here.
    """
    diagonals: Dict[int, float] = {}
    for frame_id, boxes in annotations.items():
        if not boxes:
            continue
        diags = [
            float(np.hypot(w * img_w, h * img_h))
            for _, _, w, h, _ in boxes
        ]
        diagonals[frame_id] = float(np.mean(diags))
    return diagonals


def estimate_altitudes(
    frame_diagonals: Dict[int, float], h_max: float
) -> Tuple[Dict[int, float], np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame altitude estimate (metres) using RANSAC on 1/l.

    H ∝ 1/l (perspective geometry: boxes appear smaller at greater altitude).
    Fits a degree-4 polynomial with RANSACRegressor to find the calibration
    peak (poly_max ↔ H_max). Per-frame altitude is then the raw 1/l value
    anchored to that peak: H = H_max · (1/l) / poly_max. Falls back to plain
    np.polyfit when fewer than 20 frames are available.

    Returns
    -------
    altitudes        : {frame_id: altitude_m}
    frames           : sorted frame indices (1-D array)
    dense_frames     : 1000-point frame indices for plotting
    dense_inv_diags  : predicted 1/l at dense_frames (from the fit)
    """
    if not frame_diagonals:
        return {}, np.array([]), np.array([]), np.array([])

    frames    = np.array(sorted(frame_diagonals.keys()), dtype=float)
    diags     = np.array([frame_diagonals[int(f)] for f in frames])
    inv_diags = 1.0 / diags

    dense_frames = np.linspace(frames.min(), frames.max(), 1000)

    if len(frames) >= 20:
        ransac = RANSACRegressor(
            make_pipeline(PolynomialFeatures(4), LinearRegression()),
            residual_threshold=2.0,
            random_state=0,
            min_samples=20,
        )
        ransac.fit(frames[:, np.newaxis], inv_diags)
        dense_inv_diags = ransac.predict(dense_frames[:, np.newaxis])
        poly_max        = float(dense_inv_diags.max())
    else:
        span    = max(frames.max() - frames.min(), 1.0)
        t       = (frames - frames.min()) / span
        t_dense = (dense_frames - frames.min()) / span
        if len(frames) >= 5:
            coeffs          = np.polyfit(t, inv_diags, deg=4)
            dense_inv_diags = np.polyval(coeffs, t_dense)
            poly_max        = float(dense_inv_diags.max())
        else:
            dense_inv_diags = np.array([])
            poly_max        = float(inv_diags.max())

    height_scale = h_max / poly_max
    altitudes = {
        int(f): float(inv_d * height_scale)
        for f, inv_d in zip(frames, inv_diags)
    }
    return altitudes, frames, dense_frames, dense_inv_diags


def compute_frame_metadata(
    annotations: Dict[int, List[Tuple[float, float, float, float, float]]],
    img_w: int,
    img_h: int,
    video_meta: Dict[str, Any],
) -> Dict[int, Dict[str, Optional[float]]]:
    """Compute per-frame metadata for every annotated frame in a video.

    Returns {frame_id: {field: value, ...}}. The dict is spread directly into
    each frame's metadata.json entry, so adding a new field here automatically
    propagates to storage and to the W&B callback.

    Parameters
    ----------
    annotations : {frame_id: [(cx, cy, w, h, angle), ...]}
    img_w, img_h : image dimensions in pixels
    video_meta   : entry from load_video_csv() for this video
    """
    diagonals = compute_frame_diagonals(annotations, img_w, img_h)
    altitudes, *_ = estimate_altitudes(diagonals, video_meta["h_max"])
    return {
        frame_id: {
            "mean_diag_px": diagonals.get(frame_id),
            "altitude_m":   altitudes.get(frame_id),
        }
        for frame_id in annotations
    }
