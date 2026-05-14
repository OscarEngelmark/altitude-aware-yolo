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
) -> Dict[int, float]:
    """Per-frame altitude estimate (metres) using the paper's method.

    H ∝ 1/l (perspective geometry: boxes appear smaller at greater altitude).
    Fits a 4th-degree polynomial to (frame_id, 1/l) across all annotated
    frames. The polynomial maximum corresponds to the frame where the drone
    is highest (H_max). Per-frame altitude is then:

        H_frame = H_max · (1/l_frame) / poly_max

    The time axis is normalised to [0, 1] for numerical stability.

    Also returns the fitted polynomial coefficients and frame time axis for
    use in diagnostic plots — see estimate_altitudes_with_fit().
    """
    if not frame_diagonals:
        return {}

    frames     = np.array(sorted(frame_diagonals.keys()), dtype=float)
    diags      = np.array([frame_diagonals[int(f)] for f in frames])
    inv_diags  = 1.0 / diags

    span = max(frames.max() - frames.min(), 1.0)
    t    = (frames - frames.min()) / span

    if len(frames) >= 5:
        coeffs   = np.polyfit(t, inv_diags, deg=4)
        t_dense  = np.linspace(0.0, 1.0, 1000)
        poly_max = float(np.polyval(coeffs, t_dense).max())
    else:
        poly_max = float(inv_diags.max())

    return {
        int(f): float(h_max * inv_d / poly_max)
        for f, inv_d in zip(frames, inv_diags)
    }


def estimate_altitudes_with_fit(
    frame_diagonals: Dict[int, float], h_max: float
) -> Tuple[Dict[int, float], np.ndarray, np.ndarray, np.ndarray]:
    """Same as estimate_altitudes but also returns the polynomial fit for 
    plots.

    Returns
    -------
    altitudes   : {frame_id: altitude_m}
    frames      : sorted frame indices (1-D array)
    t           : normalised time axis in [0, 1] for each frame
    coeffs      : polynomial coefficients (degree-4, fit to 1/l values)
    """
    if not frame_diagonals:
        return {}, np.array([]), np.array([]), np.array([])

    frames     = np.array(sorted(frame_diagonals.keys()), dtype=float)
    diags      = np.array([frame_diagonals[int(f)] for f in frames])
    inv_diags  = 1.0 / diags

    span = max(frames.max() - frames.min(), 1.0)
    t    = (frames - frames.min()) / span

    if len(frames) >= 5:
        coeffs   = np.polyfit(t, inv_diags, deg=4)
        t_dense  = np.linspace(0.0, 1.0, 1000)
        poly_max = float(np.polyval(coeffs, t_dense).max())
    else:
        coeffs   = np.array([])
        poly_max = float(inv_diags.max())

    altitudes = {
        int(f): float(h_max * inv_d / poly_max)
        for f, inv_d in zip(frames, inv_diags)
    }
    return altitudes, frames, t, coeffs


def estimate_altitudes_ransac(
    frame_diagonals: Dict[int, float], h_max: float
) -> Dict[int, float]:
    """Per-frame altitude estimate using the authors' RANSAC method.

    Identical physical model to estimate_altitudes() (H ∝ 1/l), but fits the
    degree-4 polynomial with RANSACRegressor for outlier robustness, and uses
    the smoothed polynomial predictions as per-frame altitudes rather than raw
    1/l values.  Falls back to the plain np.polyfit path when there are fewer
    than 20 frames (RANSAC min_samples requirement).

    Parameters match estimate_altitudes(); return type is identical.
    """
    if not frame_diagonals:
        return {}

    frames    = np.array(sorted(frame_diagonals.keys()), dtype=float)
    diags     = np.array([frame_diagonals[int(f)] for f in frames])
    inv_diags = 1.0 / diags

    if len(frames) >= 20:
        ransac = RANSACRegressor(
            make_pipeline(PolynomialFeatures(4), LinearRegression()),
            residual_threshold=2.0,
            random_state=0,
            min_samples=20,
        )
        ransac.fit(frames[:, np.newaxis], inv_diags)
        line_x   = np.linspace(frames.min(), frames.max(), 1000)
        poly_max = float(ransac.predict(line_x[:, np.newaxis]).max())
        smoothed = ransac.predict(frames[:, np.newaxis])
    else:
        # Fewer than 20 frames: RANSAC min_samples cannot be satisfied; fall
        # back to the plain polyfit path used by estimate_altitudes().
        span = max(frames.max() - frames.min(), 1.0)
        t    = (frames - frames.min()) / span
        if len(frames) >= 5:
            coeffs   = np.polyfit(t, inv_diags, deg=4)
            t_dense  = np.linspace(0.0, 1.0, 1000)
            poly_max = float(np.polyval(coeffs, t_dense).max())
            smoothed = np.polyval(coeffs, t)
        else:
            poly_max = float(inv_diags.max())
            smoothed = inv_diags

    height_scale = h_max / poly_max
    return {
        int(f): float(s * height_scale)
        for f, s in zip(frames, smoothed)
    }


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
    altitudes = estimate_altitudes_ransac(diagonals, video_meta["h_max"])
    return {
        frame_id: {
            "mean_diag_px": diagonals.get(frame_id),
            "altitude_m":   altitudes.get(frame_id),
        }
        for frame_id in annotations
    }
