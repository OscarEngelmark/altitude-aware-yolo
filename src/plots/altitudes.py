"""
Sanity check: plot per-frame altitude estimates and 1/l polynomial fits.

Reads data/processed/metadata.json (written by preprocessing.py) and
data/video_data.csv.  Produces one column per video with two rows:

  Top:    mean bounding-box diagonal (px) vs frame index
          + 4th-degree polynomial fit to 1/l back-projected to px
  Bottom: estimated altitude (m) vs frame index
          + h_max ceiling (red dashed) and h_min floor (orange dashed)

Usage
-----
    cd src && python plots/altitudes.py
    cd src && python plots/altitudes.py --out results/altitudes.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import globals as g
import style
from frame_metadata import load_video_csv, estimate_altitudes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path, default=None,
        help="output path (default: results/altitudes.{pdf|png})",
    )
    p.add_argument(
        "--style", choices=style.STYLES, default=None,
        help="output style: 'report' (PDF, small fonts) or 'ppt' "
             "(PNG, large fonts)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fmt = style.output_fmt(args.style) if args.style else "png"
    dpi = style.save_dpi(args.style) if args.style else 150
    if args.out is None:
        args.out = g.RESULTS_DIR / f"altitudes.{fmt}"
    if args.style:
        style.apply_style(args.style)

    with open(g.OUT_DIR / "metadata.json") as f:
        metadata: Dict[str, Dict[str, Any]] = json.load(f)

    video_csv = load_video_csv()

    # Group frame entries by video stem
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in metadata.values():
        by_video[entry["video"]].append(entry)

    videos    = sorted(by_video)
    n_videos  = len(videos)
    fs = (
        style.figsize(args.style, n_rows=2, n_cols=n_videos)
        if args.style else (5 * n_videos, 8)
    )
    fig, axes = plt.subplots(2, n_videos, figsize=fs, squeeze=False)

    for col, video_stem in enumerate(videos):
        frames_sorted = sorted(
            by_video[video_stem], key=lambda e: e["frame_id"]
        )
        frame_ids = np.array(
            [e["frame_id"]     for e in frames_sorted], dtype=float
        )
        diags = np.array(
            [e["mean_diag_px"] for e in frames_sorted], dtype=float
        )
        alts = np.array(
            [e["altitude_m"]   for e in frames_sorted], dtype=float
        )

        vmeta        = video_csv.get(video_stem, {})
        h_max        = vmeta.get("h_max")
        altitude_str = vmeta.get("altitude_str", "")

        # Refit polynomial (new algorithm) from stored diagonals
        frame_diagonals = {
            int(e["frame_id"]): e["mean_diag_px"] for e in frames_sorted
        }
        _, _, _, coeffs = estimate_altitudes(
            frame_diagonals, h_max or 1.0
        )

        span    = max(frame_ids.max() - frame_ids.min(), 1.0)
        t_dense = np.linspace(0.0, 1.0, 1000)
        f_dense = frame_ids.min() + t_dense * span

        # ── top: diagonal scatter + polynomial back-projected to px ─────────
        ax_top = axes[0, col]
        ax_top.scatter(frame_ids, diags, s=12, alpha=0.6, label="raw diagonal")
        if coeffs.size > 0:
            inv_poly = np.polyval(coeffs, t_dense)
            with np.errstate(divide="ignore", invalid="ignore"):
                diag_poly = np.where(inv_poly > 0, 1.0 / inv_poly, np.nan)
            ax_top.plot(
                f_dense, diag_poly, "r-", lw=1.5,
                label="poly fit (1/l → px)",
            )
        ax_top.set_title(video_stem.replace("_", " "))
        ax_top.set_xlabel("frame index")
        ax_top.set_ylabel("mean diagonal (px)")
        ax_top.legend()

        # ── bottom: altitude scatter + polynomial in altitude space ─────────
        ax_bot = axes[1, col]
        ax_bot.scatter(frame_ids, alts, s=12, alpha=0.6, label="est. altitude")

        if coeffs.size > 0 and h_max is not None:
            inv_poly = np.polyval(coeffs, t_dense)
            poly_max = float(inv_poly.max())
            alt_poly = h_max * inv_poly / poly_max
            ax_bot.plot(f_dense, alt_poly, "r-", lw=1.5, label="poly fit")

        if h_max is not None:
            ax_bot.axhline(h_max, color="red", ls="--", lw=1,
                           label=f"h_max = {h_max:.0f} m")
            parts = altitude_str.lower().replace(" m", "").strip().split("-")
            if len(parts) == 2:
                h_min = float(parts[0])
                ax_bot.axhline(h_min, color="orange", ls="--", lw=1,
                               label=f"h_min = {h_min:.0f} m")

        ax_bot.set_xlabel("frame index")
        ax_bot.set_ylabel("altitude (m)")
        ax_bot.legend()

    fig.suptitle("Altitude sanity check — per video")
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
