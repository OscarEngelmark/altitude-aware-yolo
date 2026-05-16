"""
Compare altitude estimates: current method (np.polyfit, raw 1/l) vs.
authors' method (RANSAC, smoothed polynomial).

Reads data/processed/metadata.json and data/video_data.csv.
For each video, re-runs both estimation methods from the stored
mean_diag_px values and plots:

  Left panel:  altitude (m) vs frame index — scatter overlay
  Right panel: altitude histogram — overlay both methods

Also prints a per-video summary table with mean, std, and KS test
statistics comparing the two distributions.

Usage
-----
    cd src && python plots/altitude_method_comparison.py
    cd src && python plots/altitude_method_comparison.py --out results/method_comparison.png
    cd src && python plots/altitude_method_comparison.py --video "2022-12-02 Asjo 01_stabilized"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import globals as g
from frame_metadata import (
    estimate_altitudes,
    estimate_altitudes_ransac,
    load_video_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path, default=None,
        help="output path for figure (default: results/method_comparison.png)",
    )
    p.add_argument(
        "--video", type=str, default=None,
        help="restrict comparison to a single video name",
    )
    return p.parse_args()


def _print_summary(
    video: str,
    current: List[float],
    ransac: List[float],
) -> None:
    ks_stat, ks_p = stats.ks_2samp(current, ransac)
    print(
        f"\n{video}\n"
        f"  {'':20s} {'mean':>8s} {'std':>8s}\n"
        f"  {'current (polyfit)':20s} {np.mean(current):>8.1f} {np.std(current):>8.1f}\n"
        f"  {'authors (RANSAC)':20s} {np.mean(ransac):>8.1f} {np.std(ransac):>8.1f}\n"
        f"  KS statistic: {ks_stat:.4f}   p-value: {ks_p:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = g.RESULTS_DIR / "method_comparison.png"

    with open(g.OUT_DIR / "metadata.json") as f:
        metadata: Dict[str, Dict[str, Any]] = json.load(f)

    video_csv = load_video_csv()

    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in metadata.values():
        by_video[entry["video"]].append(entry)

    if args.video:
        if args.video not in by_video:
            sys.exit(f"Video '{args.video}' not found in metadata.json")
        videos = [args.video]
    else:
        videos = sorted(by_video)

    n = len(videos)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle(
        "Altitude estimation: current (polyfit) vs. authors' (RANSAC)",
        fontsize=12,
    )

    print(
        f"\n{'Video':40s} {'method':20s} {'mean':>8s} {'std':>8s} "
        f"{'KS stat':>10s} {'p-value':>10s}"
    )
    print("-" * 100)

    for row, video_stem in enumerate(videos):
        frames_sorted = sorted(
            by_video[video_stem], key=lambda e: e["frame_id"]
        )
        frame_diagonals: Dict[int, float] = {
            int(e["frame_id"]): e["mean_diag_px"]
            for e in frames_sorted
            if e.get("mean_diag_px") is not None
        }
        frame_ids = np.array(sorted(frame_diagonals), dtype=float)

        vmeta = video_csv.get(video_stem, {})
        h_max: Optional[float] = vmeta.get("h_max")
        if h_max is None:
            print(f"  [skip] {video_stem}: no h_max in video_data.csv")
            continue

        alts_current = estimate_altitudes(frame_diagonals, h_max)
        alts_ransac  = estimate_altitudes_ransac(frame_diagonals, h_max)

        cur_vals = np.array([alts_current[f] for f in frame_ids.astype(int)])
        ran_vals = np.array([alts_ransac[f]  for f in frame_ids.astype(int)])

        ks_stat, ks_p = stats.ks_2samp(cur_vals, ran_vals)
        print(
            f"  {video_stem:38s} {'current':20s} "
            f"{np.mean(cur_vals):>8.1f} {np.std(cur_vals):>8.1f} "
            f"{ks_stat:>10.4f} {ks_p:>10.4f}"
        )
        print(
            f"  {'':38s} {'RANSAC':20s} "
            f"{np.mean(ran_vals):>8.1f} {np.std(ran_vals):>8.1f}"
        )

        # Left: altitude vs frame index
        ax_left = axes[row, 0]
        ax_left.scatter(
            frame_ids, cur_vals, s=8, alpha=0.5,
            label="current (polyfit, raw 1/l)",
        )
        ax_left.scatter(
            frame_ids, ran_vals, s=8, alpha=0.5,
            label="authors (RANSAC, smoothed)",
        )
        ax_left.axhline(h_max, color="black", ls="--", lw=1,
                        label=f"h_max = {h_max:.0f} m")
        ax_left.set_title(video_stem)
        ax_left.set_xlabel("frame index")
        ax_left.set_ylabel("altitude (m)")
        ax_left.legend(fontsize=7)

        # Right: histogram overlay
        ax_right = axes[row, 1]
        bins = np.linspace(
            min(cur_vals.min(), ran_vals.min()),
            max(cur_vals.max(), ran_vals.max()),
            40,
        )
        ax_right.hist(
            cur_vals, bins=bins, alpha=0.5,
            label=f"current  μ={np.mean(cur_vals):.0f} σ={np.std(cur_vals):.0f}",
        )
        ax_right.hist(
            ran_vals, bins=bins, alpha=0.5,
            label=f"RANSAC   μ={np.mean(ran_vals):.0f} σ={np.std(ran_vals):.0f}",
        )
        ax_right.set_title(f"KS stat={ks_stat:.3f}  p={ks_p:.3f}")
        ax_right.set_xlabel("altitude (m)")
        ax_right.set_ylabel("frame count")
        ax_right.legend(fontsize=7)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
