"""
Box size vs altitude diagnostic plot.

For each annotated frame, plots the median OBB short-side length (px)
against estimated flight altitude (m).  Overlays a fitted C/H hyperbola
(perspective geometry: short_side ≈ C / altitude) and horizontal reference
lines at each YOLOv9 FPN stride (8, 16, 32 px) with corresponding critical
altitudes marked on the x-axis.

The critical altitude for stride S is the altitude at which the median car
short side equals S pixels — above this the model's stride-S feature map
can no longer resolve the object.

Usage
-----
    cd src && python plots/size_vs_altitude.py
    cd src && python plots/size_vs_altitude.py --splits train val
    cd src && python plots/size_vs_altitude.py \
        --out results/size_vs_altitude.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import globals as g
import style

IMG_W = 1920
IMG_H = 1080
FPN_STRIDES = [8, 16, 32]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="which splits to include (default: all)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="output path (default: results/size_vs_altitude.{pdf|png})",
    )
    p.add_argument(
        "--style", choices=style.STYLES, default=None,
        help="output style: 'report' (PDF, small fonts) or 'ppt' "
             "(PNG, large fonts)",
    )
    return p.parse_args()


def obb_short_side_px(pts: np.ndarray, iw: int, ih: int) -> float:
    """Return the OBB short-side length in pixels."""
    pts_px = pts * np.array([iw, ih])
    a = float(np.linalg.norm(pts_px[1] - pts_px[0]))
    b = float(np.linalg.norm(pts_px[2] - pts_px[1]))
    return min(a, b)


def load_data(
    splits: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (altitudes_m, short_sides_px) arrays, one entry per box.

    Boxes from frames without an altitude estimate are dropped.
    """
    from PIL import Image

    meta_path = g.OUT_DIR / "metadata.json"
    try:
        with open(meta_path) as fh:
            metadata: Dict[str, Dict] = json.load(fh)
    except FileNotFoundError:
        raise RuntimeError(
            "metadata.json not found — run preprocessing.py first."
        )

    img_cache: Dict[Path, Tuple[int, int]] = {}

    def img_wh(lbl_path: Path, split: str) -> Tuple[int, int]:
        p = g.IMG_DIR / split / (lbl_path.stem + ".jpg")
        if p not in img_cache:
            try:
                img_cache[p] = Image.open(p).size
            except FileNotFoundError:
                img_cache[p] = (IMG_W, IMG_H)
        return img_cache[p]

    altitudes: List[float] = []
    short_sides: List[float] = []

    for split in splits:
        lbl_dir = g.LBL_DIR / split
        for path in sorted(lbl_dir.glob("*.txt")):
            frame_meta = metadata.get(path.stem, {})
            alt = frame_meta.get("altitude_m")
            if alt is None:
                continue
            iw, ih = img_wh(path, split)
            for line in path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                pts = np.array(parts[1:], dtype=float).reshape(4, 2)
                short_sides.append(obb_short_side_px(pts, iw, ih))
                altitudes.append(float(alt))

    if not altitudes:
        raise RuntimeError("No altitude-annotated boxes found.")

    return np.array(altitudes), np.array(short_sides)


def fit_perspective_constant(
    altitudes: np.ndarray, short_sides: np.ndarray
) -> float:
    """Estimate C in short_side ≈ C / altitude (median estimator)."""
    return float(np.median(short_sides * altitudes))


def print_thresholds(C: float, n: int) -> None:
    print(f"\n  Perspective constant  C = {C:.0f} px·m")
    print(f"  (short_side ≈ C / altitude,  {n:,} boxes)")
    print(
        "\n  Altitude where short side drops below FPN stride threshold:"
    )
    for stride in FPN_STRIDES:
        print(f"    stride {stride:2d} px  →  H_crit ≈ {C / stride:.0f} m")
    print()


def main() -> None:
    args = parse_args()

    fmt = style.output_fmt(args.style) if args.style else "png"
    dpi = style.save_dpi(args.style) if args.style else 150
    if args.out is None:
        args.out = g.RESULTS_DIR / f"size_vs_altitude.{fmt}"
    if args.style:
        style.apply_style(args.style)
    print(f"Loading data from splits: {args.splits} …")
    altitudes, short_sides = load_data(args.splits)
    print(f"  {len(altitudes):,} boxes with altitude loaded.")

    C = fit_perspective_constant(altitudes, short_sides)
    print_thresholds(C, len(altitudes))

    # ── plot ─────────────────────────────────────────────────────────────
    fs = style.figsize(args.style) if args.style else (9, 6)
    fig, ax = plt.subplots(figsize=fs)

    ax.scatter(
        altitudes, short_sides,
        s=6, alpha=0.15, color="steelblue", linewidths=0,
        label="individual boxes",
    )

    # per-altitude median line
    alt_bins = np.unique(np.round(altitudes / 10) * 10)
    bin_meds: List[Tuple[float, float]] = []
    for b in alt_bins:
        mask = np.abs(altitudes - b) <= 5
        if mask.sum() >= 3:
            bin_meds.append((b, float(np.median(short_sides[mask]))))
    if bin_meds:
        bx, by = zip(*sorted(bin_meds))
        ax.plot(bx, by, "o-", color="navy", ms=5, lw=1.2,
                label="median per 10 m bin")

    # C/H fit
    h_lo = max(altitudes.min() * 0.80, 1.0)
    h_hi = max(altitudes.max() * 1.25,
               max(C / s for s in FPN_STRIDES) * 1.05)
    h_range = np.linspace(h_lo, h_hi, 400)
    ax.plot(
        h_range, C / h_range,
        lw=2.0, color="black", ls="--",
        label=f"$C/H$ fit  ($C = {C:.0f}$ px·m)",
    )

    # FPN stride thresholds
    stride_colors = ["steelblue", "firebrick", "darkorange"]
    for stride, col in zip(FPN_STRIDES, stride_colors):
        H_crit = C / stride
        ax.axhline(stride, lw=1.1, color=col, ls=":",
                   label=f"stride {stride} px  →  H_crit ≈ {H_crit:.0f} m")
        ax.axvline(H_crit, lw=0.9, color=col, ls=":", alpha=0.6)
        ax.text(
            H_crit, 0.99, f" {H_crit:.0f} m",
            color=col, fontsize=7, va="top",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlabel("flight altitude (m)")
    ax.set_ylabel("OBB short side (px)")
    ax.set_title(
        f"Box size vs altitude  "
        f"(splits: {', '.join(args.splits)},  n={len(altitudes):,})"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlim(left=h_lo)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
