"""
Plot altitude distribution of cars per dataset split.

By default (scale=0, no mosaic) the raw estimated altitudes are shown.

--scale / --mosaic: simulate standard YOLOv9 scale augmentation:
    apparent_altitude = actual * mosaic_factor / U(1-scale, 1+scale)

--altitude-aware: simulate altitude-aware augmentation where for each
training frame at altitude h, a target altitude is sampled over
[alt_min, alt_max] and the required scale factor s = h / h_target is
applied:
    apparent_altitude = h_target  (exactly, when s is within clamp bounds)
Scale factors are clamped to [0.1, 4.0] to stay within feasible image
scaling limits; frames where clamping alters the target are still included
but their apparent altitude will differ from h_target.

--dist: target altitude distribution to use with --altitude-aware.
    uniform    (default): h_target ~ U(alt_min, alt_max)
    triangular:           h_target ~ Triangular(alt_min, alt_mode, alt_max)
                          --alt-mode sets the peak (default: midpoint)

Augmented altitudes are weighted by n_boxes / n_samples so total car count
is preserved.

Usage
-----
    cd src && python plots/altitude_dist.py
    cd src && python plots/altitude_dist.py --scale 0.7
    cd src && python plots/altitude_dist.py --altitude-aware
    cd src && python plots/altitude_dist.py --altitude-aware --alt-min 80 \
        --alt-max 400
    cd src && python plots/altitude_dist.py --altitude-aware \
        --dist triangular --alt-mode 250
    cd src && python plots/altitude_dist.py --out results/altitude_dist.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

import globals as g
import style
from altitude_augment import SCALE_CEILING, SCALE_FLOOR

SPLITS = ["train", "val", "test"]
COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
TRAIN_AUG_COLOR = "#9467BD"

DEFAULT_SCALE = 0.0
DEFAULT_N_SAMPLES = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: results/altitude_dist.{pdf|png})",
    )
    p.add_argument(
        "--style", choices=style.STYLES, default=None,
        help="Output style: 'report' (PDF, small fonts) or 'ppt' "
             "(PNG, large fonts)",
    )
    p.add_argument(
        "--bins", type=int, default=100,
        help="Number of histogram bins (default: 100)",
    )
    p.add_argument(
        "--scale", type=float, default=DEFAULT_SCALE,
        help=(
            f"Random scale jitter range (default: {DEFAULT_SCALE}). "
            "Augmentation factor = mosaic_factor / U(1-scale, 1+scale)."
        ),
    )
    p.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        dest="n_samples",
        help=(
            f"Augmentation draws per training frame "
            f"(default: {DEFAULT_N_SAMPLES})"
        ),
    )
    p.add_argument(
        "--altitude-aware", action="store_true", dest="altitude_aware",
        help=(
            "Simulate altitude-aware scale augmentation: sample "
            "h_target ~ U(alt_min, alt_max) per frame, apply s = h/h_target"
        ),
    )
    p.add_argument(
        "--alt-min", type=float, default=100.0, dest="alt_min",
        help="Lower bound of target altitude range in metres (default: 80)",
    )
    p.add_argument(
        "--alt-max", type=float, default=300.0, dest="alt_max",
        help="Upper bound of target altitude range in metres (default: 300)",
    )
    p.add_argument(
        "--dist", choices=["uniform", "triangular"], default="uniform",
        help=(
            "Target altitude distribution for --altitude-aware: "
            "'uniform' (default) or 'triangular' (see --alt-mode)"
        ),
    )
    p.add_argument(
        "--alt-mode", type=float, default=None, dest="alt_mode",
        help=(
            "Peak of the triangular target distribution in metres. "
            "Only used with --dist triangular. "
            "Defaults to midpoint of [alt_min, alt_max]."
        ),
    )
    p.add_argument(
        "--x-max", type=float, default=350.0, dest="x_max",
        help="Upper x-axis limit for all histogram plots (default: 350)",
    )
    p.add_argument(
        "--seed", type=int, default=g.SEED,
        help="RNG seed for reproducibility",
    )
    return p.parse_args()


def load_split_data(
    metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, List[float]]]:
    """Return per-split dicts with 'alts' and 'weights' (n_boxes) lists."""
    by_split: Dict[str, Dict[str, List[float]]] = {
        s: {"alts": [], "weights": []} for s in SPLITS
    }
    for entry in metadata.values():
        alt = entry.get("altitude_m")
        split = entry.get("split")
        n_boxes = entry.get("n_boxes", 1)
        if alt is not None and split in SPLITS:
            by_split[split]["alts"].append(float(alt))
            by_split[split]["weights"].append(float(n_boxes))
    return by_split


def augment_train(
    alts: List[float],
    weights: List[float],
    scale: float,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[List[float], List[float]]:
    """Return (augmented_alts, augmented_weights) for the training split.

    apparent_altitude = actual / U(1-scale, 1+scale)
    Weights are divided by n_samples to preserve total car count.
    When scale=0, returns the inputs unchanged.
    """
    if scale == 0.0:
        return alts, weights

    alts_arr = np.array(alts)
    w_arr = np.array(weights) / n_samples

    raw = rng.uniform(1.0 - scale, 1.0 + scale, size=(len(alts), n_samples))
    aug_alts = (alts_arr[:, None] / raw).ravel()
    aug_weights = np.repeat(w_arr, n_samples)

    mask = aug_alts > 0
    return aug_alts[mask].tolist(), aug_weights[mask].tolist()


def augment_train_altitude_aware(
    alts: List[float],
    weights: List[float],
    alt_min: float,
    alt_max: float,
    n_samples: int,
    rng: np.random.Generator,
    dist: str = "uniform",
    alt_mode: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    """Altitude-aware augmentation: sample h_target from the chosen
    distribution over [alt_min, alt_max], compute s = eff_alt / h_target,
    clamp to [SCALE_FLOOR, SCALE_CEILING], then apparent_altitude =
    eff_alt / s.  When s is unclamped, apparent_altitude == h_target exactly.

    Mosaic uses a center crop (not a downscale), so objects appear at full
    tile resolution and eff_alt = h (no factor needed regardless of mosaic).

    dist='uniform'    -> h_target ~ U(alt_min, alt_max)
    dist='triangular' -> h_target ~ Triangular(alt_min, alt_mode, alt_max)
                         alt_mode defaults to midpoint when None
    """
    alts_arr = np.array(alts)
    w_arr = np.array(weights) / n_samples

    eff_alts = alts_arr

    size = (len(alts), n_samples)
    if dist == "triangular":
        mode = alt_mode if alt_mode is not None else (alt_min + alt_max) / 2
        h_target = rng.triangular(alt_min, mode, alt_max, size=size)
    else:
        h_target = rng.uniform(alt_min, alt_max, size=size)

    s = eff_alts[:, None] / h_target
    s = np.clip(s, SCALE_FLOOR, SCALE_CEILING)

    aug_alts = (eff_alts[:, None] / s).ravel()
    aug_weights = np.repeat(w_arr, n_samples)
    return aug_alts.tolist(), aug_weights.tolist()


def plot_histograms(
    by_split: Dict[str, Dict[str, List[float]]],
    aug_alts: List[float],
    aug_weights: List[float],
    bins: int,
    axes: List[matplotlib.axes.Axes],
    x_max: float,
    train_augmented: bool,
) -> None:
    all_alts = (
        [a for s in by_split.values() for a in s["alts"]]
        + aug_alts
    )
    x_min = min(all_alts)
    edges = np.linspace(x_min, x_max, bins + 1).tolist()

    train_label = "train (augmented)" if train_augmented else "train"
    train_color = TRAIN_AUG_COLOR if train_augmented else COLORS["train"]
    rows = [
        (train_label, aug_alts,                   aug_weights,
         train_color),
        ("val",        by_split["val"]["alts"],   by_split["val"]["weights"],
         COLORS["val"]),
        ("test",       by_split["test"]["alts"],  by_split["test"]["weights"],
         COLORS["test"]),
    ]
    for ax, (label, alts, weights, color) in zip(axes, rows):
        total_cars = sum(weights)
        ax.hist(
            alts, bins=edges, weights=weights,
            color=color, label=f"cars={total_cars:,.0f}",
        )
        ax.set_title(label.capitalize())
        ax.set_xlabel("Estimated altitude (m)")
        ax.set_ylabel("Car count")
        ax.legend()

    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(top=y_max)


def main() -> None:
    args = parse_args()
    fmt = style.output_fmt(args.style) if args.style else "png"
    dpi = style.save_dpi(args.style) if args.style else 150
    if args.out is None:
        args.out = g.RESULTS_DIR / f"altitude_dist.{fmt}"
    if args.style:
        style.apply_style(args.style)
    rng = np.random.default_rng(args.seed)

    with open(g.OUT_DIR / "metadata.json") as f:
        metadata: Dict[str, Dict[str, Any]] = json.load(f)

    by_split = load_split_data(metadata)

    if not by_split["train"]["alts"]:
        raise ValueError("No training data found in metadata.json")

    if args.altitude_aware:
        mode = args.alt_mode
        if args.dist == "triangular" and mode is None:
            mode = (args.alt_min + args.alt_max) / 2
        aug_alts, aug_weights = augment_train_altitude_aware(
            by_split["train"]["alts"],
            by_split["train"]["weights"],
            alt_min=args.alt_min,
            alt_max=args.alt_max,
            n_samples=args.n_samples,
            rng=rng,
            dist=args.dist,
            alt_mode=mode,
        )
        train_augmented = True
        if args.dist == "triangular":
            title = (
                f"Altitude distribution — altitude-aware augmentation "
                f"(target: triangular({args.alt_min:.0f}, "
                f"{mode:.0f}, {args.alt_max:.0f}) m)"
            )
        else:
            title = (
                f"Altitude distribution — altitude-aware augmentation "
                f"(target: {args.alt_min:.0f}–{args.alt_max:.0f} m)"
            )
    else:
        aug_alts, aug_weights = augment_train(
            by_split["train"]["alts"],
            by_split["train"]["weights"],
            scale=args.scale,
            n_samples=args.n_samples,
            rng=rng,
        )
        train_augmented = args.scale != 0.0
        if train_augmented:
            title = (
                f"Altitude distribution — train augmented "
                f"(scale={args.scale})"
            )
        else:
            title = "Altitude distribution by split"

    fs = style.figsize(args.style, n_rows=3) if args.style else (12, 10)
    fig, axes = plt.subplots(
        3, 1, figsize=fs, sharex=True, squeeze=False
    )
    plot_histograms(
        by_split, aug_alts, aug_weights,
        bins=args.bins,
        axes=list(axes[:, 0]),
        x_max=args.x_max,
        train_augmented=train_augmented,
    )
    fig.suptitle(title)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
