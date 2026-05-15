"""Training-curve comparison: baseline vs. AAS.

2×2 panel layout:
  (1,1) Class loss   — train (solid) + val (dashed), both runs
  (1,2) Box loss     — train (solid) + val (dashed), both runs
  (2,1) Val mAP50-95 — both runs
  (2,2) Val mAP50    — both runs

Usage
-----
    cd src && python plots/training_curves.py
    cd src && python plots/training_curves.py --smooth 5
    cd src && python plots/training_curves.py --style ppt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import globals as g
import style

DEFAULT_BASELINE = "yolov9s-aug-4"
DEFAULT_AAS      = "yolov9s-aas-25"

# Colors consistent with aug_comparison.py
COLOR_BASELINE = "#4C72B0"
COLOR_AAS      = "#9467BD"


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average smoothing."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _smooth(series: pd.Series, window: int) -> np.ndarray:
    arr = series.to_numpy(dtype=float)
    if window <= 1:
        return arr
    alpha = 2.0 / (window + 1)
    return _ema(arr, alpha)


def _load(run_name: str) -> pd.DataFrame:
    csv = g.RUNS_DIR / run_name / "results.csv"
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    return df


def _plot_loss(
    ax: plt.Axes,
    epochs: np.ndarray,
    train_baseline: np.ndarray,
    val_baseline: np.ndarray,
    train_aas: np.ndarray,
    val_aas: np.ndarray,
    title: str,
) -> None:
    ax.plot(epochs, train_baseline, color=COLOR_BASELINE,
            linestyle="-",  label="Baseline train")
    ax.plot(epochs, val_baseline,   color=COLOR_BASELINE,
            linestyle="--", label="Baseline val")
    ax.plot(epochs, train_aas,      color=COLOR_AAS,
            linestyle="-",  label="AAS train")
    ax.plot(epochs, val_aas,        color=COLOR_AAS,
            linestyle="--", label="AAS val")
    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.legend(fontsize="small")


def _plot_metric(
    ax: plt.Axes,
    epochs: np.ndarray,
    baseline: np.ndarray,
    aas: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    ax.plot(epochs, baseline, color=COLOR_BASELINE, label="Baseline")
    ax.plot(epochs, aas,      color=COLOR_AAS,      label="AAS")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="small")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline", default=DEFAULT_BASELINE,
        help=f"Baseline run name under runs/ (default: {DEFAULT_BASELINE})",
    )
    p.add_argument(
        "--aas", default=DEFAULT_AAS,
        help=f"AAS run name under runs/ (default: {DEFAULT_AAS})",
    )
    p.add_argument(
        "--smooth", type=int, default=5,
        help="EMA smoothing window in epochs (default: 5; use 1 to disable)",
    )
    p.add_argument(
        "--style", choices=style.STYLES, default=None,
        help="Output style: 'report' (PDF) or 'ppt' (PNG). "
             "Omit to produce both.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: results/training_curves_{style}.{ext})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df_b = _load(args.baseline)
    df_a = _load(args.aas)

    epochs_b = df_b["epoch"].to_numpy()
    epochs_a = df_a["epoch"].to_numpy()

    w = args.smooth

    data: List[Tuple] = [
        # (train_b, val_b, train_a, val_a, col, title, ylabel, is_loss)
        (
            _smooth(df_b["train/cls_loss"],     w),
            _smooth(df_b["val/cls_loss"],       w),
            _smooth(df_a["train/cls_loss"],     w),
            _smooth(df_a["val/cls_loss"],       w),
            "Class loss",
        ),
        (
            _smooth(df_b["train/box_loss"],     w),
            _smooth(df_b["val/box_loss"],       w),
            _smooth(df_a["train/box_loss"],     w),
            _smooth(df_a["val/box_loss"],       w),
            "Box loss",
        ),
    ]
    metrics: List[Tuple] = [
        (
            _smooth(df_b["metrics/mAP50-95(B)"], w),
            _smooth(df_a["metrics/mAP50-95(B)"], w),
            "Val mAP50-95",
            "mAP50-95",
        ),
        (
            _smooth(df_b["metrics/mAP50(B)"],    w),
            _smooth(df_a["metrics/mAP50(B)"],    w),
            "Val mAP50",
            "mAP50",
        ),
    ]

    styles_to_run = [args.style] if args.style else style.STYLES
    for s in styles_to_run:
        fmt = style.output_fmt(s)
        dpi = style.save_dpi(s)
        out = (
            args.out if args.out is not None
            else g.RESULTS_DIR / f"training_curves_{s}.{fmt}"
        )
        style.apply_style(s)

        PPT_WIDTH = 13.0
        fs = style.figsize(s, n_rows=2, n_cols=2)
        if s == style.PPT:
            font_scale = PPT_WIDTH / fs[0]
            plt.rcParams.update({
                k: round(plt.rcParams[k] * font_scale)
                for k in (
                    "font.size", "axes.titlesize", "axes.labelsize",
                    "legend.fontsize", "xtick.labelsize", "ytick.labelsize",
                )
            })
            fs = (PPT_WIDTH, fs[1] * font_scale)

        fig, axes = plt.subplots(2, 2, figsize=fs)

        for col, (train_b, val_b, train_a, val_a, title) in enumerate(data):
            _plot_loss(
                axes[0, col], epochs_b,
                train_b, val_b, train_a, val_a,
                title,
            )

        for col, (baseline, aas, title, ylabel) in enumerate(metrics):
            _plot_metric(
                axes[1, col], epochs_b,
                baseline, aas,
                title, ylabel,
            )

        for ax in axes[1, :]:
            ax.set_xlabel("Epoch")

        smooth_label = f"EMA-{w}" if w > 1 else "raw"
        fig.suptitle(
            f"Training dynamics — {args.baseline} vs. {args.aas}"
            f"  ({smooth_label})"
        )
        plt.tight_layout()

        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()


if __name__ == "__main__":
    main()
