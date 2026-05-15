"""Training-curve comparison: baseline vs. AAS.

2×2 panel layout:
  (1,1) Class loss   — train (solid) + val (dashed), both runs
  (1,2) Box loss     — train (solid) + val (dashed), both runs
  (2,1) Val mAP50-95 — both runs
  (2,2) Val mAP50    — both runs

Raw curves are shown at low opacity behind the smoothed curves.

Usage
-----
    cd src && python plots/training_curves.py
    cd src && python plots/training_curves.py --smooth 15
    cd src && python plots/training_curves.py --style ppt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import globals as g
import style

DEFAULT_BASELINE = "yolov9s-aug-4"
DEFAULT_AAS      = "yolov9s-aas-25"

RAW_ALPHA = 0.20

# Colors consistent with aug_comparison.py
COLOR_BASELINE = "#4C72B0"
COLOR_AAS      = "#9467BD"

# (width, height) in inches for the PPT 2×2 figure
PPT_FIGSIZE = (13.0, 6.5)


def _smooth(series: pd.Series, window: int) -> np.ndarray:
    return (
        series.rolling(window, min_periods=1).mean().to_numpy(dtype=float)
    )


def _load(run_name: str) -> pd.DataFrame:
    csv = g.RUNS_DIR / run_name / "results.csv"
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    return df


def _plot_loss(
    ax: plt.Axes,
    epochs: np.ndarray,
    raw_train_b: np.ndarray,
    raw_val_b: np.ndarray,
    raw_train_a: np.ndarray,
    raw_val_a: np.ndarray,
    sm_val_b: np.ndarray,
    sm_val_a: np.ndarray,
    title: str,
) -> None:
    # val: raw background + smoothed foreground
    ax.plot(epochs, raw_val_b,  color=COLOR_BASELINE,
            linestyle="--", alpha=RAW_ALPHA)
    ax.plot(epochs, raw_val_a,  color=COLOR_AAS,
            linestyle="--", alpha=RAW_ALPHA)
    # train: raw only (already smooth)
    ax.plot(epochs, raw_train_b, color=COLOR_BASELINE,
            linestyle="-",  label="Baseline train")
    ax.plot(epochs, raw_train_a, color=COLOR_AAS,
            linestyle="-",  label="AAS train")
    # smoothed val foreground
    ax.plot(epochs, sm_val_b,   color=COLOR_BASELINE,
            linestyle="--", label="Baseline val")
    ax.plot(epochs, sm_val_a,   color=COLOR_AAS,
            linestyle="--", label="AAS val")
    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.legend(fontsize="small")


def _plot_metric(
    ax: plt.Axes,
    epochs: np.ndarray,
    raw_b: np.ndarray,
    raw_a: np.ndarray,
    sm_b: np.ndarray,
    sm_a: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    ax.plot(epochs, raw_b, color=COLOR_BASELINE, alpha=RAW_ALPHA)
    ax.plot(epochs, raw_a, color=COLOR_AAS,      alpha=RAW_ALPHA)
    ax.plot(epochs, sm_b,  color=COLOR_BASELINE, label="Baseline")
    ax.plot(epochs, sm_a,  color=COLOR_AAS,      label="AAS")
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
        "--smooth", type=int, default=10,
        help="Rolling-average window in epochs (default: 10; use 1 to disable)",
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

    epochs = df_b["epoch"].to_numpy()
    w = args.smooth

    loss_cols: List[Tuple[str, str, str]] = [
        ("train/cls_loss", "val/cls_loss", "Class loss"),
        ("train/box_loss", "val/box_loss", "Box loss"),
    ]
    metric_cols: List[Tuple[str, str, str]] = [
        ("metrics/mAP50(B)",    "Val mAP50",    "mAP50"),
        ("metrics/mAP50-95(B)", "Val mAP50-95", "mAP50-95"),
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

        if s == style.PPT:
            fs = PPT_FIGSIZE
        else:
            fs = style.figsize(s, n_rows=2, n_cols=2)

        fig, axes = plt.subplots(2, 2, figsize=fs)

        for col, (train_col, val_col, title) in enumerate(loss_cols):
            _plot_loss(
                axes[0, col], epochs,
                df_b[train_col].to_numpy(dtype=float),
                df_b[val_col].to_numpy(dtype=float),
                df_a[train_col].to_numpy(dtype=float),
                df_a[val_col].to_numpy(dtype=float),
                _smooth(df_b[val_col], w),
                _smooth(df_a[val_col], w),
                title,
            )

        for col, (metric_col, title, ylabel) in enumerate(metric_cols):
            _plot_metric(
                axes[1, col], epochs,
                df_b[metric_col].to_numpy(dtype=float),
                df_a[metric_col].to_numpy(dtype=float),
                _smooth(df_b[metric_col], w),
                _smooth(df_a[metric_col], w),
                title, ylabel,
            )

        for ax in axes[1, :]:
            ax.set_xlabel("Epoch")

        fig.suptitle("Training dynamics — Baseline vs. AAS")
        plt.tight_layout()

        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()


if __name__ == "__main__":
    main()
