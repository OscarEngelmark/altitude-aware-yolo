"""Results plots: training curves + metrics bar chart.

Generates up to two figures per style:
  training_curves_{style}.{ext}  — 2×2 training curve comparison
  metrics_bar_{style}.{ext}      — grouped bar chart (only when --weights given)

The same (run, weights, label) triples drive both figures: weights determine
the star marker position on the curves and which evaluation row to pull for
the bar chart.

Usage
-----
    cd src && python plots/plot_results.py \\
        --run-name yolov9s-aug-4  --weights epoch80.pt \\
        --run-name yolov9s-aas-25 --weights epoch35.pt \\
        --label Baseline --label AAS --style ppt

    # Training curves only (no weights → no bar chart, no markers)
    cd src && python plots/plot_results.py --style ppt
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

import globals as g
import style

# ── constants ──────────────────────────────────────────────────────────────

EVAL_CSV = g.RESULTS_DIR / "evaluations.csv"

DEFAULT_BASELINE = "yolov9s-aug-4"
DEFAULT_AAS      = "yolov9s-aas-25"

RAW_ALPHA            = 0.20
MARKER_SIZE_PPT      = 180
MARKER_SIZE_REPORT   = 60

PALETTE: List[str] = [
    "#4C72B0",
    "#9467BD",
    "#DD8452",
    "#55A868",
    "#C44E52",
]

PPT_FIGSIZE_CURVES    = (13.0, 6.5)
REPORT_FIGSIZE_CURVES = (3.3,  6.0)  # 4×1 single IEEE column
PPT_FIGSIZE_BAR       = (9.0,  5.0)

BAR_METRICS: List[Tuple[str, str]] = [
    ("precision", "Precision"),
    ("recall",    "Recall"),
    ("mAP50",     "mAP50"),
    ("mAP50-95",  "mAP50-95"),
]

# (training_df, label, color, marker_epoch | None)
RunData = Tuple[pd.DataFrame, str, str, Optional[int]]


# ── shared helpers ─────────────────────────────────────────────────────────

def _smooth(series: pd.Series, window: int) -> np.ndarray:
    return (
        series.rolling(window, min_periods=1).mean().to_numpy(dtype=float)
    )


def _load(run_name: str) -> pd.DataFrame:
    csv = g.RUNS_DIR / run_name / "results.csv"
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    return df


def _load_eval_row(
    eval_df: pd.DataFrame,
    run_name: str,
    weights: str,
) -> pd.Series:
    mask = (eval_df["run_name"] == run_name) & (eval_df["weights"] == weights)
    matches = eval_df[mask]
    if matches.empty:
        raise SystemExit(
            f"No evaluation row found for run_name={run_name!r},"
            f" weights={weights!r}"
        )
    if len(matches) > 1:
        print(
            f"Warning: {len(matches)} rows match {run_name}/{weights};"
            " using the last one"
        )
    return matches.iloc[-1]


def _parse_epoch(weights: str, df: pd.DataFrame) -> Optional[int]:
    """Return the epoch number for a weights filename.

    Supports epochN.pt, last.pt, and best.pt (Ultralytics fitness).
    """
    stem = Path(weights).stem
    if stem == "last":
        return int(df["epoch"].iloc[-1])
    if stem == "best":
        fitness = (
            0.1 * df["metrics/mAP50(B)"]
            + 0.9 * df["metrics/mAP50-95(B)"]
        )
        return int(df["epoch"].iloc[fitness.argmax()])
    m = re.match(r"epoch(\d+)", stem)
    if m:
        return int(m.group(1)) + 1  # checkpoints are 0-indexed; results.csv is 1-indexed
    return None


def _epoch_idx(epochs: np.ndarray, epoch: int) -> int:
    idx = np.searchsorted(epochs, epoch)
    return min(int(idx), len(epochs) - 1)


# ── training curve helpers ─────────────────────────────────────────────────

def _plot_loss(
    ax: Axes,
    runs: List[RunData],
    train_col: str,
    val_col: str,
    title: str,
    window: int,
    marker_size: int,
) -> None:
    for df, label, color, marker_epoch in runs:
        epochs    = df["epoch"].to_numpy()
        raw_train = df[train_col].to_numpy(dtype=float)
        raw_val   = df[val_col].to_numpy(dtype=float)
        sm_val    = _smooth(df[val_col], window)

        ax.plot(epochs, raw_val,   color=color, linestyle="--", alpha=RAW_ALPHA)
        ax.plot(epochs, raw_train, color=color, linestyle="-",
                label=f"{label} train")
        ax.plot(epochs, sm_val,    color=color, linestyle="--",
                label=f"{label} val")

        if marker_epoch is not None:
            idx = _epoch_idx(epochs, marker_epoch)
            ax.scatter(
                [epochs[idx]], [raw_val[idx]],
                marker="*", s=marker_size, color=color, zorder=6,
            )

    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.legend(fontsize="small")


def _plot_metric(
    ax: Axes,
    runs: List[RunData],
    metric_col: str,
    title: str,
    ylabel: str,
    window: int,
    marker_size: int,
) -> None:
    for df, label, color, marker_epoch in runs:
        epochs = df["epoch"].to_numpy()
        raw    = df[metric_col].to_numpy(dtype=float)
        sm     = _smooth(df[metric_col], window)

        ax.plot(epochs, raw, color=color, alpha=RAW_ALPHA)
        ax.plot(epochs, sm,  color=color, label=label)

        if marker_epoch is not None:
            idx = _epoch_idx(epochs, marker_epoch)
            ax.scatter(
                [epochs[idx]], [raw[idx]],
                marker="*", s=marker_size, color=color, zorder=6,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="small")


# ── figure generators ──────────────────────────────────────────────────────

def _generate_training_curves(
    runs: List[RunData],
    smooth: int,
    styles_to_run: List[str],
) -> None:
    loss_cols: List[Tuple[str, str, str]] = [
        ("train/cls_loss", "val/cls_loss", "Class loss"),
        ("train/box_loss", "val/box_loss", "Box loss"),
    ]
    metric_cols: List[Tuple[str, str, str]] = [
        ("metrics/mAP50(B)",    "Val mAP50",    "mAP50"),
        ("metrics/mAP50-95(B)", "Val mAP50-95", "mAP50-95"),
    ]

    for s in styles_to_run:
        fmt = style.output_fmt(s)
        dpi = style.save_dpi(s)
        out = g.RESULTS_DIR / f"training_curves_{s}.{fmt}"
        style.apply_style(s)

        if s == style.PPT:
            ms = MARKER_SIZE_PPT
            fig, axes = plt.subplots(2, 2, figsize=PPT_FIGSIZE_CURVES)
            for col, (train_col, val_col, title) in enumerate(loss_cols):
                _plot_loss(axes[0, col], runs, train_col, val_col, title, smooth, ms)
            for col, (metric_col, title, ylabel) in enumerate(metric_cols):
                _plot_metric(axes[1, col], runs, metric_col, title, ylabel, smooth, ms)
            for ax in axes[1, :]:
                ax.set_xlabel("Epoch")
        else:
            ms = MARKER_SIZE_REPORT
            plt.rcParams.update({
                "font.size": 7,
                "axes.titlesize": 7,
                "axes.labelsize": 7,
                "legend.fontsize": 6,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
            })
            fig, axs = plt.subplots(4, 1, figsize=REPORT_FIGSIZE_CURVES, sharex=True)
            for i, (train_col, val_col, title) in enumerate(loss_cols):
                _plot_loss(axs[i], runs, train_col, val_col, title, smooth, ms)
            for i, (metric_col, title, ylabel) in enumerate(metric_cols):
                _plot_metric(axs[2 + i], runs, metric_col, title, ylabel, smooth, ms)
            axs[-1].set_xlabel("Epoch")

        plt.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()


def _generate_metrics_bar(
    runs: List[RunData],
    eval_rows: List[pd.Series],
    styles_to_run: List[str],
) -> None:
    metric_keys   = [m[0] for m in BAR_METRICS]
    metric_labels = [m[1] for m in BAR_METRICS]
    n_metrics = len(BAR_METRICS)
    n_runs    = len(runs)

    values = np.array(
        [[float(row[k]) for k in metric_keys] for row in eval_rows],
        dtype=float,
    )  # shape: (n_runs, n_metrics)

    bar_width = 0.7 / n_runs
    x = np.arange(n_metrics, dtype=float)
    offsets = (np.arange(n_runs) - (n_runs - 1) / 2.0) * bar_width

    for s in styles_to_run:
        fmt = style.output_fmt(s)
        dpi = style.save_dpi(s)
        out = g.RESULTS_DIR / f"metrics_bar_{s}.{fmt}"
        style.apply_style(s)

        fs = PPT_FIGSIZE_BAR if s == style.PPT else style.figsize(s)
        fig, ax = plt.subplots(figsize=fs)

        for i, run_vals in enumerate(values):
            _df, label, color, _epoch = runs[i]
            bars = ax.bar(
                x + offsets[i], run_vals,
                width=bar_width,
                color=color,
                label=label,
                zorder=3,
            )
            for bar, val in zip(bars, run_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.008,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=plt.rcParams.get("xtick.labelsize", 7),
                    color=color,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, zorder=0, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend()

        plt.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out}")
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-name",
        dest="run_names",
        action="append",
        default=[],
        metavar="NAME",
        help="Run name under runs/ (repeatable). "
             f"Defaults to {DEFAULT_BASELINE} + {DEFAULT_AAS}.",
    )
    p.add_argument(
        "--weights",
        dest="weights_list",
        action="append",
        default=[],
        metavar="WEIGHTS",
        help=(
            "Weights file (repeatable; must match --run-name count if given). "
            "Adds a star marker to curves and generates the bar chart. "
            "Supports epochN.pt / last.pt / best.pt."
        ),
    )
    p.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=[],
        metavar="LABEL",
        help="Legend label for each run (optional; defaults to run name)",
    )
    p.add_argument(
        "--smooth", type=int, default=10,
        help="Rolling-average window in epochs (default: 10; use 1 to disable)",
    )
    p.add_argument(
        "--style", choices=style.STYLES, default=None,
        help="Output style: 'report' (PDF) or 'ppt' (PNG). Omit for both.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_names = args.run_names or [DEFAULT_BASELINE, DEFAULT_AAS]

    if args.weights_list and len(args.weights_list) != len(run_names):
        raise SystemExit(
            f"Got {len(run_names)} run(s) but {len(args.weights_list)} "
            "--weights; counts must match (or omit --weights entirely)."
        )

    runs: List[RunData] = []
    for i, rn in enumerate(run_names):
        df           = _load(rn)
        label        = args.labels[i] if i < len(args.labels) else rn
        color        = PALETTE[i % len(PALETTE)]
        weights      = args.weights_list[i] if args.weights_list else None
        marker_epoch = _parse_epoch(weights, df) if weights else None
        runs.append((df, label, color, marker_epoch))

    styles_to_run = [args.style] if args.style else style.STYLES

    _generate_training_curves(runs, args.smooth, styles_to_run)

    if args.weights_list:
        if not EVAL_CSV.exists():
            print(f"Warning: {EVAL_CSV} not found — skipping bar chart.")
        else:
            eval_df = pd.read_csv(EVAL_CSV)
            eval_df.columns = eval_df.columns.str.strip()
            eval_rows = [
                _load_eval_row(eval_df, rn, w)
                for rn, w in zip(run_names, args.weights_list)
            ]
            _generate_metrics_bar(runs, eval_rows, styles_to_run)


if __name__ == "__main__":
    main()
