"""Metrics bar chart: compare runs and weights side-by-side.

Each of the four metrics (Precision, Recall, mAP50, mAP50-95) gets a group
of bars — one bar per specified (run, weights) pair.

Usage
-----
    cd src && python plots/metrics_bar.py \\
        --run-name yolov9s-aug-4  --weights epoch80.pt \\
        --run-name yolov9s-aas-25 --weights epoch35.pt
    cd src && python plots/metrics_bar.py \\
        --run-name yolov9s-aug-4  --weights epoch80.pt \\
        --run-name yolov9s-aas-25 --weights epoch35.pt \\
        --label Baseline --label AAS \\
        --style ppt
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

EVAL_CSV = g.RESULTS_DIR / "from_remote" / "evaluations.csv"

METRICS: List[Tuple[str, str]] = [
    ("precision", "Precision"),
    ("recall",    "Recall"),
    ("mAP50",     "mAP50"),
    ("mAP50-95",  "mAP50-95"),
]

PALETTE: List[str] = [
    "#4C72B0",
    "#9467BD",
    "#DD8452",
    "#55A868",
    "#C44E52",
]

PPT_FIGSIZE = (9.0, 5.0)


def _load_row(
    df: pd.DataFrame,
    run_name: str,
    weights: str,
) -> pd.Series:
    mask = (df["run_name"] == run_name) & (df["weights"] == weights)
    matches = df[mask]
    if matches.empty:
        raise SystemExit(
            f"No row found for run_name={run_name!r}, weights={weights!r}"
        )
    if len(matches) > 1:
        print(
            f"Warning: {len(matches)} rows match {run_name}/{weights}; "
            "using the last one"
        )
    return matches.iloc[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-name",
        dest="run_names",
        action="append",
        default=[],
        metavar="NAME",
        help="Run name (repeatable; must pair with a --weights)",
    )
    p.add_argument(
        "--weights",
        dest="weights_list",
        action="append",
        default=[],
        metavar="WEIGHTS",
        help="Weights file (repeatable; must pair with a --run-name)",
    )
    p.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=[],
        metavar="LABEL",
        help="Legend label for each run (optional; default: 'run weights')",
    )
    p.add_argument(
        "--style", choices=style.STYLES, default=None,
        help="Output style: 'report' (PDF) or 'ppt' (PNG). Omit for both.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: results/metrics_bar_{style}.{ext})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.run_names:
        raise SystemExit(
            "Provide at least one --run-name / --weights pair.\n"
            "Example: --run-name yolov9s-aug-4 --weights epoch80.pt"
        )
    if len(args.run_names) != len(args.weights_list):
        raise SystemExit(
            f"Got {len(args.run_names)} --run-name(s) but "
            f"{len(args.weights_list)} --weights; counts must match."
        )

    df = pd.read_csv(EVAL_CSV)
    df.columns = df.columns.str.strip()

    rows = [
        _load_row(df, rn, w)
        for rn, w in zip(args.run_names, args.weights_list)
    ]

    labels: List[str] = []
    for i, (rn, w) in enumerate(zip(args.run_names, args.weights_list)):
        if i < len(args.labels):
            labels.append(args.labels[i])
        else:
            labels.append(f"{rn} / {w}")

    metric_keys   = [m[0] for m in METRICS]
    metric_labels = [m[1] for m in METRICS]
    n_metrics = len(METRICS)
    n_runs    = len(rows)

    values = np.array(
        [[float(row[k]) for k in metric_keys] for row in rows],
        dtype=float,
    )  # shape: (n_runs, n_metrics)

    bar_width = 0.7 / n_runs
    x = np.arange(n_metrics, dtype=float)
    offsets = (np.arange(n_runs) - (n_runs - 1) / 2.0) * bar_width

    styles_to_run = [args.style] if args.style else style.STYLES
    for s in styles_to_run:
        fmt = style.output_fmt(s)
        dpi = style.save_dpi(s)
        out = (
            args.out if args.out is not None
            else g.RESULTS_DIR / f"metrics_bar_{s}.{fmt}"
        )
        style.apply_style(s)

        fs = PPT_FIGSIZE if s == style.PPT else style.figsize(s)
        fig, ax = plt.subplots(figsize=fs)

        for i, (run_vals, label) in enumerate(zip(values, labels)):
            color = PALETTE[i % len(PALETTE)]
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


if __name__ == "__main__":
    main()
