"""Grouped bar chart of evaluation metrics for one or more runs.

Generates:
  metrics_bar_{style}.{ext}  — grouped bar chart (P / R / mAP50 / mAP50-95)

Reads per-checkpoint metrics from results/evaluations.csv.

Usage
-----
    cd src && python plots/metrics_bar.py \\
        --run-name yolov9s-aug-4  --weights epoch100.pt \\
        --run-name yolov9s-aas-31 --weights epoch36.pt \\
        --run-name yolov9s-aas-34 --weights epoch29.pt \\
        --label Baseline --label AAS --label "AAS + MixUp 0.3" --style ppt
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

# ── constants ──────────────────────────────────────────────────────────────

EVAL_CSV = g.RESULTS_DIR / "evaluations.csv"

PALETTE: List[str] = [
    "#4C72B0",
    "#9467BD",
    "#2E7D32",
    "#55A868",
    "#C44E52",
]

PPT_FIGSIZE_BAR = (9.0, 5.0)

BAR_METRICS: List[Tuple[str, str]] = [
    ("precision", "Precision"),
    ("recall",    "Recall"),
    ("mAP50",     "mAP50"),
    ("mAP50-95",  "mAP50-95"),
]

# (label, color)
EntryData = Tuple[str, str]


# ── helpers ────────────────────────────────────────────────────────────────

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


# ── figure generator ───────────────────────────────────────────────────────

def _generate(
    entries: List[EntryData],
    eval_rows: List[pd.Series],
    styles_to_run: List[str],
) -> None:
    metric_keys   = [m[0] for m in BAR_METRICS]
    metric_labels = [m[1] for m in BAR_METRICS]
    n_metrics = len(BAR_METRICS)
    n_runs    = len(entries)

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

        for i, (label, color) in enumerate(entries):
            bars = ax.bar(
                x + offsets[i], values[i],
                width=bar_width,
                color=color,
                label=label,
                zorder=3,
            )
            for bar, val in zip(bars, values[i]):
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
    p = argparse.ArgumentParser(
        description="Grouped bar chart of evaluation metrics for one or more runs."
    )
    p.add_argument(
        "--run-name", dest="run_names", action="append", default=[],
        metavar="NAME", required=True,
        help="Run name under runs/ (repeatable).",
    )
    p.add_argument(
        "--weights", dest="weights_list", action="append", default=[],
        metavar="WEIGHTS", required=True,
        help="Weights file for each run (repeatable; must match --run-name count).",
    )
    p.add_argument(
        "--label", dest="labels", action="append", default=[],
        metavar="LABEL",
        help="Legend label for each run (defaults to run name).",
    )
    p.add_argument(
        "--style", choices=style.STYLES, default=None,
        help="Output style: 'report' (PDF) or 'ppt' (PNG). Omit for both.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.weights_list) != len(args.run_names):
        raise SystemExit(
            f"Got {len(args.run_names)} --run-name(s) but "
            f"{len(args.weights_list)} --weights; counts must match."
        )

    if not EVAL_CSV.exists():
        raise SystemExit(f"Evaluations CSV not found: {EVAL_CSV}")

    eval_df = pd.read_csv(EVAL_CSV)
    eval_df.columns = eval_df.columns.str.strip()

    entries: List[EntryData] = []
    eval_rows: List[pd.Series] = []
    for i, (rn, w) in enumerate(zip(args.run_names, args.weights_list)):
        label = args.labels[i] if i < len(args.labels) else rn
        color = PALETTE[i % len(PALETTE)]
        entries.append((label, color))
        eval_rows.append(_load_eval_row(eval_df, rn, w))

    styles_to_run = [args.style] if args.style else style.STYLES
    _generate(entries, eval_rows, styles_to_run)


if __name__ == "__main__":
    main()
