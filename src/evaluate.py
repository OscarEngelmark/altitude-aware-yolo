"""
Evaluate a trained YOLOv9-OBB checkpoint on the test or val split.

Prints overall and per-bucket metrics, then saves a 2×2 bar-chart PNG to
results/<run-name>.png.

Usage
-----
python src/evaluate.py --run test-run --weights best.pt
python src/evaluate.py --run test-run --weights best.pt epoch45.pt
python src/evaluate.py --run test-run --all-weights
python src/evaluate.py --run test-run --weights best.pt --split val
python src/evaluate.py --run test-run --weights best.pt --run-name my-eval
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

import globals as g
from callbacks import get_last_bucket_metrics, register_metadata_callbacks
from train import write_dataset_yaml

# Set PyTorch CUDA allocator to allow fragmentation (prevents GPU OOM errors)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEVICE: str = "0" if torch.cuda.is_available() else "cpu"

METRICS = [
    ("Precision",  "precision"),
    ("Recall",     "recall"),
    ("mAP50",      "mAP50"),
    ("mAP50-95",   "mAP50-95"),
]


CSV_PATH = g.RESULTS_DIR / "evaluations.csv"
CSV_FIELDS = ["timestamp", "run_name", "weights", "split", "precision",
              "recall", "mAP50", "mAP50-95"]


def save_metrics_csv(
    weights: Path,
    overall: Dict[str, float],
    split: str,
) -> None:
    g.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_name":   weights.parent.parent.name,
            "weights":    weights.name,
            "split":      split,
            "precision":  f"{overall['precision']:.4f}",
            "recall":     f"{overall['recall']:.4f}",
            "mAP50":      f"{overall['mAP50']:.4f}",
            "mAP50-95":   f"{overall['mAP50-95']:.4f}",
        })


def plot_metrics(
    overall: Dict[str, float],
    bucket_metrics: Dict[str, float],
    run_name: str,
    split: str = "test",
) -> Path:
    """Save a 2x2 grid of bar charts — one per metric — to RESULTS_DIR."""
    bucket_labels = [label for label, *_ in g.ALTITUDE_BUCKETS]
    prefix = "val_alt"

    n_cars_overall = sum(
        int(bucket_metrics.get(f"{prefix}/{b}/n_targets", 0))
        for b in bucket_labels
    )
    x_labels = [f"Overall\n({n_cars_overall} cars)"]
    for b in bucket_labels:
        n = int(bucket_metrics.get(f"{prefix}/{b}/n_targets", 0))
        x_labels.append(f"{b}\n({n} cars)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{split.capitalize()} metrics — {run_name}", fontsize=14)

    for ax, (title, key) in zip(axes.flat, METRICS):
        values = [overall[key]]
        for bucket in bucket_labels:
            v = bucket_metrics.get(f"{prefix}/{bucket}/{key}")
            values.append(v if v is not None else 0.0)

        bars = ax.bar(x_labels, values)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)

        for bar, val in zip(bars, values):
            inside = val > 0.88
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.03 if inside else bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                va="top" if inside else "bottom",
                fontsize=8,
                color="white" if inside else "black",
            )

    fig.tight_layout()
    g.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = g.RESULTS_DIR / f"{run_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a YOLOv9-OBB checkpoint on the test split"
    )
    p.add_argument(
        "--run", type=str, required=True,
        help="run directory name under runs/, e.g. yolov9s-aas-12",
    )
    wt = p.add_mutually_exclusive_group(required=True)
    wt.add_argument(
        "--weights", type=str, nargs="+",
        help="one or more filenames under <run>/weights/, e.g. best.pt "
             "epoch45.pt",
    )
    wt.add_argument(
        "--all-weights", action="store_true",
        help="evaluate every .pt file found in <run>/weights/",
    )
    p.add_argument(
        "--run-name", type=str, default=None,
        help="name used for the output plot; only applies when a single "
             "weights file is given (defaults to 'eval-<stem>-<run>')",
    )
    p.add_argument(
        "--imgsz", type=int, default=1920,
        help="inference image size (should match training)",
    )
    p.add_argument(
        "--batch", type=int, default=8,
    )
    p.add_argument(
        "--workers", type=int, default=16,
    )
    p.add_argument(
        "--split", type=str, default="test", choices=["test", "val"],
        help="dataset split to evaluate on (default: test)",
    )
    return p.parse_args()


def evaluate_checkpoint(
    weights_path: Path,
    run_name: str,
    dataset_yaml: str,
    split: str,
    imgsz: int,
    batch: int,
    workers: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"Weights:  {weights_path}")
    print(f"Run name: {run_name}")

    model = YOLO(str(weights_path))
    register_metadata_callbacks(model, training=False)

    results = model.val(
        data=dataset_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=DEVICE,
        project=str(g.RUNS_DIR),
        name=run_name,
    )

    box = results.box
    overall = {
        "precision": float(box.mp),
        "recall":    float(box.mr),
        "mAP50":     float(box.map50),
        "mAP50-95":  float(box.map),
    }

    s = split.capitalize()
    print(f"\n{s} mAP50:      {overall['mAP50']:.4f}")
    print(f"{s} mAP50-95:   {overall['mAP50-95']:.4f}")
    print(f"{s} precision:  {overall['precision']:.4f}")
    print(f"{s} recall:     {overall['recall']:.4f}")
    out = plot_metrics(overall, get_last_bucket_metrics(), run_name, split)
    print(f"Plot saved to:  {out}")
    save_metrics_csv(weights_path, overall, split)
    print(f"Metrics saved to: {CSV_PATH}")


def main() -> None:
    args = parse_args()

    weights_dir = g.RUNS_DIR / args.run / "weights"
    if args.all_weights:
        weights_paths = sorted(weights_dir.glob("*.pt"))
        if not weights_paths:
            raise FileNotFoundError(
                f"No .pt files found in {weights_dir}"
            )
    else:
        weights_paths = []
        for wf in args.weights:
            p = weights_dir / wf
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            weights_paths.append(p)

    dataset_yaml = write_dataset_yaml()
    print(f"Dataset:  {dataset_yaml}")
    print(f"Device:   {DEVICE}")
    print(
        f"Evaluating {len(weights_paths)} checkpoint(s) from run '{args.run}'"
    )

    single = len(weights_paths) == 1
    for weights_path in weights_paths:
        run_name = (
            args.run_name
            if single and args.run_name
            else f"eval-{weights_path.stem}-{args.run}"
        )
        evaluate_checkpoint(
            weights_path, run_name, dataset_yaml,
            args.split, args.imgsz, args.batch, args.workers,
        )


if __name__ == "__main__":
    main()
