"""Run inference on a dataset split and save per-image predictions to JSON.

The output JSON is keyed by image stem and consumed by view_data.py --run.

Usage
-----
python src/predict.py --run <run-name>
python src/predict.py --run <run-name> --weights epoch45.pt --split val
python src/predict.py --run <run-name> --conf 0.3 --imgsz 1920
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from ultralytics import YOLO

import globals as g

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEVICE: str = "0" if torch.cuda.is_available() else "cpu"


def pred_json_path(run: str, weights: str, split: str) -> Path:
    stem = Path(weights).stem
    return g.RESULTS_DIR / "predictions" / f"{run}_{stem}_{split}.json"


def run_predict(
    weights_path: Path,
    images: List[Path],
    imgsz: int,
    conf: float,
    batch: int,
    workers: int,
) -> Dict[str, Dict]:
    model = YOLO(str(weights_path))
    all_results = model.predict(
        source=[str(p) for p in images],
        imgsz=imgsz,
        conf=conf,
        batch=batch,
        workers=workers,
        device=DEVICE,
        stream=False,
        verbose=False,
    )

    predictions: Dict[str, Dict] = {}
    for img_path, result in zip(images, all_results):
        if result.obb is not None and len(result.obb):
            boxes = np.asarray(result.obb.xyxyxyxy).astype(int).tolist()
            confs = np.asarray(result.obb.conf).tolist()
        else:
            boxes, confs = [], []
        predictions[img_path.stem] = {"boxes": boxes, "confs": confs}

    return predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference on a split and save predictions to JSON"
    )
    p.add_argument(
        "--run", type=str, required=True,
        help="Run directory name under runs/, e.g. yolov9s-aas-12",
    )
    p.add_argument(
        "--weights", type=str, default="best.pt",
        help="Weights filename under <run>/weights/ (default: best.pt)",
    )
    p.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Dataset split to run inference on (default: test)",
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    p.add_argument(
        "--imgsz", type=int, default=1920,
        help="Inference image size; should match training (default: 1920)",
    )
    p.add_argument(
        "--batch", type=int, default=8,
        help="Prediction batch size (default: 8)",
    )
    p.add_argument(
        "--workers", type=int, default=16,
        help="Dataloader workers (default: 16)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = g.RUNS_DIR / args.run / "weights" / args.weights
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    images = sorted((g.IMG_DIR / args.split).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"No images found in {g.IMG_DIR / args.split}"
        )

    out_path = pred_json_path(args.run, args.weights, args.split)

    print(f"Weights:  {weights_path}")
    print(f"Split:    {args.split}  ({len(images)} images)")
    print(f"Device:   {DEVICE}")
    print(f"Output:   {out_path}")
    print("Running inference...")

    predictions = run_predict(
        weights_path, images, args.imgsz, args.conf, args.batch, args.workers,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, indent=2))
    print(f"Saved {len(predictions)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
