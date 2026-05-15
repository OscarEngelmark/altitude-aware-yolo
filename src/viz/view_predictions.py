"""Visualize model predictions overlaid on frames from a saved JSON file.

Run predict.py first to generate the predictions JSON, then use this script
to browse results instantly without a GPU.

Usage
-----
    cd src && python viz/view_predictions.py --run yolov9s-aas-12 --split test
    cd src && python viz/view_predictions.py --run yolov9s-aas-12 --split test --show-gt
    cd src && python viz/view_predictions.py --run yolov9s-aas-12 --weights epoch45.pt

Controls: any key -> next  |  p/← -> prev  |  s -> save  |  q -> quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import globals as g
from viz.utils import (
    _WINDOW_NAME,
    _unique_path,
    collect_images,
    draw_corners,
    load_metadata,
    load_obb_corners,
    overlay_lines,
    resize_for_display,
)


def _render(
    img_path: Path,
    pred: Dict[str, Any],
    metadata: Dict[str, float],
    lbl_dir: Path,
    show_gt: bool,
    max_dim: Optional[int],
) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if show_gt:
        gt_corners = load_obb_corners(
            lbl_dir / img_path.with_suffix(".txt").name, w, h
        )
        draw_corners(img, gt_corners, color=(0, 255, 0))

    boxes = pred.get("boxes", [])
    confs_list = pred.get("confs", [])
    n_pred = len(boxes)
    conf_str = ""
    if n_pred:
        pred_corners = np.array(boxes, dtype=np.int32)  # (N, 4, 2)
        confs = np.array(confs_list)                    # (N,)
        conf_str = f"conf {confs.min():.2f}-{confs.max():.2f}"
        draw_corners(img, pred_corners, color=(0, 0, 255))
        for corners, conf in zip(pred_corners, confs):
            cx = int(corners[:, 0].mean())
            cy = max(int(corners[:, 1].min()) - 4, 12)
            label = f"{conf:.2f}"
            cv2.putText(
                img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 200, 255), 1, cv2.LINE_AA,
            )

    alt = metadata.get(img_path.stem)
    alt_str = f"{alt:.0f} m" if alt is not None else "unknown"
    pred_info = f"Pred: {n_pred} boxes" + (f"  {conf_str}" if conf_str else "")
    info: List[str] = [img_path.name, f"Alt: {alt_str}  |  {pred_info}"]
    if show_gt:
        info.append("Green = GT  |  Red = Pred")
    overlay_lines(img, info)
    return resize_for_display(img, max_dim)


def _run_viewer(
    img_paths: List[Path],
    predictions: Dict[str, Dict],
    metadata: Dict[str, float],
    lbl_dir: Path,
    show_gt: bool,
    max_dim: Optional[int],
    save_dir: Path,
) -> None:
    n = len(img_paths)
    idx = 0
    while True:
        pred = predictions.get(img_paths[idx].stem, {"boxes": [], "confs": []})
        frame = _render(
            img_paths[idx], pred, metadata, lbl_dir, show_gt, max_dim,
        )
        if frame is None:
            print(f"Could not read {img_paths[idx].name}, skipping.")
            idx = min(idx + 1, n - 1)
            continue

        cv2.imshow(_WINDOW_NAME, frame)
        cv2.resizeWindow(_WINDOW_NAME, frame.shape[1], frame.shape[0])
        cv2.setWindowTitle(
            _WINDOW_NAME, f"[{idx + 1}/{n}]  {img_paths[idx].name}"
        )

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_dir.mkdir(parents=True, exist_ok=True)
            out = _unique_path(save_dir / f"{img_paths[idx].stem}_pred.jpg")
            cv2.imwrite(str(out), frame)
            print(f"Saved {out}")
        elif key in (ord("p"), 81):
            idx = max(0, idx - 1)
        else:
            idx = min(n - 1, idx + 1)


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse model predictions loaded from a JSON file."
    )
    parser.add_argument(
        "--run", type=str, required=True,
        help="Run directory name under runs/ (e.g. yolov9s-aas-12)",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Dataset split to visualize (default: test)",
    )
    parser.add_argument(
        "--weights", type=str, default="best.pt",
        help="Weights filename used to generate predictions JSON (default: best.pt)",
    )
    parser.add_argument(
        "--show-gt", action="store_true", dest="show_gt",
        help="Overlay ground-truth OBBs in green alongside predictions",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Filter by source filename substring (e.g. 'Nyland')",
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Maximum number of images to show",
    )
    parser.add_argument(
        "--max-dim", type=int, default=None, dest="max_dim",
        help="Resize display so the longest side is at most this many pixels",
    )
    return parser.parse_args()


def main() -> None:
    opt = parse_opt()
    img_dir = g.IMG_DIR / opt.split
    lbl_dir = g.LBL_DIR / opt.split

    if not img_dir.is_dir():
        sys.exit(f"No images found at {img_dir} — run preprocessing.py first.")

    json_path = (
        g.RESULTS_DIR / "predictions"
        / f"{opt.run}_{Path(opt.weights).stem}_{opt.split}.json"
    )
    if not json_path.exists():
        sys.exit(
            f"Predictions not found: {json_path}\n"
            f"Run:  python predict.py --run {opt.run} "
            f"--weights {opt.weights} --split {opt.split}"
        )

    with json_path.open() as f:
        predictions: Dict[str, Dict] = json.load(f)

    images = collect_images(img_dir, opt.source, opt.max)
    if not images:
        sys.exit("No matching images found.")

    metadata = load_metadata(g.OUT_DIR / "metadata.json")
    save_dir = g.RESULTS_DIR / "predictions" / opt.run

    print(f"Predictions: {json_path}")
    print(f"Split:       {opt.split}  ({len(images)} images)")
    print(
        "  any key = next  |  p/← = prev  |  s = save  |  q = quit\n"
        f"  Saves go to: {save_dir}"
    )

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    _run_viewer(
        images, predictions, metadata,
        lbl_dir, opt.show_gt, opt.max_dim, save_dir,
    )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
