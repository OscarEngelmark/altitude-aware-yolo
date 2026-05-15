"""Visualize processed frames with ground-truth OBB labels.

Usage
-----
    cd src && python viz/view_raw.py
    cd src && python viz/view_raw.py --split val --source Nyland
    cd src && python viz/view_raw.py --split train --max 200 --max-dim 1280

Controls: any key -> next  |  s -> save  |  q -> quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import globals as g
from viz.utils import (
    _WINDOW_NAME,
    collect_images,
    draw_obb,
    resize_for_display,
    run_viewer,
)


def _render(
    img_path: Path,
    lbl_dir: Path,
    max_dim: Optional[int],
) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    draw_obb(img, lbl_dir / img_path.with_suffix(".txt").name)
    return resize_for_display(img, max_dim)


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize OBB-labelled frames with ground-truth boxes."
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Dataset split to visualize (default: train)",
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

    images = collect_images(img_dir, opt.source, opt.max)
    if not images:
        sys.exit("No matching images found.")

    print(
        f"Showing {len(images)} images from '{opt.split}' split. "
        "Press any key to advance, 's' to save, 'q' to quit."
    )

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    run_viewer(
        images,
        lambda p: _render(p, lbl_dir, opt.max_dim),
        g.RESULTS_DIR / "viz",
        lambda p: p.name,
    )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
