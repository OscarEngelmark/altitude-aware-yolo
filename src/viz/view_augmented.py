"""Visualize the full training augmentation pipeline applied to frames.

Usage
-----
    cd src && python viz/view_augmented.py --augment paper
    cd src && python viz/view_augmented.py --augment aas2 --source Asjo --max 50
    cd src && python viz/view_augmented.py --augment paper --alt-min 80 --alt-max 320

Controls: any key -> next  |  r -> re-augment same image  |  s -> save  |  q -> quit
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import globals as g
from altitude_augment import AltitudeAwareRandomPerspective
from viz.utils import (
    _WINDOW_NAME,
    collect_images,
    draw_corners,
    load_metadata,
    load_obb_corners,
    overlay_lines,
    resize_for_display,
    run_viewer,
)


class _InstrumentedPerspective(AltitudeAwareRandomPerspective):
    """Records scale and apparent target altitude after each affine call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_scale: Optional[float] = None
        self.last_h_target: Optional[float] = None

    def affine_transform(
        self,
        img: np.ndarray,
        border: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        img_out, M, s = super().affine_transform(img, border)
        self.last_scale = s
        self.last_h_target = (
            self._altitude_m / s
            if self._altitude_m is not None and s > 0
            else None
        )
        return img_out, M, s


# ---------------------------------------------------------------------------
# Mosaic helpers
# ---------------------------------------------------------------------------

def _filter_corners(
    corners: np.ndarray, w: int, h: int, min_overlap: float = 0.1
) -> np.ndarray:
    """Drop OBB boxes with less than min_overlap fraction inside [0,w]×[0,h]."""
    if corners.shape[0] == 0:
        return corners
    keep: List[np.ndarray] = []
    for box in corners:
        bx1, by1 = float(box[:, 0].min()), float(box[:, 1].min())
        bx2, by2 = float(box[:, 0].max()), float(box[:, 1].max())
        box_area = (bx2 - bx1) * (by2 - by1)
        if box_area <= 0:
            continue
        ix = max(0.0, min(bx2, float(w)) - max(bx1, 0.0))
        iy = max(0.0, min(by2, float(h)) - max(by1, 0.0))
        if (ix * iy) / box_area >= min_overlap:
            keep.append(box)
    return (
        np.array(keep, dtype=np.float32)
        if keep
        else np.zeros((0, 4, 2), dtype=np.float32)
    )


def _build_mosaic(
    items: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Tile 4 (img, corners) pairs using ultralytics-style random-centre layout."""
    h0, w0 = items[0][0].shape[:2]
    cy = random.randint(h0 // 2, 3 * h0 // 2)
    cx = random.randint(w0 // 2, 3 * w0 // 2)
    canvas = np.full((2 * h0, 2 * w0, 3), 114, dtype=np.uint8)

    tile_origins: List[Tuple[int, int]] = [
        (cx - w0, cy - h0),
        (cx,      cy - h0),
        (cx - w0, cy),
        (cx,      cy),
    ]
    crop_x, crop_y = cx - w0 // 2, cy - h0 // 2

    all_corners: List[np.ndarray] = []
    for (img, corners), (ox, oy) in zip(items, tile_origins):
        ih, iw = img.shape[:2]
        img_r = cv2.resize(img, (w0, h0))
        x1c, y1c = max(ox, 0), max(oy, 0)
        x2c, y2c = min(ox + w0, 2 * w0), min(oy + h0, 2 * h0)
        x1i, y1i = x1c - ox, y1c - oy
        canvas[y1c:y2c, x1c:x2c] = img_r[
            y1i:y1i + (y2c - y1c), x1i:x1i + (x2c - x1c)
        ]
        if len(corners):
            c = corners.copy().astype(np.float32)
            c[:, :, 0] = c[:, :, 0] * (w0 / iw) + (ox - crop_x)
            c[:, :, 1] = c[:, :, 1] * (h0 / ih) + (oy - crop_y)
            all_corners.append(c)

    mosaic = canvas[crop_y:crop_y + h0, crop_x:crop_x + w0].copy()
    merged = (
        np.concatenate(all_corners, axis=0)
        if all_corners
        else np.zeros((0, 4, 2), dtype=np.float32)
    )
    return mosaic, _filter_corners(merged, w0, h0)


def _build_source(
    img_path: Path,
    raw: np.ndarray,
    corners: np.ndarray,
    altitude_m: Optional[float],
    all_split_images: List[Path],
    lbl_dir: Path,
    metadata: Dict[str, float],
    aug_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Return (src_img, src_corners, src_alt) for the augmentation step."""
    use_mosaic = random.random() < aug_cfg.get("mosaic", 0.0)
    if use_mosaic and len(all_split_images) >= 4:
        pool = [p for p in all_split_images if p != img_path]
        extra_paths = random.sample(pool, min(3, len(pool)))
        items: List[Tuple[np.ndarray, np.ndarray]] = [(raw, corners)]
        tile_alts: List[float] = (
            [altitude_m] if altitude_m is not None else []
        )
        for extra_path in extra_paths:
            extra_img = cv2.imread(str(extra_path))
            if extra_img is None:
                items.append((raw, corners))
            else:
                eh, ew = extra_img.shape[:2]
                extra_corners = load_obb_corners(
                    lbl_dir / extra_path.with_suffix(".txt").name, ew, eh
                )
                items.append((extra_img, extra_corners))
            alt_e = metadata.get(extra_path.stem)
            if alt_e is not None:
                tile_alts.append(alt_e)
        while len(items) < 4:
            items.append(items[0])
        src_img, src_corners = _build_mosaic(items[:4])
        src_alt: Optional[float] = (
            sum(tile_alts) / len(tile_alts) if tile_alts else None
        )
    else:
        src_img, src_corners, src_alt = raw, corners, altitude_m
    return src_img, src_corners, src_alt


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _apply_copy_paste(
    img: np.ndarray,
    corners: np.ndarray,
    p: float,
    mode: str = "flip",
) -> Tuple[np.ndarray, np.ndarray]:
    if p == 0 or len(corners) == 0:
        return img, corners
    from ultralytics.data.augment import CopyPaste
    from ultralytics.utils.instance import Instances

    bboxes = np.stack([
        corners[:, :, 0].min(1), corners[:, :, 1].min(1),
        corners[:, :, 0].max(1), corners[:, :, 1].max(1),
    ], axis=1).astype(np.float32)
    inst = Instances(
        bboxes, segments=corners, bbox_format="xyxy", normalized=False
    )
    labels: Dict = {
        "img": img.copy(),
        "cls": np.zeros(len(corners), dtype=np.int32),
        "instances": inst,
    }
    result = CopyPaste(dataset=None, p=p, mode=mode)(labels)
    out_corners = result["instances"].segments
    return result["img"], out_corners.astype(np.float32)


def _apply_mixup(
    img: np.ndarray,
    corners: np.ndarray,
    p: float,
    all_split_images: List[Path],
    lbl_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if p == 0 or random.random() > p or not all_split_images:
        return img, corners
    other_path = random.choice(all_split_images)
    other_img = cv2.imread(str(other_path))
    if other_img is None:
        return img, corners
    h, w = img.shape[:2]
    other_img = cv2.resize(other_img, (w, h))
    other_corners = load_obb_corners(
        lbl_dir / other_path.with_suffix(".txt").name, w, h
    )
    r = float(np.random.beta(32, 32))
    mixed = (img * r + other_img * (1 - r)).astype(np.uint8)
    if len(corners) and len(other_corners):
        combined = np.concatenate([corners, other_corners], axis=0)
    elif len(other_corners):
        combined = other_corners
    else:
        combined = corners
    return mixed, combined


def _apply_augment(
    raw: np.ndarray,
    corners: np.ndarray,
    altitude_m: Optional[float],
    transform: _InstrumentedPerspective,
    aug_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    """Apply the full aug pipeline: affine -> HSV -> flipud -> fliplr."""
    from ultralytics.data.augment import RandomHSV

    img = raw.copy()
    transform.size = (img.shape[1], img.shape[0])
    transform._altitude_m = (
        float(altitude_m) if altitude_m is not None else None
    )
    aug_img, M, _s = transform.affine_transform(img, border=(0, 0))

    aug_corners = corners.copy()
    if corners.shape[0] > 0:
        pts = corners.reshape(-1, 1, 2).astype(np.float32)
        aug_corners = cv2.perspectiveTransform(pts, M).reshape(corners.shape)

    RandomHSV(
        hgain=aug_cfg.get("hsv_h", 0.0),
        sgain=aug_cfg.get("hsv_s", 0.0),
        vgain=aug_cfg.get("hsv_v", 0.0),
    )({"img": aug_img})

    h_out, w_out = aug_img.shape[:2]
    if random.random() < aug_cfg.get("flipud", 0.0):
        aug_img = np.ascontiguousarray(np.flipud(aug_img))
        if len(aug_corners):
            aug_corners[:, :, 1] = h_out - aug_corners[:, :, 1]
    if random.random() < aug_cfg.get("fliplr", 0.0):
        aug_img = np.ascontiguousarray(np.fliplr(aug_img))
        if len(aug_corners):
            aug_corners[:, :, 0] = w_out - aug_corners[:, :, 0]

    return aug_img, aug_corners, transform.last_scale, transform.last_h_target


# ---------------------------------------------------------------------------
# Frame renderer
# ---------------------------------------------------------------------------

def _render(
    img_path: Path,
    all_split_images: List[Path],
    lbl_dir: Path,
    metadata: Dict[str, float],
    aug_cfg: Dict,
    transform: _InstrumentedPerspective,
    max_dim: Optional[int],
    augment_name: str,
) -> Optional[np.ndarray]:
    raw = cv2.imread(str(img_path))
    if raw is None:
        return None
    h, w = raw.shape[:2]
    corners = load_obb_corners(lbl_dir / img_path.with_suffix(".txt").name, w, h)
    altitude_m = metadata.get(img_path.stem)

    src_img, src_corners, src_alt = _build_source(
        img_path, raw, corners, altitude_m,
        all_split_images, lbl_dir, metadata, aug_cfg,
    )
    src_img, src_corners = _apply_copy_paste(
        src_img, src_corners,
        aug_cfg.get("copy_paste", 0.0),
        aug_cfg.get("copy_paste_mode", "flip"),
    )
    aug, aug_c, _, _ = _apply_augment(
        src_img, src_corners, src_alt, transform, aug_cfg
    )
    aug, aug_c = _apply_mixup(
        aug, aug_c,
        aug_cfg.get("mixup", 0.0),
        all_split_images, lbl_dir,
    )
    draw_corners(aug, aug_c)
    overlay_lines(aug, [img_path.stem, f"Augmentation: {augment_name}.yaml"])
    return resize_for_display(aug, max_dim)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the training augmentation pipeline on frames."
    )
    parser.add_argument(
        "--augment", type=str, required=True,
        help="Augmentation preset stem from augmentations/ (e.g. 'paper', 'aas1')",
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
    parser.add_argument(
        "--alt-min", type=float, default=100.0, dest="alt_min",
        help="AAS minimum target altitude in metres (default: 100)",
    )
    parser.add_argument(
        "--alt-max", type=float, default=300.0, dest="alt_max",
        help="AAS maximum target altitude in metres (default: 300)",
    )
    parser.add_argument(
        "--alt-dist", choices=["uniform", "triangular"], default="uniform",
        dest="alt_dist",
        help="Target altitude distribution: uniform (default) or triangular",
    )
    parser.add_argument(
        "--alt-mode", type=float, default=None, dest="alt_mode",
        help=(
            "Peak of the triangular target distribution in metres "
            "(only used with --alt-dist triangular; defaults to midpoint)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    import yaml

    opt = parse_opt()
    img_dir = g.IMG_DIR / opt.split
    lbl_dir = g.LBL_DIR / opt.split

    if not img_dir.is_dir():
        sys.exit(f"No images found at {img_dir} — run preprocessing.py first.")

    yaml_path = g.AUGS_DIR / f"{opt.augment}.yaml"
    if not yaml_path.is_file():
        sys.exit(f"Augmentation preset not found: {yaml_path}")
    with open(yaml_path) as f:
        aug_cfg: Dict = yaml.safe_load(f) or {}

    all_split_images = (
        sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    )
    images = collect_images(img_dir, opt.source, opt.max)
    if not images:
        sys.exit("No matching images found.")

    metadata = load_metadata(g.OUT_DIR / "metadata.json")
    transform = _InstrumentedPerspective(
        alt_min=opt.alt_min,
        alt_max=opt.alt_max,
        dist=opt.alt_dist,
        alt_mode=opt.alt_mode,
        degrees=float(aug_cfg.get("degrees", 10.0)),
        translate=float(aug_cfg.get("translate", 0.1)),
        scale=float(aug_cfg.get("scale", 0.5)),
        shear=float(aug_cfg.get("shear", 0.0)),
        perspective=0.0,
        pre_transform=None,
    )

    print(
        f"Showing {len(images)} images from '{opt.split}' split "
        f"| augment={opt.augment}"
        f" alt_min={opt.alt_min} alt_max={opt.alt_max}\n"
        "Any key -> next  |  r -> re-augment  |  s -> save  |  q -> quit"
    )

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    run_viewer(
        images,
        lambda p: _render(
            p, all_split_images, lbl_dir, metadata,
            aug_cfg, transform, opt.max_dim, opt.augment,
        ),
        g.RESULTS_DIR / "viz_augmented",
        lambda p: f"{p.stem}_aug.jpg",
        allow_rerender=True,
    )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
