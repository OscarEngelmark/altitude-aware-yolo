"""Shared drawing and display utilities for the viz/ viewer scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

_WINDOW_NAME = "viewer"


def _parse_obb_label(label_path: Path) -> List[List[float]]:
    """Return normalized [x1,y1,...,x4,y4] float coords per valid OBB line."""
    if not label_path.is_file():
        return []
    rows: List[List[float]] = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            rows.append(list(map(float, parts[1:])))
    return rows


def draw_obb(
    img: np.ndarray,
    label_path: Path,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    for coords in _parse_obb_label(label_path):
        pts = np.array(
            [(coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)],
            dtype=np.int32,
        )
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def load_metadata(path: Path) -> Dict[str, float]:
    if not path.is_file():
        return {}
    with open(path) as f:
        raw: dict = json.load(f)
    return {
        stem: float(v["altitude_m"])
        for stem, v in raw.items()
        if v.get("altitude_m") is not None
    }


def load_obb_corners(label_path: Path, w: int, h: int) -> np.ndarray:
    """Return (N, 4, 2) pixel-space corners from an OBB label file."""
    rows = _parse_obb_label(label_path)
    if not rows:
        return np.zeros((0, 4, 2), dtype=np.float32)
    result: List[List[Tuple[float, float]]] = []
    for coords in rows:
        result.append(
            [(coords[i] * w, coords[i + 1] * h) for i in range(0, 8, 2)]
        )
    return np.array(result, dtype=np.float32)


def draw_corners(
    img: np.ndarray,
    corners: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    for box in corners.astype(np.int32):
        cv2.polylines(img, [box], isClosed=True, color=color, thickness=thickness)


def overlay_lines(img: np.ndarray, lines: List[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    thick = 1
    pad = 6
    for i, text in enumerate(lines):
        (_tw, th), _bl = cv2.getTextSize(text, font, fscale, thick)
        y = pad + (th + pad) * (i + 1)
        cv2.putText(
            img, text, (pad + 1, y + 1),
            font, fscale, (0, 0, 0), thick + 1, cv2.LINE_AA,
        )
        cv2.putText(
            img, text, (pad, y),
            font, fscale, (255, 255, 255), thick, cv2.LINE_AA,
        )


def resize_for_display(img: np.ndarray, max_dim: Optional[int]) -> np.ndarray:
    if max_dim and max(img.shape[:2]) > max_dim:
        sc = max_dim / max(img.shape[:2])
        return cv2.resize(
            img, (int(img.shape[1] * sc), int(img.shape[0] * sc))
        )
    return img


def _show_frame(frame: np.ndarray) -> None:
    cv2.imshow(_WINDOW_NAME, frame)
    cv2.resizeWindow(_WINDOW_NAME, frame.shape[1], frame.shape[0])


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    counter = 1
    while path.exists():
        path = path.parent / f"{stem}_{counter}{suffix}"
        counter += 1
    return path


def run_viewer(
    images: List[Path],
    render_fn: Callable[[Path], Optional[np.ndarray]],
    save_dir: Path,
    save_name_fn: Callable[[Path], str],
    allow_rerender: bool = False,
) -> None:
    """Forward-only viewer loop shared by raw and augmented modes.

    Keys: any -> next  |  r -> re-render (if allow_rerender)  |  s -> save
          q -> quit
    """
    idx = 0
    while idx < len(images):
        frame = render_fn(images[idx])
        if frame is None:
            idx += 1
            continue
        _show_frame(frame)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                return
            elif key == ord("r") and allow_rerender:
                new_frame = render_fn(images[idx])
                if new_frame is not None:
                    frame = new_frame
                _show_frame(frame)
            elif key == ord("s"):
                save_dir.mkdir(parents=True, exist_ok=True)
                out_path = _unique_path(save_dir / save_name_fn(images[idx]))
                cv2.imwrite(str(out_path), frame)
                print(f"Saved {out_path}")
            else:
                break
        idx += 1


def collect_images(
    img_dir: Path,
    source: Optional[str],
    max_count: Optional[int],
) -> List[Path]:
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if source:
        images = [p for p in images if source.lower() in p.name.lower()]
    if max_count:
        images = images[:max_count]
    return images
