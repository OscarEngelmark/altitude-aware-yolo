"""
Preprocessing script: extract annotated frames from zipped drone footage
and convert CVAT XML annotations to YOLO OBB format.

Output structure:
  data/processed/
    images/{train,val,test}/   - JPEG frames that have at least one annotation
    labels/{train,val,test}/   - YOLO OBB .txt files
                                 (class x1 y1 x2 y2 x3 y3 x4 y4, normalized)
    dataset.yaml
    metadata.json
"""

import json
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import globals as g
import numpy as np
import yaml

from frame_metadata import load_video_csv, compute_frame_metadata

JPEG_QUALITY = 95

SPLIT_MAP: Dict[str, str] = {
    "2022-12-02 Asjo 01_stabilized.zip":      "train",
    "2022-12-04 Bjenberg 02.zip":             "train",
    "2022-12-23 Asjo 01_HD 5x stab.zip":      "train",
    "2022-12-03 Nyland 01_stabilized.zip":    "val",
    "2022-12-23 Bjenberg 02_stabilized.zip":  "test",
}

# ── helpers ─────────────────────────────────────────────────────────────────

def parse_annotations(
        xml_bytes: bytes
    ) -> Tuple[
        Dict[int, List[Tuple[float, float, float, float, float]]], int, int
        ]:
    """
    Parse a CVAT interpolation XML.

    Returns
    -------
    annotations : {frame_id: [(cx_norm, cy_norm, w_norm, h_norm,
                               angle_deg), ...]}
        Only frames where at least one box has outside=0.
    img_w, img_h : image dimensions from <original_size>
    """
    root = ET.fromstring(xml_bytes)

    # image dimensions
    size = root.find(".//original_size")
    if size is None:
        raise ValueError("XML missing <original_size>")
    img_w = int(size.findtext("width") or 1920)
    img_h = int(size.findtext("height") or 1080)

    annotations = defaultdict(list)

    for track in root.findall(".//track"):
        for box in track.findall("box"):
            if int(box.attrib.get("outside", "0")):
                continue  # object not visible in this frame

            frame = int(box.attrib["frame"])
            xtl   = float(box.attrib["xtl"])
            ytl   = float(box.attrib["ytl"])
            xbr   = float(box.attrib["xbr"])
            ybr   = float(box.attrib["ybr"])
            angle = float(box.attrib.get("rotation", "0.0"))

            cx = ((xtl + xbr) / 2) / img_w
            cy = ((ytl + ybr) / 2) / img_h
            w  = (xbr - xtl) / img_w
            h  = (ybr - ytl) / img_h

            annotations[frame].append((cx, cy, w, h, angle))

    return dict(annotations), img_w, img_h


def xywha_to_corners(
    cx: float, cy: float, w: float, h: float, angle_deg: float,
    img_w: int, img_h: int,
) -> np.ndarray:
    """Convert rotated box (cx, cy, w, h normalized, angle_deg) to
    4 normalized corners.

    Rotation is applied in pixel space to match CVAT's coordinate convention.
    Normalizing before rotation distorts angles on non-square images
    (e.g. 1920×1080).
    """
    cx_px, cy_px = cx * img_w, cy * img_h
    w_px,  h_px  = w  * img_w, h  * img_h
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    dx, dy = w_px / 2, h_px / 2
    corners = np.array(
        [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64
    )
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts_px = (corners @ R.T) + np.array([cx_px, cy_px])
    return np.clip(pts_px / np.array([img_w, img_h]), 0.0, 1.0)


def save_label(
        path: Path,
        boxes: List[Tuple[float, float, float, float, float]],
        img_w: int,
        img_h: int,
    ) -> None:
    """Write YOLO OBB label file: class x1 y1 … x4 y4 (class=0 = car)."""
    lines = []
    for cx, cy, w, h, angle in boxes:
        pts = xywha_to_corners(cx, cy, w, h, angle, img_w, img_h).flatten()
        coords = " ".join(f"{v:.6f}" for v in pts)
        lines.append(f"0 {coords}")
    path.write_text("\n".join(lines))


def frame_stem(zip_stem: str, frame_id: int) -> str:
    """Canonical file stem, e.g. 'asjo01_f00080'."""
    tag = zip_stem.lower().replace(" ", "_")
    return f"{tag}_f{frame_id:05d}"


# ── per-zip extractors ──────────────────────────────────────────────────────

def _record_metadata(
    metadata: Dict[str, Dict[str, Any]],
    stem: str,
    frame_id: int,
    n_boxes: int,
    frame_fmeta: Dict[str, Optional[float]],
    zip_stem: str,
    split: str,
    vmeta: Dict[str, Any],
) -> None:
    metadata[stem] = {
        "video":      zip_stem,
        "split":      split,
        "frame_id":   frame_id,
        "n_boxes":    n_boxes,
        **frame_fmeta,
        "snow_cover": vmeta["snow_cover"],
        "cloud_cover": vmeta["cloud_cover"],
    }


def process_video_zip(
    zf: zipfile.ZipFile,
    video_name: str,
    xml_name: str,
    zip_stem: str,
    img_dir: Path,
    lbl_dir: Path,
    vmeta: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    split: str,
) -> int:
    """Extract annotated frames from a zip that contains a video file."""
    print(f"  Parsing annotations from {xml_name} …")
    annotations, img_w, img_h = parse_annotations(zf.read(xml_name))
    annotated_frames = set(annotations.keys())
    print(f"  {len(annotated_frames)} annotated frames found")

    frame_meta = compute_frame_metadata(annotations, img_w, img_h, vmeta)

    print(f"  Extracting video {video_name} to memory …")
    tmp_path = g.OUT_DIR / "_tmp_video"
    tmp_path.write_bytes(zf.read(video_name))

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_name}")

    saved = 0
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in annotated_frames:
            stem = frame_stem(zip_stem, frame_id)
            cv2.imwrite(
                str(img_dir / f"{stem}.jpg"), frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            save_label(
                lbl_dir / f"{stem}.txt",
                annotations[frame_id], img_w, img_h,
            )
            _record_metadata(
                metadata, stem, frame_id,
                len(annotations[frame_id]),
                frame_meta.get(frame_id, {}),
                zip_stem, split, vmeta,
            )
            saved += 1
        frame_id += 1

    cap.release()
    tmp_path.unlink()
    return saved


def process_frames_zip(
    zf: zipfile.ZipFile,
    xml_name: str,
    zip_stem: str,
    img_dir: Path,
    lbl_dir: Path,
    vmeta: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    split: str,
) -> int:
    """Extract annotated frames from a zip that already contains PNG frames."""
    print(f"  Parsing annotations from {xml_name} …")
    annotations, img_w, img_h = parse_annotations(zf.read(xml_name))
    annotated_frames = set(annotations.keys())
    print(f"  {len(annotated_frames)} annotated frames found")

    frame_meta = compute_frame_metadata(annotations, img_w, img_h, vmeta)

    png_names = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))

    saved = 0
    for png_name in png_names:
        # frame index encoded in filename: frame_000080.PNG → 80
        frame_id = int(Path(png_name).stem.split("_")[-1])

        if frame_id not in annotated_frames:
            continue

        arr = np.frombuffer(zf.read(png_name), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  Warning: could not decode {png_name}, skipping")
            continue

        stem = frame_stem(zip_stem, frame_id)
        cv2.imwrite(
            filename=str(img_dir / f"{stem}.jpg"),
            img=frame,
            params=[cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        save_label(
            lbl_dir / f"{stem}.txt",
            annotations[frame_id], img_w, img_h,
        )
        _record_metadata(
            metadata, stem, frame_id,
            len(annotations[frame_id]),
            frame_meta.get(frame_id, {}),
            zip_stem, split, vmeta,
        )
        saved += 1

    return saved


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    video_csv = load_video_csv()

    # create split subdirectories
    for split in ("train", "val", "test"):
        (g.OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (g.OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_saved = 0
    metadata: Dict[str, Dict[str, Any]] = {}

    for zip_name, split in SPLIT_MAP.items():
        zip_path = g.DATA_DIR / zip_name
        if not zip_path.exists():
            print(f"[SKIP] {zip_name} not found")
            continue

        zip_stem = Path(zip_name).stem
        if zip_stem not in video_csv:
            raise KeyError(
                f"{zip_name} not found in data/video_data.csv — "
                "add it and set Annotated=TRUE."
            )
        vmeta = video_csv[zip_stem]

        print(f"\n[{split}] {zip_name}  "
              f"(H_max={vmeta['h_max']:.0f} m, "
              f"snow={vmeta['snow_cover']}, cloud={vmeta['cloud_cover']})")
        img_dir = g.OUT_DIR / "images" / split
        lbl_dir = g.OUT_DIR / "labels" / split

        with zipfile.ZipFile(zip_path) as zf:
            members     = zf.namelist()
            xml_files   = [n for n in members if n.endswith(".xml")]
            video_files = [
                n for n in members
                if n.lower().endswith((".mp4", ".avi", ".mov"))
            ]
            png_files   = [n for n in members if n.lower().endswith(".png")]

            if not xml_files:
                print("  No XML found, skipping")
                continue

            if video_files:
                saved = process_video_zip(
                    zf, video_files[0], xml_files[0],
                    zip_stem, img_dir, lbl_dir, vmeta, metadata, split,
                )
            elif png_files:
                saved = process_frames_zip(
                    zf, xml_files[0], zip_stem, img_dir, lbl_dir,
                    vmeta, metadata, split,
                )
            else:
                print("  No video or PNG frames found, skipping")
                continue

        print(f"  Saved {saved} frames → {split}")
        total_saved += saved

    # write dataset.yaml for YOLOv9
    dataset_yaml = {
        "path": str(g.OUT_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": {0: "car"},
    }
    yaml_path = g.OUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    metadata_path = g.OUT_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"\nDone. {total_saved} total frames → {g.OUT_DIR}")
    print(f"dataset.yaml written to {yaml_path}")
    print(f"metadata.json written to"
          f"{metadata_path} ({len(metadata)} entries)")


if __name__ == "__main__":
    main()
