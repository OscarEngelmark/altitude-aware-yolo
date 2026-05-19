# Altitude-Aware Scaling for Car Detection on the Nordic Vehicle Dataset

Course project for D7047E. Trains a YOLOv9 Oriented Bounding Box (OBB) model to detect
cars in aerial drone footage captured at 120–250 m altitude under Nordic winter conditions.

Single detection class: `car`.

---

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data](#data)
- [Pipeline](#pipeline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization & Plots](#visualization--plots)
- [Project Structure](#project-structure)
- [Contributors](#contributors)

---

## Overview

The model replicates the approach from \[1\] using YOLOv9 with an OBB detection head
to handle rotated bounding boxes in top-down imagery. The key addition on top of the
baseline is altitude-aware scale augmentation (AAS), which normalizes the apparent car
size distribution across altitudes by sampling a target altitude and scaling each
training frame accordingly.

> \[1\] H. Mokayed, A. Nayebiastaneh, K. De, S. Sozos, O. Hagner, and B. Backe,
> "Nordic Vehicle Dataset (NVD): Performance of vehicle detectors using newly
> captured NVD from UAV in different snowy weather conditions,"
> in *Proc. IEEE/CVF CVPRW*, pp. 5314–5322, 2023,
> doi: [10.1109/CVPRW59228.2023.00560](https://doi.org/10.1109/CVPRW59228.2023.00560).

---

## Environment Setup

The project uses two environments:

| Environment | Purpose |
|---|---|
| Local | Code editing; local `.venv` for tooling only |
| Remote (JupyterLabs, Linux) | Training on NVIDIA RTX 2080 Ti |

### Remote setup (run after each environment reset)

```bash
bash setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` installs system dependencies (`git`, `libgl1`), creates a venv,
installs all pinned dependencies from `requirements.txt`, and enables W&B in
Ultralytics settings.

### Running commands

All Python commands must be run from `src/` (or with `src/` on `PYTHONPATH`)
because `globals.py` resolves paths relative to `__file__`.

```bash
cd src && python <script>.py
```

---

## Data

Raw data lives in `data/`:

| Item | Description |
|---|---|
| `data/*.zip` | Raw drone footage (video clips or PNG frames) + CVAT XML annotations — gitignored |
| `data/video_data.csv` | Per-video metadata: flight altitude, snow cover, cloud cover, resolution, FPS |
| `data/processed/` | Output of `preprocess.py` — gitignored |

`video_data.csv` is the only tracked file in `data/`; it is sourced from the
[Nordic Vehicle Dataset repository](https://github.com/amrdev-pixel/Nordic-Vehicle-Dataset).
The full dataset is available at [nvd.ltu-ai.dev](https://nvd.ltu-ai.dev/).
It is the source of truth for altitude ranges. Only rows with `Annotated=TRUE` are used.
Split assignment (train/val/test) is hardcoded in `src/preprocess.py` (`SPLIT_MAP`).

---

## Pipeline

### 1. Preprocess

Extracts annotated frames from ZIPs, converts CVAT XML to YOLO OBB label format,
and estimates per-frame altitudes.

```bash
cd src && python preprocess.py
```

**Outputs:**
- `data/processed/images/{train,val,test}/` — JPEG frames
- `data/processed/labels/{train,val,test}/` — YOLO OBB label files
  (`class x1 y1 x2 y2 x3 y3 x4 y4`, normalized)
- `data/processed/dataset.yaml` — dataset config (regenerated at train time)
- `data/processed/metadata.json` — per-frame altitude estimate + categorical metadata

Altitude is estimated using perspective geometry: bounding-box diagonal `l ∝ 1/H`.
A 4th-degree polynomial is fit to the `1/l` time series; the polynomial maximum
corresponds to `H_max` from the CSV.

---

## Training

```bash
cd src && python train.py --run-name <name>
```

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `yolov9s` | Model variant (`yolov9s`, `yolov9c`) |
| `--epochs` | `100` | Number of training epochs |
| `--batch` | `8` | Batch size |
| `--imgsz` | `1920` | Input image size |
| `--augment` | — | Augmentation preset stem (e.g. `paper`, `aug1`) |
| `--aas` | off | Enable altitude-aware scale augmentation |
| `--alt-min` / `--alt-max` | `100` / `300` | Altitude range for AAS (meters) |
| `--no-wandb` | off | Disable Weights & Biases logging |
| `--resume` | — | Resume from checkpoint path or W&B run ID |

### Examples

```bash
# Baseline with paper augmentation preset
python train.py --augment paper --run-name paper-baseline

# Larger model, fewer epochs
python train.py --model yolov9c --epochs 50 --batch 4 --run-name yolov9c-50ep

# Altitude-aware scaling with mosaic
python train.py --aas --augment paper --run-name aas-mosaic-01

# Quick test without W&B
python train.py --no-wandb --epochs 5 --run-name smoke-test
```

### Augmentation presets

Presets live in `augmentations/*.yaml`. Pass the filename stem to `--augment`.

| File | Description |
|---|---|
| `paper.yaml` | NVD paper preset — degrees=45, mosaic=1.0, mixup=0.1, copy_paste=0.1 |
| `aug1.yaml`–`aug6.yaml` | Experimental variants |

### Altitude-aware scale augmentation (AAS)

For a training frame at altitude `h`, a target altitude `h_target ~ U(alt_min, alt_max)`
is sampled and a scale `s = clip(h / h_target, 0.1, 3.0)` is applied to the whole image.
This flattens the apparent-altitude distribution over the training set.

Enable with `--aas`. When using mosaic, also pass an augmentation preset that sets
`mosaic=1.0` (e.g. `--augment paper`).

---

## Evaluation

```bash
cd src && python evaluate.py --run <run-name> --weights best.pt --split test
```

| Flag | Description |
|---|---|
| `--run` | Run name (subdirectory under `runs/`) |
| `--weights` | Checkpoint filename (`best.pt`, `last.pt`, or epoch number) |
| `--all-weights` | Evaluate all checkpoints in the run |
| `--split` | `test` or `val` |
| `--save-predictions` | Write per-image bounding boxes to JSON |

Appends a row to `results/evaluations.csv` with timestamp, run name, split,
precision, recall, mAP50, and mAP50-95.

---

## Visualization & Plots

### Frame viewers (interactive)

```bash
# Browse raw frames with ground-truth OBB labels
cd src && python viz/view_raw.py --split test

# Browse frames after augmentation pipeline
cd src && python viz/view_augmented.py --augment paper

# Browse saved model predictions
cd src && python viz/view_predictions.py --run <run-name> --split test --show-gt
```

### Diagnostic plots

```bash
# Altitude and bounding-box diagonal plots (requires metadata.json)
cd src && python plots/altitudes.py --out results/altitudes.png

# Simulate apparent-altitude distribution from AAS
cd src && python plots/altitude_dist.py

# Bounding-box size vs altitude
cd src && python plots/size_vs_altitude.py

# Side-by-side augmentation comparison
cd src && python plots/aug_comparison.py
```

### Tests

```bash
cd src && pytest test_altitude_augment.py -v
```

---

## Project Structure

```
.
├── augmentations/                # Augmentation preset YAMLs (paper.yaml, aug1–aug6.yaml)
├── configs/                      # YOLOv9 OBB model YAMLs (yolov9s-obb, yolov9c-obb)
├── data/
│   ├── *.zip                     # Raw drone footage + CVAT XML
│   ├── video_data.csv            # Per-video altitude, snow/cloud cover metadata
│   └── processed/                # Preprocessed frames, labels, metadata (gitignored)
├── models/                       # Pretrained YOLO weights (downloaded on first run)
├── results/                      # Evaluation CSVs, prediction JSONs, analysis plots
├── runs/                         # Ultralytics training outputs (weights, logs)
├── src/
│   ├── globals.py                # All path constants, altitude buckets, W&B config
│   ├── preprocess.py             # ZIPs → processed frames + YOLO OBB labels + metadata
│   ├── train.py                  # Training entrypoint
│   ├── evaluate.py               # Evaluation + metrics logging
│   ├── frame_metadata.py         # Per-frame altitude estimation
│   ├── altitude_augment.py       # Altitude-aware scale augmentation
│   ├── callbacks.py              # Per-bucket metrics, freeze/unfreeze, W&B ID save
│   ├── test_altitude_augment.py  # pytest suite
│   ├── viz/
│   │   ├── utils.py              # Shared drawing utilities
│   │   ├── view_raw.py           # Raw frame browser
│   │   ├── view_augmented.py     # Augmented frame browser
│   │   └── view_predictions.py   # Prediction browser
│   └── plots/
│       ├── altitudes.py          # Altitude/diagonal diagnostics
│       ├── altitude_dist.py      # Apparent-altitude distribution simulation
│       ├── size_vs_altitude.py   # Bbox size vs altitude
│       └── aug_comparison.py     # Augmentation comparison
├── requirements.txt
└── setup_env.sh                  # Remote environment setup
```

---

## Contributors

- Oscar Lundqvist Engelmark
- Hamid Sabeti
- Swagatam Biswas

Licensed under the [MIT License](LICENSE).
