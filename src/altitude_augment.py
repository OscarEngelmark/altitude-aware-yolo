"""
Altitude-aware scale augmentation for YOLO OBB training.

For each training frame at altitude h, a target altitude h_target is
sampled and scale s = h / h_target is applied, so the apparent altitude
equals h_target (up to clamping at SCALE_FLOOR / SCALE_CEILING).

Two target distributions are supported:
  uniform:    h_target ~ U(alt_min, alt_max)          -> flat distribution
  triangular: h_target ~ Triangular(alt_min, alt_max, alt_mode)

Public API
----------
AltitudeAwareOBBTrainer   pass to YOLO.train(trainer=...)
compute_scale_bounds       utility exposed for testing / plotting

Notes
-----
Altitude is injected into the labels dict by AltitudeAwareYOLODataset
and read by AltitudeAwareRandomPerspective.  When mosaic=0 the labels
dict passes through Mosaic unchanged.  When mosaic>0, AltitudeAwareMosaic
preserves altitude_m as mosaic_factor * mean(h_i), so the affine transform
fires with a valid effective altitude.  Because apparent_alt = h_target
regardless of the effective altitude, the triangular distribution propagates
through the mosaic path unchanged.
"""

import json
import math
import random
from pathlib import Path
from typing import cast, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch.nn as nn
from ultralytics.data.augment import Compose, Mosaic, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model

import globals as g

SCALE_FLOOR = 0.1
SCALE_CEILING = 3.0


def compute_scale_bounds(
    altitude_m: float,
    alt_min: float,
    alt_max: float,
) -> Tuple[float, float]:
    """Return (s_lo, s_hi) for altitude-aware scale augmentation.

    s = h / h_target; targeting h_max gives s_lo, h_min gives s_hi.
    Both values are clamped to [SCALE_FLOOR, SCALE_CEILING].
    """
    s_lo = float(np.clip(altitude_m / alt_max, SCALE_FLOOR, SCALE_CEILING))
    s_hi = float(np.clip(altitude_m / alt_min, SCALE_FLOOR, SCALE_CEILING))
    return s_lo, s_hi


class AltitudeAwareRandomPerspective(RandomPerspective):
    """RandomPerspective that samples scale from altitude-dependent bounds.

    Reads altitude_m from the labels dict (injected by
    AltitudeAwareYOLODataset) and samples scale from
    [h/alt_max, h/alt_min] instead of [1-scale, 1+scale].
    Falls back to symmetric bounds when altitude_m is absent.
    """

    def __init__(
        self,
        alt_min: float = 100.0,
        alt_max: float = 300.0,
        alt_mode: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alt_min = alt_min
        self.alt_max = alt_max
        self.alt_mode = alt_mode
        self._altitude_m: Optional[float] = None

    def affine_transform(
        self,
        img: np.ndarray,
        border: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Identical to RandomPerspective.affine_transform except uses
        self._scale_lo / self._scale_hi for the scale sample."""
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2
        C[1, 2] = -img.shape[0] / 2

        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)

        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        if self._altitude_m is not None:
            if self.alt_mode is not None:
                h_target = random.triangular(
                    self.alt_min, self.alt_max, self.alt_mode
                )
            else:
                h_target = random.uniform(self.alt_min, self.alt_max)
            s = float(np.clip(
                self._altitude_m / h_target, SCALE_FLOOR, SCALE_CEILING
            ))
        else:
            s = random.uniform(1.0 - self.scale, 1.0 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )
        S[1, 0] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )

        T = np.eye(3, dtype=np.float32)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate)
            * self.size[0]
        )
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate)
            * self.size[1]
        )

        M = T @ S @ R @ P @ C

        if (
            (border[0] != 0)
            or (border[1] != 0)
            or (M != np.eye(3)).any()
        ):
            if self.perspective:
                img = cv2.warpPerspective(
                    img, M,
                    dsize=self.size,
                    borderValue=(114, 114, 114),
                )
            else:
                img = cv2.warpAffine(
                    img, M[:2],
                    dsize=self.size,
                    borderValue=(114, 114, 114),
                )
            if img.ndim == 2:
                img = img[..., None]
        return img, M, s

    def __call__(self, labels: Dict) -> Dict:
        altitude_m = labels.get("altitude_m")
        self._altitude_m = (
            float(altitude_m) if altitude_m is not None else None
        )
        return super().__call__(labels)


def _swap_affine(
    transforms: Compose,
    alt_min: float,
    alt_max: float,
    alt_mode: Optional[float] = None,
) -> None:
    """In-place: replace RandomPerspective with AltitudeAwareRandomPerspective
    everywhere inside a Compose tree."""
    for i, t in enumerate(transforms.transforms):
        if isinstance(t, Compose):
            _swap_affine(t, alt_min, alt_max, alt_mode)
        elif type(t) is RandomPerspective:
            transforms.transforms[i] = AltitudeAwareRandomPerspective(
                alt_min=alt_min,
                alt_max=alt_max,
                alt_mode=alt_mode,
                degrees=t.degrees,
                translate=t.translate,
                scale=t.scale,
                shear=t.shear,
                perspective=t.perspective,
                border=t.border,
                pre_transform=t.pre_transform,
            )


class AltitudeAwareMosaic(Mosaic):
    """Mosaic that preserves altitude_m as the effective altitude.

    Ultralytics' Mosaic._cat_labels builds a fresh labels dict that drops
    all non-standard keys.  This override re-inserts altitude_m as the
    effective apparent altitude of the mosaic, so that
    AltitudeAwareRandomPerspective can still fire with a valid altitude
    after mosaicing.

    Mosaic places n frames (each imgsz x imgsz) on a sqrt(n)*imgsz canvas
    then crops back to imgsz.  Each frame therefore contributes at
    1/sqrt(n) linear scale, making objects appear sqrt(n)x further away.
    We store sqrt(n) * mean(h_i) so the perspective transform targets the
    correct h_target.
    """

    def _cat_labels(self, mosaic_labels: List) -> Dict:
        final_labels = super()._cat_labels(mosaic_labels)
        alts = [
            float(lbl["altitude_m"])
            for lbl in mosaic_labels
            if lbl.get("altitude_m") is not None
        ]
        if alts:
            mosaic_factor = int(self.n ** 0.5)  # 2 for n=4, 3 for n=9
            mean_alt = sum(alts) / len(alts)
            final_labels["altitude_m"] = mosaic_factor * mean_alt
        return final_labels


def _swap_mosaic(transforms: Compose) -> None:
    """In-place: replace Mosaic with AltitudeAwareMosaic in a Compose tree."""
    for i, t in enumerate(transforms.transforms):
        if isinstance(t, Compose):
            _swap_mosaic(t)
        elif type(t) is Mosaic:
            new_mosaic = AltitudeAwareMosaic(
                dataset=t.dataset,
                imgsz=t.imgsz,
                p=t.p,
                n=t.n,
            )
            new_mosaic.pre_transform = t.pre_transform
            transforms.transforms[i] = new_mosaic


class AltitudeAwareYOLODataset(YOLODataset):
    """YOLODataset that injects per-frame altitude into labels and uses
    AltitudeAwareRandomPerspective for training augmentation."""

    def __init__(
        self,
        *args,
        alt_min: float = 100.0,
        alt_max: float = 300.0,
        alt_mode: Optional[float] = None,
        metadata_path: Path = g.OUT_DIR / "metadata.json",
        **kwargs,
    ) -> None:
        self.alt_min = alt_min
        self.alt_max = alt_max
        self.alt_mode = alt_mode
        with open(metadata_path) as f:
            raw: dict = json.load(f)
        self._stem_to_alt: dict[str, float] = {
            stem: float(v["altitude_m"])
            for stem, v in raw.items()
            if v.get("altitude_m") is not None
        }
        super().__init__(*args, **kwargs)

    def get_image_and_label(self, index: int) -> Dict:
        label = super().get_image_and_label(index)
        stem = Path(label["im_file"]).stem
        alt = self._stem_to_alt.get(stem)
        if alt is not None:
            label["altitude_m"] = alt
        return label

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            _swap_mosaic(transforms)
            _swap_affine(transforms, self.alt_min, self.alt_max, self.alt_mode)
        return transforms


class AltitudeAwareOBBTrainer(OBBTrainer):
    """OBBTrainer that uses AltitudeAwareYOLODataset for the training split.

    alt_min and alt_max are extracted from the overrides dict (pass them
    as keyword arguments to YOLO.train).
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: Optional[Dict] = None,
        _callbacks: Optional[Dict] = None,
    ) -> None:
        overrides = dict(overrides or {})
        self.alt_min = float(overrides.pop("alt_min", 100.0))
        self.alt_max = float(overrides.pop("alt_max", 300.0))
        _mode = overrides.pop("alt_mode", None)
        self.alt_mode = float(_mode) if _mode is not None else None
        super().__init__(cfg, overrides, _callbacks)

    def optimizer_step(self) -> None:
        super().optimizer_step()
        if not self.ema:
            return
        ema_module = unwrap_model(self.ema.ema)
        ema_sd = ema_module.state_dict()
        if not any(
            v.is_floating_point() and not v.isfinite().all()
            for v in ema_sd.values()
        ):
            return
        model_sd = unwrap_model(cast(nn.Module, self.model)).state_dict()
        if all(
            not v.is_floating_point() or v.isfinite().all()
            for v in model_sd.values()
        ):
            for name, p in ema_module.named_parameters():
                if name in model_sd:
                    p.data.copy_(model_sd[name])
            for name, b in ema_module.named_buffers():
                if name in model_sd:
                    b.data.copy_(model_sd[name])
            LOGGER.warning(
                "NaN/Inf in EMA after update; "
                "reset to current model weights"
            )
        else:
            LOGGER.warning(
                "NaN/Inf in both EMA and model weights; "
                "consider stopping and resuming from last checkpoint"
            )

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: Optional[int] = None,
    ):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        stride = unwrap_model(cast(nn.Module, self.model)).stride
        gs = max(int(stride.max()), 32)  # type: ignore[operator]
        return AltitudeAwareYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=True,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
            alt_min=self.alt_min,
            alt_max=self.alt_max,
            alt_mode=self.alt_mode,
        )
