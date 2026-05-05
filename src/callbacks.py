"""
Ultralytics training and validation callbacks for the YOLOv9-OBB project.

Three groups:

Training callbacks (registered via attach_callbacks in train.py):
  on_train_start       — saves the W&B run ID to disk for --resume support
  on_train_epoch_start — optionally unfreezes backbone layers at a given epoch

Metadata / validation callbacks (registered via register_metadata_callbacks):
  on_val_start — wraps validator.update_metrics to pair per-image stats with
                 filenames before ultralytics discards the mapping
  on_val_end   — groups stats by altitude bucket, snow cover, and cloud cover
                 and stores them in _last_bucket_metrics
  on_fit_epoch_end — single W&B log per epoch (training mode only)

Design note
-----------
Ultralytics' validator clears its per-image stats inside get_stats() before
the on_val_end callback fires, and never stores the image-file <-> stats
mapping anywhere reachable. To work around this, on_val_start wraps
validator.update_metrics so each batch's im_file list is paired with the
fresh entries appended to validator.metrics.stats and stashed on the
validator. on_val_end then groups and computes per-bucket metrics with
ultralytics' own ap_per_class; on_fit_epoch_end picks them up and logs
everything to W&B in a single call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import wandb

import globals as g


# ── module state (metadata callbacks) ────────────────────────────────────────

_metadata_cache: Optional[Dict[str, Dict[str, Any]]] = None
_last_bucket_metrics: Dict[str, float] = {}


# ── training callbacks ───────────────────────────────────────────────────────

def make_save_wandb_id_callback() -> Callable:
    def on_train_start(trainer) -> None:
        if wandb.run is None:
            return
        id_file = Path(trainer.save_dir) / "wandb_run_id.txt"
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text(wandb.run.id)
        print(f"[wandb] Run ID saved to {id_file}")
    return on_train_start


def make_unfreeze_callback(
        unfreeze_epoch: int, lr_factor: float = 1.0
) -> Callable:
    """Build an on_train_epoch_start callback that unfreezes the backbone.

    Three Ultralytics-specific gotchas are handled here:

    1. `trainer.freeze_layer_names` is reset to `[".dfl"]` so that
       `BaseTrainer._model_train()` stops forcing previously-frozen
       BatchNorm layers into eval() mode at the start of each epoch.
       Without this, BN running stats stay locked to pretrain values
       and BN weight/bias gradients are computed against an
       out-of-distribution normalization.

    2. The DFL conv (`model.{N}.dfl.conv.weight`) is *not* unfrozen.
       Its weights are a fixed integration kernel `[0, 1, ..., reg_max-1]`
       used by the loss; training it breaks the head's distance regression.
       Ultralytics' `always_freeze_names` keeps it frozen by default and
       so do we.

    3. The trigger is `>=` plus a one-shot flag, not `==`. On resume,
       `_setup_train` re-applies the original `args.freeze`; with strict
       equality, resuming past `unfreeze_epoch` would silently leave the
       backbone frozen for the rest of training.

    4. LR rescaling is gated on `trainer.start_epoch <= unfreeze_epoch`.
       When resuming from a checkpoint saved *after* the original unfreeze
       fired, the optimizer's `lr` and `initial_lr` are already scaled,
       and applying `lr_factor` again would double-scale them.
    """
    def on_train_epoch_start(trainer) -> None:
        if getattr(trainer, "_did_unfreeze", False):
            return
        if trainer.epoch < unfreeze_epoch:
            return

        # 1. Flip requires_grad on every param except the DFL integral.
        n_unfrozen = 0
        for name, param in trainer.model.named_parameters():
            if ".dfl" in name:
                continue
            if not param.requires_grad:
                param.requires_grad = True
                n_unfrozen += 1

        # 2. Stop _model_train() from re-eval()ing backbone BN each epoch.
        #    `.dfl` is preserved to match Ultralytics' always_freeze_names;
        #    DFL has no BN inside it, so this is purely defensive.
        trainer.freeze_layer_names = [".dfl"]

        trainer._did_unfreeze = True
        print(
            f"[unfreeze] Unfroze {n_unfrozen} params at epoch "
            f"{trainer.epoch} (target {unfreeze_epoch})"
        )

        # 3. Optional LR rescaling. Both `lr` and `initial_lr` are updated:
        #    LambdaLR recomputes `lr = initial_lr * lf(epoch)` each step,
        #    and warmup interpolation also reads from `initial_lr`.
        #    Skip on resume past unfreeze_epoch: the loaded optimizer
        #    state already carries the scaled values from the original run.
        if lr_factor != 1.0 and trainer.start_epoch <= unfreeze_epoch:
            for pg in trainer.optimizer.param_groups:
                pg["lr"] *= lr_factor
                pg["initial_lr"] *= lr_factor
            new_lr = trainer.optimizer.param_groups[0]["lr"]
            print(
                f"[unfreeze] LR scaled by {lr_factor} → {new_lr:.6f}"
            )
        elif lr_factor != 1.0:
            print(
                f"[unfreeze] Skipping LR rescale on resume "
                f"(start_epoch={trainer.start_epoch} > "
                f"unfreeze_epoch={unfreeze_epoch}); "
                f"optimizer state already carries scaled LR"
            )
    return on_train_epoch_start


# ── metadata / validation callbacks ──────────────────────────────────────────

def _load_metadata() -> Dict[str, Dict[str, Any]]:
    global _metadata_cache
    if _metadata_cache is None:
        path = g.OUT_DIR / "metadata.json"
        with open(path) as f:
            _metadata_cache = json.load(f)
    assert _metadata_cache is not None
    return _metadata_cache


def _bucket_for(altitude: Optional[float]) -> Optional[str]:
    if altitude is None:
        return None
    for label, lo, hi in g.ALTITUDE_BUCKETS:
        if lo <= altitude < hi:
            return label
    return None


def _on_val_start(validator) -> None:
    """Wrap update_metrics once per validator instance, reset stats each call.

    The trainer keeps a single validator across all epochs (created once in
    BaseTrainer._setup_train), so naively rewrapping update_metrics every
    epoch nests wrappers — by epoch N, n_new entries get appended N times
    each call. We guard against this with a sentinel attribute and always
    reset _per_image_stats here.
    """
    validator._per_image_stats = []
    if getattr(validator, "_metadata_wrap_installed", False):
        return

    original_update = validator.update_metrics
    stats_dict = validator.metrics.stats

    def wrapped_update_metrics(preds: Any, batch: Any) -> None:
        before = {k: len(v) for k, v in stats_dict.items()}
        original_update(preds, batch)
        n_new = len(stats_dict["tp"]) - before["tp"]
        for i in range(n_new):
            entry = {k: stats_dict[k][before[k] + i] for k in stats_dict}
            validator._per_image_stats.append(
                (batch["im_file"][i], entry)
            )

    validator.update_metrics = wrapped_update_metrics
    validator._metadata_wrap_installed = True


def _to_key(s: str) -> str:
    """Normalize a metadata string to a W&B-safe key component.

    e.g. 'Fresh (5-10 cm)' -> 'fresh_5-10cm', 'Overcast' -> 'overcast'
    """
    return (
        s.lower()
         .replace("(", "").replace(")", "")
         .replace(" ", "_")
         .replace("__", "_")
         .strip("_")
    )


def _per_bucket_metrics(
        entries: List[Dict[str, Any]]
) -> Optional[Tuple[float, float, float, float]]:
    from ultralytics.utils.metrics import ap_per_class

    tp         = np.concatenate([e["tp"]         for e in entries], axis=0)
    conf       = np.concatenate([e["conf"]       for e in entries], axis=0)
    pred_cls   = np.concatenate([e["pred_cls"]   for e in entries], axis=0)
    target_cls = np.concatenate([e["target_cls"] for e in entries], axis=0)

    if tp.shape[0] == 0 or target_cls.shape[0] == 0:
        return None

    results = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    # ap_per_class returns: tp, fp, p, r, f1, ap, unique_classes, ...
    p, r, _, ap = results[2], results[3], results[4], results[5]
    if len(p) == 0:
        return None
    return float(p[0]), float(r[0]), float(ap[0, 0]), float(ap[0].mean())


def _log_categorical_buckets(
    per_image_stats: List[Tuple[str, Dict[str, Any]]],
    metadata: Dict[str, Dict[str, Any]],
    field: str,
    prefix: str,
) -> Dict[str, Union[float, int]]:
    bucketed: Dict[str, List[Dict[str, Any]]] = {}
    for im_file, entry in per_image_stats:
        stem = Path(im_file).stem
        meta = metadata.get(stem)
        val  = meta.get(field) if meta else None
        key  = _to_key(val) if val else "unknown"
        bucketed.setdefault(key, []).append(entry)

    log: Dict[str, Union[float, int]] = {}
    for bucket, entries in bucketed.items():
        n_imgs    = len(entries)
        n_targets = sum(int(e["target_cls"].size) for e in entries)
        log[f"{prefix}/{bucket}/n_images"]  = n_imgs
        log[f"{prefix}/{bucket}/n_targets"] = n_targets
        if n_targets == 0:
            continue
        m = _per_bucket_metrics(entries)
        if m is None:
            continue
        precision, recall, map50, map5095 = m
        log[f"{prefix}/{bucket}/precision"] = precision
        log[f"{prefix}/{bucket}/recall"]    = recall
        log[f"{prefix}/{bucket}/mAP50"]     = map50
        log[f"{prefix}/{bucket}/mAP50-95"]  = map5095
    return log


def _on_val_end(validator) -> None:
    if not getattr(validator, "_per_image_stats", None):
        return

    metadata        = _load_metadata()
    per_image_stats = validator._per_image_stats

    # ── altitude buckets ─────────────────────────────────────────────────────
    bucketed = {b[0]: [] for b in g.ALTITUDE_BUCKETS}
    bucketed["unknown"] = []
    for im_file, entry in per_image_stats:
        stem   = Path(im_file).stem
        meta   = metadata.get(stem)
        bucket = _bucket_for(meta["altitude_m"]) if meta else None
        bucketed.setdefault(bucket or "unknown", []).append(entry)

    log: Dict[str, Union[float, int]] = {}
    for bucket, entries in bucketed.items():
        if not entries:
            continue
        n_imgs    = len(entries)
        n_targets = sum(int(e["target_cls"].size) for e in entries)
        log[f"val_alt/{bucket}/n_images"]  = n_imgs
        log[f"val_alt/{bucket}/n_targets"] = n_targets
        if n_targets == 0:
            continue
        m = _per_bucket_metrics(entries)
        if m is None:
            continue
        precision, recall, map50, map5095 = m
        log[f"val_alt/{bucket}/precision"] = precision
        log[f"val_alt/{bucket}/recall"]    = recall
        log[f"val_alt/{bucket}/mAP50"]     = map50
        log[f"val_alt/{bucket}/mAP50-95"]  = map5095

    # ── snow cover & cloud cover buckets ─────────────────────────────────────
    log.update(_log_categorical_buckets(
        per_image_stats, metadata, "snow_cover", "val_snow",
    ))
    log.update(_log_categorical_buckets(
        per_image_stats, metadata, "cloud_cover", "val_cloud",
    ))

    _last_bucket_metrics.clear()
    _last_bucket_metrics.update(log)


def _on_fit_epoch_end(trainer) -> None:
    """Single source of truth for per-epoch W&B logging.

    Replaces ultralytics' built-in W&B integration (disabled in train.py) so
    every metric goes through one wandb.log call with no explicit step. The
    `epoch` field drives the chart x-axis via define_metric in train.py, so
    resumed runs can never trigger step-monotonicity warnings.
    """
    if wandb.run is None:
        return

    log: Dict[str, Any] = {"epoch": trainer.epoch + 1}
    if getattr(trainer, "tloss", None) is not None:
        log.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
    log.update(trainer.metrics)
    log.update(trainer.lr)
    log.update(_last_bucket_metrics)

    wandb.log(log)


# ── public API ───────────────────────────────────────────────────────────────

def get_last_bucket_metrics() -> Dict[str, float]:
    """Return the bucket metrics dict from the most recent validation pass."""
    return dict(_last_bucket_metrics)


def register_metadata_callbacks(model: Any, training: bool = True) -> None:
    """Register validation and per-epoch W&B logging callbacks on a YOLO model.

    training: also register on_fit_epoch_end for W&B logging (set False
    when running model.val() one-shot, e.g. in evaluate.py).
    """
    model.add_callback("on_val_start", _on_val_start)
    model.add_callback("on_val_end",   _on_val_end)
    if training:
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
