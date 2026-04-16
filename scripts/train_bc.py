#!/usr/bin/env python3
"""BC training driver.

Usage:
    uv run python scripts/train_bc.py --config configs/bc.yaml

Runs the full TBPTT training loop defined in ``src/mkw_rl/bc/train.py``,
plus checkpointing, logging, and the three spec-mandated diagnostics
printed at every epoch (per-bin steering CE, A-button F1, LSTM grad norm).

Single-epoch dry-run:

    uv run python scripts/train_bc.py --config configs/bc.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from glob import glob
from pathlib import Path

import torch

from mkw_rl.bc.train import (
    EpochStats,
    TrainConfig,
    build_model_and_optim,
    load_pickled_samples,
    make_dataset_and_loader,
    split_train_val,
    train_epoch,
    val_epoch,
)
from mkw_rl.utils.config import load_config
from mkw_rl.utils.logging import make_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train_bc")


def config_from_yaml(path: Path) -> TrainConfig:
    raw = load_config(path)
    return TrainConfig(
        demo_glob=raw.get("data", {}).get("demo_glob", TrainConfig.demo_glob),
        train_val_split=raw.get("data", {}).get("train_val_split", TrainConfig.train_val_split),
        batch_size=raw.get("data", {}).get("batch_size", TrainConfig.batch_size),
        num_workers=raw.get("data", {}).get("num_workers", TrainConfig.num_workers),
        stack_size=raw.get("data", {}).get("stack_size", TrainConfig.stack_size),
        seq_len=raw.get("data", {}).get("seq_len", TrainConfig.seq_len),
        frame_skip=raw.get("data", {}).get("frame_skip", TrainConfig.frame_skip),
        lr=raw.get("optim", {}).get("lr", TrainConfig.lr),
        weight_decay=raw.get("optim", {}).get("weight_decay", TrainConfig.weight_decay),
        epochs=raw.get("optim", {}).get("epochs", TrainConfig.epochs),
        grad_clip=raw.get("optim", {}).get("grad_clip", TrainConfig.grad_clip),
        steering_weight=raw.get("loss", {}).get("steering_weight", TrainConfig.steering_weight),
        button_weight=raw.get("loss", {}).get("button_weight", TrainConfig.button_weight),
        wandb_project=raw.get("logging", {}).get("wandb_project", TrainConfig.wandb_project),
        log_dir=Path(raw.get("logging", {}).get("log_dir", str(TrainConfig().log_dir))),
        checkpoint_every=raw.get("logging", {}).get("checkpoint_every", TrainConfig.checkpoint_every),
        device=raw.get("runtime", {}).get("device", TrainConfig.device),
        seed=raw.get("runtime", {}).get("seed", TrainConfig.seed),
    )


def report_diagnostics(stats: EpochStats) -> None:
    """Print the three spec-mandated diagnostics loudly."""
    log.info("=" * 80)
    log.info("EPOCH DIAGNOSTICS (spec §2.3)")
    log.info("=" * 80)

    # 1. Per-bin steering CE.
    per_bin = stats.per_bin_steering_loss
    import numpy as np

    seen = ~np.isnan(per_bin)
    if seen.any():
        seen_bins = np.where(seen)[0].tolist()
        losses = per_bin[seen]
        log.info("(1) per-bin steering CE, %d bins seen:", int(seen.sum()))
        for b, loss in zip(seen_bins, losses, strict=True):
            log.info("      bin %2d: %.4f", b, float(loss))
    else:
        log.info("(1) per-bin steering CE: NO BINS SEEN (empty training data?)")

    # 2. A-button F1.
    a_f1 = stats.per_button_f1.get("accelerate", 0.0)
    log.info("(2) A-button F1: %.4f  (spec threshold for 'learning': > 0.6)", a_f1)

    # 3. LSTM grad norm.
    log.info("(3) mean LSTM grad norm: %.6f  (spec threshold: > 1e-4)", stats.lstm_grad_norm)

    log.info("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="train 1 epoch and exit")
    parser.add_argument("--device", type=str, default=None, help="override device from config")
    args = parser.parse_args()

    cfg = config_from_yaml(args.config)
    if args.device:
        cfg = replace(cfg, device=args.device)
    if args.dry_run:
        cfg = replace(cfg, epochs=1)
    log.info("effective config: %s", cfg)

    device = torch.device(cfg.device)
    # Seed all three sources of randomness (M-2 audit fix).
    import random as _random

    import numpy as _np

    torch.manual_seed(cfg.seed)
    _np.random.seed(cfg.seed)
    _random.seed(cfg.seed)

    # Load data.
    # Expand user (~) and environment vars in the glob pattern (L-5 audit fix).
    import os as _os

    demo_glob_expanded = _os.path.expanduser(_os.path.expandvars(cfg.demo_glob))
    demo_paths = sorted(glob(demo_glob_expanded))
    if not demo_paths:
        log.error("no pickled demos matched %s", demo_glob_expanded)
        log.error("run scripts/parse_demo.py first to produce them")
        return 1
    log.info("loading %d demo payload(s)", len(demo_paths))
    samples = load_pickled_samples([Path(p) for p in demo_paths])
    log.info("loaded %d demos", len(samples))

    train_samples, val_samples = split_train_val(samples, train_frac=cfg.train_val_split, seed=cfg.seed)
    log.info("split: %d train demos / %d val demos", len(train_samples), len(val_samples))

    train_ds, train_loader = make_dataset_and_loader(train_samples, cfg, shuffle=True)
    log.info("train dataset: %d chunks across %d demos", len(train_ds), len(train_ds.demo_ids))

    # Validation loader. Only built if there ARE val demos and enough data
    # for at least one seq_len window; otherwise we skip val and track best
    # by training loss (with a loud warning).
    val_loader = None
    if val_samples:
        try:
            val_ds, val_loader = make_dataset_and_loader(val_samples, cfg, shuffle=False)
            log.info("val dataset: %d chunks across %d demos", len(val_ds), len(val_ds.demo_ids))
        except ValueError as exc:
            # MkwBCDataset raises if no demo is long enough for a chunk.
            log.warning("val dataset unavailable (%s); best-checkpoint will use train loss", exc)

    if val_loader is None:
        log.warning(
            "no validation loader — 'bc_best.pt' will be picked by training loss, "
            "which is NOT a good proxy for generalization. Consider adding more demos."
        )

    model, optimizer, scheduler = build_model_and_optim(cfg, device)
    log.info("model param count: %.2fM", model.param_count() / 1e6)

    # Sanitize config for wandb — convert Paths to strings (H-4 audit fix).
    def _jsonable(v):
        return str(v) if isinstance(v, Path) else v

    sanitized_cfg = {k: _jsonable(v) for k, v in cfg.__dict__.items()}

    logger = make_logger(
        project=cfg.wandb_project,
        csv_fallback_path=cfg.log_dir / "metrics.csv",
        config=sanitized_cfg,
    )

    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Persist a serializable snapshot of TrainConfig + BCPolicyConfig in
    # every checkpoint so eval_bc.py can rebuild the model with the same
    # architecture knobs (M-3 audit fix).
    ckpt_config = {
        "train_config": sanitized_cfg,
        "model_config": model.config.__dict__,
    }

    best_key = float("inf")
    best_metric_name = "val_total" if val_loader is not None else "train_total"
    for epoch in range(cfg.epochs):
        log.info("epoch %d/%d starting", epoch + 1, cfg.epochs)
        stats = train_epoch(model, train_loader, optimizer, scheduler, cfg, device)
        log.info(
            "epoch %d train: total=%.4f steering=%.4f buttons=%.4f grad_norm=%.6f",
            epoch + 1,
            stats.loss_total,
            stats.loss_steering,
            stats.loss_buttons,
            stats.lstm_grad_norm,
        )
        report_diagnostics(stats)

        log_row = {
            "epoch": epoch + 1,
            "train/total": stats.loss_total,
            "train/steering": stats.loss_steering,
            "train/buttons": stats.loss_buttons,
            "train/lstm_grad_norm": stats.lstm_grad_norm,
            **{f"train/button_f1_{k}": v for k, v in stats.per_button_f1.items()},
        }

        # Val pass (H-2 audit fix).
        if val_loader is not None:
            v = val_epoch(model, val_loader, cfg, device)
            log.info(
                "epoch %d val:   total=%.4f steering=%.4f buttons=%.4f",
                epoch + 1,
                v.loss_total,
                v.loss_steering,
                v.loss_buttons,
            )
            log_row.update(
                {
                    "val/total": v.loss_total,
                    "val/steering": v.loss_steering,
                    "val/buttons": v.loss_buttons,
                }
            )
            current_key = v.loss_total
        else:
            current_key = stats.loss_total

        logger.log(log_row)

        # Checkpoints.
        if (epoch + 1) % cfg.checkpoint_every == 0:
            ckpt_path = cfg.log_dir / f"bc_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": ckpt_config,
                    "epoch": epoch + 1,
                    "best_metric": best_metric_name,
                },
                ckpt_path,
            )
            log.info("wrote checkpoint %s", ckpt_path)
        if current_key < best_key:
            best_key = current_key
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": ckpt_config,
                    "epoch": epoch + 1,
                    "best_metric": best_metric_name,
                },
                cfg.log_dir / "bc_best.pt",
            )

    logger.close()
    log.info("training done. best %s: %.4f", best_metric_name, best_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
