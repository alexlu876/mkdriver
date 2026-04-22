"""CLI entry point for BTR training.

Usage
-----
::

    # Full training run — production config.
    uv run python scripts/train_btr.py --config configs/btr.yaml

    # Small smoke test against live Dolphin (Luigi Circuit only, tiny model,
    # ~500 env steps, tiny replay). Sanity-check that the training loop
    # doesn't crash before committing compute.
    uv run python scripts/train_btr.py --config configs/btr.yaml --testing

    # Device override for Vast.ai: --device cuda. For M4: --device mps
    # (remember to set layer_norm=false in btr.yaml or override via env).

What it does
------------
1. Load ``configs/btr.yaml`` (merging the ``testing:`` subtree if ``--testing``).
2. Construct the env, policy, target, replay, sampler, and logger.
3. Run episodes until ``total_frames`` env steps, logging per-episode
   returns + per-track rewards + per-component reward breakdown + sampler
   distribution to wandb (if ``WANDB_API_KEY`` set) or CSV under ``runs/btr/``.
4. Save checkpoints every ``checkpoint_every_grad_steps`` grad steps.

On first real launch
---------------------
Before burning Vast.ai compute, run ``--testing`` locally — it exercises
the full pipeline against live Dolphin on Luigi Circuit in ~1 minute and
tells you whether the env ↔ replay ↔ learn path works end-to-end. Any
crashes here are easier to debug than on a remote 4090.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/btr.yaml"),
        help="YAML config path (default: configs/btr.yaml).",
    )
    ap.add_argument(
        "--testing",
        action="store_true",
        help="Merge the YAML's ``testing:`` subtree on top of the main sections "
        "— tiny model + fast replay + ~500 env steps. For pipeline smoke tests.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override runtime.device in the YAML (cpu/mps/cuda).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override runtime.seed in the YAML.",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint .pt produced by a previous run "
        "(restores online/target/optimizer/counters; replay is re-warmed).",
    )
    args = ap.parse_args()

    # Deferred import so logging config above takes effect before any mkw_rl
    # module emits its own log lines during import.
    from mkw_rl.rl.train import load_config, train

    if not args.config.exists():
        print(f"error: config {args.config} not found", file=sys.stderr)
        return 1

    cfg = load_config(args.config, testing=args.testing)
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    if args.testing:
        logging.getLogger(__name__).info(
            "testing mode active — total_frames=%d, batch_size=%d, replay=%d",
            cfg.total_frames, cfg.batch_size, cfg.replay_size,
        )

    if args.resume is not None and not args.resume.exists():
        print(f"error: resume checkpoint {args.resume} not found", file=sys.stderr)
        return 1

    train(cfg, resume_from=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
