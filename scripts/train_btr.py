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
        "--num-envs",
        type=int,
        default=None,
        help="Override env.num_envs in the YAML. >1 enables multi-env "
        "parallel rollout; each env_id=0..N-1 uses its own dolphin{i}/ dir "
        "(run scripts/setup_dolphin_instances.py to clone them first).",
    )
    # Shakedown overrides. Production warmup is ~3h; each bug that surfaces
    # past warmup costs ~3h to discover. These flags let us run the full
    # production model + batch_size + threading stack with a TINY warmup
    # (couple minutes) and a TINY post-warmup training window, so a 10-min
    # shakedown verifies the whole pipeline end-to-end instead of just
    # checking imports. See SETUP.md "Shakedown runs".
    ap.add_argument(
        "--min-sampling-size",
        type=int,
        default=None,
        help="Override training.min_sampling_size (warmup env-step threshold). "
        "Default None keeps the YAML value (200_000 in production). Lower "
        "to 2000-5000 for shakedown runs.",
    )
    ap.add_argument(
        "--total-frames",
        type=int,
        default=None,
        help="Override training.total_frames (run length in env steps). "
        "Default None keeps the YAML value (500M in production). Lower "
        "to min_sampling_size + 20000 or so for shakedown runs.",
    )
    ap.add_argument(
        "--checkpoint-every-grad-steps",
        type=int,
        default=None,
        help="Override logging.checkpoint_every_grad_steps. Default None "
        "keeps the YAML value (10000). Lower for shakedowns to exercise "
        "the periodic-ckpt + rotation paths within the run window.",
    )
    ap.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint .pt produced by a previous run "
        "(restores online/target/optimizer/counters; replay is re-warmed).",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override the auto-generated run_name (used as wandb run ID + "
        "CSV log filename). If omitted and --resume is given, run_name is "
        "inferred from the checkpoint filename so logs + ckpts stay in the "
        "same namespace across resumes.",
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
    if args.num_envs is not None:
        if args.num_envs < 1:
            print(f"error: --num-envs must be >= 1, got {args.num_envs}", file=sys.stderr)
            return 1
        cfg.num_envs = args.num_envs
    if args.min_sampling_size is not None:
        if args.min_sampling_size < 1:
            print(
                f"error: --min-sampling-size must be >= 1, got {args.min_sampling_size}",
                file=sys.stderr,
            )
            return 1
        cfg.min_sampling_size = args.min_sampling_size
    if args.total_frames is not None:
        if args.total_frames < 1:
            print(
                f"error: --total-frames must be >= 1, got {args.total_frames}",
                file=sys.stderr,
            )
            return 1
        cfg.total_frames = args.total_frames
    if args.checkpoint_every_grad_steps is not None:
        if args.checkpoint_every_grad_steps < 0:
            print(
                f"error: --checkpoint-every-grad-steps must be >= 0, got "
                f"{args.checkpoint_every_grad_steps}",
                file=sys.stderr,
            )
            return 1
        cfg.checkpoint_every_grad_steps = args.checkpoint_every_grad_steps

    if args.testing:
        logging.getLogger(__name__).info(
            "testing mode active — total_frames=%d, batch_size=%d, replay=%d",
            cfg.total_frames, cfg.batch_size, cfg.replay_size,
        )

    if args.resume is not None and not args.resume.exists():
        print(f"error: resume checkpoint {args.resume} not found", file=sys.stderr)
        return 1

    train(cfg, resume_from=args.resume, run_name=args.run_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
