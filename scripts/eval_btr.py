#!/usr/bin/env python3
"""BTR online evaluation driver.

Loads a trained BTR checkpoint and rolls the policy against live Dolphin
with noisy-nets disabled (deterministic greedy). Useful for:

- Smoke-testing a checkpoint without affecting the running training job.
- Generating a return curve vs. checkpoint number to visualize learning.
- Comparing runs/checkpoints on a fixed track.

Does NOT touch the replay buffer or the optimizer — pure inference.

Usage
-----

Basic eval, 5 episodes on Luigi Circuit with CUDA::

    .venv/bin/python scripts/eval_btr.py \\
        --ckpt runs/btr/luigi_4env_20260423_022252_grad10000.pt \\
        --config configs/btr.yaml \\
        --track-slug luigi_circuit_tt \\
        --episodes 5 \\
        --device cuda

Write metrics to JSON for a return curve::

    .venv/bin/python scripts/eval_btr.py \\
        --ckpt runs/btr/luigi_4env_20260423_022252_grad20000.pt \\
        --config configs/btr.yaml \\
        --track-slug luigi_circuit_tt \\
        --episodes 3 \\
        --device cuda \\
        --output eval_out/grad20000.json

Running this while training is IN PROGRESS is supported but will share
the GPU with the trainer — expect both to run slower. Typically you
checkpoint-evaluate off the most recent ``_grad{N}.pt`` between training
sessions rather than concurrently.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--ckpt", type=Path, required=True,
        help="Path to a BTR checkpoint (e.g. *_grad10000.pt, *_final.pt).",
    )
    ap.add_argument(
        "--config", type=Path, default=Path("configs/btr.yaml"),
        help="Training config YAML (default: configs/btr.yaml). Used for "
        "env paths + model architecture — must match how the ckpt was trained.",
    )
    ap.add_argument(
        "--track-slug", type=str, required=True,
        help="Track to evaluate on (e.g. luigi_circuit_tt). Must be in the "
        "config's track_metadata.yaml + have a matching savestate on disk.",
    )
    ap.add_argument(
        "--episodes", type=int, default=5,
        help="Number of eval episodes to run (default: 5).",
    )
    ap.add_argument(
        "--device", type=str, default=None,
        help="Override runtime.device from config (cpu/cuda/mps).",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Optional: write per-episode results + summary stats as JSON here.",
    )
    ap.add_argument(
        "--env-id", type=int, default=0,
        help="Dolphin env_id (socket port offset 26330 + env_id). Default 0. "
        "Set to a non-zero value if evaluating alongside a live training run "
        "on the same box so the socket port doesn't collide.",
    )
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"error: checkpoint {args.ckpt} not found", file=sys.stderr)
        return 1
    if not args.config.exists():
        print(f"error: config {args.config} not found", file=sys.stderr)
        return 1

    # Deferred imports — keep logging.basicConfig effective before any mkw_rl
    # module sets up its own logger.
    from mkw_rl.rl.train import (
        BTRAgent,
        load_checkpoint,
        load_config,
        run_one_episode,
        _cleanup_stale_x11_state,
        _make_env,
    )

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device
    # Eval is always single-env — the multi-env parallelism is a training
    # throughput concern, not an eval one. Force num_envs=1 regardless of
    # what the YAML says.
    cfg.num_envs = 1
    cfg.env_id = args.env_id

    log.info("loading checkpoint %s", args.ckpt)
    agent = BTRAgent.build(cfg)
    load_checkpoint(agent, args.ckpt)
    log.info(
        "loaded: env_steps=%d grad_steps=%d nonfinite_streak=%d",
        agent.env_steps, agent.grad_steps, agent.nonfinite_streak,
    )

    # Disable noisy-nets exploration for deterministic greedy rollouts.
    # Without this, each forward pass re-samples noise and eval numbers
    # become stochastic run-to-run even at the same checkpoint. The
    # deterministic=True flag on act() prevents run_one_episode's rollout
    # loop from re-enabling noise via reset_noise() on every step.
    agent.online_net.disable_noise()
    agent.online_net.eval()

    _cleanup_stale_x11_state()  # Linux-only, harmless elsewhere.
    env = _make_env(cfg, env_id=args.env_id)

    returns: list[float] = []
    lengths: list[int] = []
    breakdowns: list[dict[str, float]] = []
    per_episode: list[dict] = []

    t0 = time.time()
    try:
        for i in range(args.episodes):
            log.info(
                "eval episode %d/%d on %s",
                i + 1, args.episodes, args.track_slug,
            )
            ep_return, rb_sums, n_steps = run_one_episode(
                agent, env, args.track_slug,
                skip_learn=True, deterministic=True,
            )
            returns.append(float(ep_return))
            lengths.append(int(n_steps))
            breakdowns.append({k: float(v) for k, v in rb_sums.items()})
            log.info(
                "ep %d: return=%.2f length=%d components=%s",
                i + 1, ep_return, n_steps,
                ", ".join(f"{k}={v:.2f}" for k, v in rb_sums.items()),
            )
            per_episode.append({
                "episode": i + 1,
                "return": float(ep_return),
                "length": int(n_steps),
                "reward_breakdown": {k: float(v) for k, v in rb_sums.items()},
            })
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            log.exception("error closing env")

    dt = time.time() - t0
    returns_arr = np.array(returns)
    # Mean per-component reward across all episodes.
    comp_sums: dict[str, list[float]] = {}
    for rb in breakdowns:
        for k, v in rb.items():
            comp_sums.setdefault(k, []).append(v)
    comp_means = {k: float(np.mean(v)) for k, v in comp_sums.items()}

    summary = {
        "ckpt": str(args.ckpt),
        "track_slug": args.track_slug,
        "episodes": len(returns),
        "return_mean": float(returns_arr.mean()),
        "return_std": float(returns_arr.std()),
        "return_min": float(returns_arr.min()),
        "return_max": float(returns_arr.max()),
        "length_mean": float(np.mean(lengths)),
        "reward_component_means": comp_means,
        "wall_time_s": dt,
        "agent_env_steps": int(agent.env_steps),
        "agent_grad_steps": int(agent.grad_steps),
    }

    log.info("=" * 60)
    log.info(
        "eval summary on %s over %d episodes (%.1fs wall)",
        args.track_slug, len(returns), dt,
    )
    log.info(
        "  return: mean=%.2f ± %.2f  min=%.2f  max=%.2f",
        summary["return_mean"], summary["return_std"],
        summary["return_min"], summary["return_max"],
    )
    log.info("  length mean: %.1f env steps", summary["length_mean"])
    log.info("  reward components (mean per episode):")
    for comp, val in comp_means.items():
        log.info("    %-20s %+.3f", comp, val)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "per_episode": per_episode}
        args.output.write_text(json.dumps(payload, indent=2))
        log.info("wrote eval results → %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
