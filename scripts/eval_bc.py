#!/usr/bin/env python3
"""BC offline eval driver.

Usage:
    uv run python scripts/eval_bc.py \\
        --checkpoint runs/bc/bc_best.pt \\
        --demo data/processed/user_demos/2026-04-16.pkl \\
        --output eval_out/ \\
        --seconds 30

Outputs:
    eval_out/metrics.json          — offline metrics dict
    eval_out/side_by_side.mp4      — GT left, predicted right
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import torch

from mkw_rl.bc.eval import (
    compute_metrics,
    extract_ground_truth,
    run_model_on_demo,
    write_side_by_side_video,
)
from mkw_rl.bc.model import BCPolicy, BCPolicyConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_bc")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to bc_best.pt or epoch checkpoint.")
    ap.add_argument(
        "--demo", type=Path, required=True, help="Pickled demo payload (from scripts/parse_demo.py)."
    )
    ap.add_argument(
        "--demo-id", type=str, default=None, help="Which demo inside the .pkl to evaluate (default: first)."
    )
    ap.add_argument("--output", type=Path, required=True, help="Output directory.")
    ap.add_argument("--seconds", type=int, default=30, help="Duration to render (-1 for full demo).")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--stack-size", type=int, default=4)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--chunk-len", type=int, default=32)
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load checkpoint. Reconstruct BCPolicyConfig from ckpt["config"] when
    # available (M-3 audit fix) so eval matches training architecture; fall
    # back to defaults for older checkpoints that didn't save config.
    log.info("loading checkpoint %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg: BCPolicyConfig
    saved_cfg = ckpt.get("config")
    if isinstance(saved_cfg, dict) and "model_config" in saved_cfg:
        mc = saved_cfg["model_config"]
        try:
            model_cfg = BCPolicyConfig(
                stack_size=mc.get("stack_size", args.stack_size),
                input_hw=tuple(mc.get("input_hw", (114, 140))),
                encoder_channels=tuple(mc.get("encoder_channels", (16, 32, 32))),
                feature_dim=mc.get("feature_dim", 256),
                lstm_hidden=mc.get("lstm_hidden", 512),
                lstm_layers=mc.get("lstm_layers", 1),
                n_steering_bins=mc.get("n_steering_bins", 21),
            )
            log.info("restored BCPolicyConfig from checkpoint: %s", model_cfg)
        except (TypeError, ValueError) as exc:
            log.warning("could not restore model config from checkpoint (%s); using defaults", exc)
            model_cfg = BCPolicyConfig(stack_size=args.stack_size)
    else:
        log.warning(
            "checkpoint has no model_config — using CLI defaults. "
            "If load_state_dict fails, the checkpoint was trained with non-default architecture."
        )
        model_cfg = BCPolicyConfig(stack_size=args.stack_size)
    model = BCPolicy(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Load demo.
    with args.demo.open("rb") as f:
        payload = pickle.load(f)
    sbd = payload["samples_by_demo"]
    demo_id = args.demo_id or next(iter(sbd))
    if demo_id not in sbd:
        log.error("demo_id %s not in %s; available: %s", demo_id, args.demo, list(sbd.keys()))
        return 1
    samples = sbd[demo_id]
    log.info("evaluating demo %s (%d samples)", demo_id, len(samples))

    # Run model over the full demo with carried hidden state.
    preds = run_model_on_demo(
        model,
        samples,
        device=device,
        stack_size=args.stack_size,
        frame_skip=args.frame_skip,
        chunk_len=args.chunk_len,
    )

    # Metrics.
    gt = extract_ground_truth(samples)
    metrics = compute_metrics(
        steering_logits=preds["steering_pred_logits"],
        button_logits=preds["button_logits"],
        gt_steering_bins=gt["steering_bin"],
        gt_buttons=gt["buttons"],
    )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics.as_dict(), f, indent=2)
    log.info("wrote metrics → %s", metrics_path)
    log.info("metrics: %s", json.dumps(metrics.as_dict(), indent=2))

    # Side-by-side video.
    n_seconds = None if args.seconds < 0 else args.seconds
    video_path = args.output / "side_by_side.mp4"
    log.info("rendering side-by-side video → %s", video_path)
    write_side_by_side_video(samples, preds, video_path, fps=args.fps, n_seconds=n_seconds)
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
