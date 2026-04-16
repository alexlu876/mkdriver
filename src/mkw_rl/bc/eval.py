"""BC offline evaluation: metrics + side-by-side overlay video.

Per MKW_RL_SPEC.md §2.4.

Offline metrics:
* steering top-1 and top-3 accuracy over 21 bins
* per-button F1 (accelerate, brake, drift, item)
* joint prediction accuracy ("all buttons correct AND steering bin within ±1")

Side-by-side video (the human-readable sanity check):
* Left pane: ground-truth controller state overlay (from the held-out demo).
* Right pane: predicted controller state overlay (from the BC model running
  on the same frames, with carried hidden state across the full demo).

The policy runs with carried hidden state — not reset per TBPTT window —
to match how it behaves at inference time in the eventual Phase 4 env.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mkw_rl.bc.model import BCPolicy
from mkw_rl.dtm.action_encoding import decode_steering, encode_steering
from mkw_rl.dtm.frames import load_frame
from mkw_rl.dtm.pairing import PairedSample
from mkw_rl.dtm.parser import ControllerState
from mkw_rl.dtm.viz import render_overlay

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    n_samples: int
    steering_top1: float
    steering_top3: float
    per_button_f1: dict[str, float]
    joint_accuracy: float  # all buttons correct AND steering within ±1 bin

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "steering_top1": self.steering_top1,
            "steering_top3": self.steering_top3,
            "joint_accuracy": self.joint_accuracy,
            **{f"button_f1_{k}": v for k, v in self.per_button_f1.items()},
        }


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def compute_metrics(
    steering_logits: torch.Tensor,  # (N, n_bins)
    button_logits: dict[str, torch.Tensor],  # each (N,)
    gt_steering_bins: torch.Tensor,  # (N,) long
    gt_buttons: dict[str, torch.Tensor],  # each (N,) float
) -> EvalMetrics:
    n = gt_steering_bins.shape[0]

    top1 = (steering_logits.argmax(dim=-1) == gt_steering_bins).float().mean().item()
    _, top3_idxs = steering_logits.topk(3, dim=-1)
    top3 = (top3_idxs == gt_steering_bins.unsqueeze(-1)).any(dim=-1).float().mean().item()

    button_names = ("accelerate", "brake", "drift", "item")
    f1s: dict[str, float] = {}
    buttons_all_correct = torch.ones(n, dtype=torch.bool)
    for name in button_names:
        pred = (button_logits[name] > 0).long()
        gt = gt_buttons[name].long()
        buttons_all_correct &= pred == gt
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        f1s[name] = _f1_from_counts(tp, fp, fn)

    steer_pred = steering_logits.argmax(dim=-1)
    steer_close = (steer_pred - gt_steering_bins).abs() <= 1
    joint = (buttons_all_correct & steer_close).float().mean().item()

    return EvalMetrics(
        n_samples=n,
        steering_top1=top1,
        steering_top3=top3,
        per_button_f1=f1s,
        joint_accuracy=joint,
    )


# ---------------------------------------------------------------------------
# Inference over a full demo with carried hidden state.
# ---------------------------------------------------------------------------


def _frame_stack_for(
    samples: list[PairedSample],
    t: int,
    stack_size: int,
    frame_skip: int,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """Build the (stack, H, W) float frame stack for timestep ``t`` of a demo."""
    H = frame_size[1]
    W = frame_size[0]
    stack = np.empty((stack_size, H, W), dtype=np.float32)
    for s in range(stack_size):
        offset = (stack_size - 1 - s) * frame_skip
        src_t = max(0, t - offset)
        arr = load_frame(samples[src_t].frame_path, size=frame_size, grayscale=True)
        stack[s] = arr.astype(np.float32) / 255.0
    return stack


def run_model_on_demo(
    model: BCPolicy,
    samples: list[PairedSample],
    device: torch.device,
    stack_size: int = 4,
    frame_skip: int = 4,
    chunk_len: int = 32,
    frame_size: tuple[int, int] = (140, 114),
) -> dict[str, torch.Tensor]:
    """Run the BC model across a full demo with carried hidden state.

    Returns:
        dict with tensors each shape (N,) where N = len(samples):
            steering_pred_logits: (N, n_bins)
            pred_steering_bin:    (N,) long
            pred_steering:        (N,) float, decoded bin center
            button_logits:        dict[str, (N,)]
            pred_buttons:         dict[str, (N,) bool]
    """
    model.eval()
    n = len(samples)
    all_steer_logits = []
    all_button_logits: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("accelerate", "brake", "drift", "item")
    }

    hidden = model.initial_hidden(batch_size=1, device=device)
    with torch.no_grad():
        for start in range(0, n, chunk_len):
            end = min(start + chunk_len, n)
            # Build (1, T, stack, H, W) batch.
            chunk = np.empty(
                (end - start, stack_size, frame_size[1], frame_size[0]),
                dtype=np.float32,
            )
            for t_rel, t in enumerate(range(start, end)):
                chunk[t_rel] = _frame_stack_for(samples, t, stack_size, frame_skip, frame_size)
            frames = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, T, stack, H, W)
            logits, hidden = model(frames, hidden)
            all_steer_logits.append(logits["steering"].squeeze(0).cpu())
            for name in all_button_logits:
                all_button_logits[name].append(logits[name].squeeze(0).cpu())

    steer = torch.cat(all_steer_logits, dim=0)  # (N, n_bins)
    buttons = {name: torch.cat(v, dim=0) for name, v in all_button_logits.items()}

    pred_bin = steer.argmax(dim=-1)
    pred_steering = torch.tensor([decode_steering(int(b)) for b in pred_bin.tolist()])
    pred_buttons = {name: (t > 0) for name, t in buttons.items()}

    return {
        "steering_pred_logits": steer,
        "pred_steering_bin": pred_bin,
        "pred_steering": pred_steering,
        "button_logits": buttons,
        "pred_buttons": pred_buttons,
    }


# ---------------------------------------------------------------------------
# Ground-truth extraction (convenience).
# ---------------------------------------------------------------------------


def extract_ground_truth(samples: list[PairedSample]) -> dict[str, torch.Tensor]:
    """Pull steering_bin and binary buttons out of a demo's controller states."""
    n = len(samples)
    steering_bin = torch.empty(n, dtype=torch.long)
    buttons: dict[str, torch.Tensor] = {
        name: torch.empty(n, dtype=torch.float32) for name in ("accelerate", "brake", "drift", "item")
    }
    for i, s in enumerate(samples):
        c = s.controller
        steering_bin[i] = encode_steering(c.steering)
        buttons["accelerate"][i] = float(c.accelerate)
        buttons["brake"][i] = float(c.brake)
        buttons["drift"][i] = float(c.drift)
        buttons["item"][i] = float(c.item)
    return {"steering_bin": steering_bin, "buttons": buttons}


# ---------------------------------------------------------------------------
# Side-by-side overlay video.
# ---------------------------------------------------------------------------


def _build_predicted_sample(sample: PairedSample, pred: dict[str, Any], i: int) -> PairedSample:
    """Swap a sample's controller state with the BC model's prediction."""
    c = ControllerState(
        frame_idx=sample.controller.frame_idx,
        steering=float(pred["pred_steering"][i]),
        accelerate=bool(pred["pred_buttons"]["accelerate"][i]),
        brake=bool(pred["pred_buttons"]["brake"][i]),
        drift=bool(pred["pred_buttons"]["drift"][i]),
        item=bool(pred["pred_buttons"]["item"][i]),
        look_behind=sample.controller.look_behind,  # model doesn't predict this
        _raw_analog_x=sample.controller._raw_analog_x,
        _raw_analog_y=sample.controller._raw_analog_y,
        _raw_byte0=sample.controller._raw_byte0,
        _raw_byte1=sample.controller._raw_byte1,
    )
    return PairedSample(
        frame_idx=sample.frame_idx,
        input_frame_idx=sample.input_frame_idx,
        frame_path=sample.frame_path,
        controller=c,
    )


def write_side_by_side_video(
    samples: list[PairedSample],
    predictions: dict[str, Any],
    output_path: Path | str,
    fps: int = 60,
    n_seconds: int | None = 30,
) -> Path:
    """Render an MP4 with GT overlay (left) and predicted overlay (right) per frame."""
    import imageio.v3 as iio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_max = len(samples) if n_seconds is None else min(len(samples), fps * n_seconds)
    if n_max == 0:
        raise ValueError("no samples to render")

    composed_frames: list[np.ndarray] = []
    for i in range(n_max):
        gt_sample = samples[i]
        pred_sample = _build_predicted_sample(gt_sample, predictions, i)

        gt_img = render_overlay(gt_sample).convert("RGB")
        pred_img = render_overlay(pred_sample).convert("RGB")
        # Side-by-side composite.
        w, h = gt_img.size
        composite = Image.new("RGB", (w * 2 + 4, h), color=(20, 20, 20))
        composite.paste(gt_img, (0, 0))
        composite.paste(pred_img, (w + 4, 0))
        composed_frames.append(np.asarray(composite, dtype="uint8"))

    iio.imwrite(output_path, composed_frames, fps=fps, codec="libx264", quality=7)
    return output_path
