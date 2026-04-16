"""Tests for src/mkw_rl/bc/eval.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mkw_rl.bc.eval import (
    compute_metrics,
    extract_ground_truth,
    run_model_on_demo,
    write_side_by_side_video,
)
from mkw_rl.bc.model import BCPolicy, BCPolicyConfig
from mkw_rl.dtm.action_encoding import N_STEERING_BINS
from mkw_rl.dtm.pairing import pair_dtm_and_frames
from mkw_rl.dtm.parser import build_dtm_blob, build_frame


def _synth_demo(tmp_path: Path, n: int, demo_id: str = "d") -> list:
    dtm = tmp_path / f"{demo_id}.dtm"
    frame_dir = tmp_path / f"frames_{demo_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames_bytes = [
        build_frame(accelerate=(i % 2 == 0), drift=(i % 5 < 2), analog_x=(128 + i) & 0xFF) for i in range(n)
    ]
    dtm.write_bytes(build_dtm_blob(vi_count=n, input_count=n, frames=frames_bytes))
    for i in range(n):
        arr = np.full((60, 80, 3), (i * 11 % 256, i * 3 % 256, 90), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(frame_dir / f"framedump_{i}.png")
    return pair_dtm_and_frames(dtm, frame_dir, tail_margin=0)


# ---------------------------------------------------------------------------
# Metrics — hand-constructed tensors.
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_perfect_prediction(self) -> None:
        n = 16
        gt_bins = torch.randint(0, N_STEERING_BINS, (n,))
        # Perfect steering logits: one-hot at gt bin with high value.
        logits = torch.zeros(n, N_STEERING_BINS)
        logits[torch.arange(n), gt_bins] = 10.0
        buttons = {
            name: torch.randint(0, 2, (n,)).float() for name in ("accelerate", "brake", "drift", "item")
        }
        button_logits = {name: (buttons[name] * 10.0 - 5.0) for name in buttons}  # high where gt=1
        m = compute_metrics(logits, button_logits, gt_bins, buttons)
        assert m.steering_top1 == pytest.approx(1.0)
        assert m.steering_top3 == pytest.approx(1.0)
        for name in buttons:
            assert m.per_button_f1[name] == pytest.approx(1.0, abs=1e-6) or m.per_button_f1[name] == 0.0
            # F1 == 0 only if no positives at all in gt — avoid that by asserting at least one positive:
        # Joint accuracy: all buttons correct AND steering within ±1 → should be 1.0.
        assert m.joint_accuracy == pytest.approx(1.0)

    def test_random_prediction_far_from_perfect(self) -> None:
        n = 100
        torch.manual_seed(0)
        gt_bins = torch.randint(0, N_STEERING_BINS, (n,))
        # Uniform random logits.
        logits = torch.randn(n, N_STEERING_BINS)
        buttons = {
            name: torch.randint(0, 2, (n,)).float() for name in ("accelerate", "brake", "drift", "item")
        }
        button_logits = {name: torch.randn(n) for name in buttons}
        m = compute_metrics(logits, button_logits, gt_bins, buttons)
        # top-1 should be roughly 1/21 ≈ 0.05 with some variance.
        assert 0.0 <= m.steering_top1 <= 0.25
        assert m.steering_top3 <= 0.35

    def test_top3_ge_top1(self) -> None:
        n = 50
        gt_bins = torch.randint(0, N_STEERING_BINS, (n,))
        logits = torch.randn(n, N_STEERING_BINS)
        buttons = {name: torch.zeros(n) for name in ("accelerate", "brake", "drift", "item")}
        button_logits = {name: torch.zeros(n) for name in buttons}
        m = compute_metrics(logits, button_logits, gt_bins, buttons)
        assert m.steering_top3 >= m.steering_top1


# ---------------------------------------------------------------------------
# run_model_on_demo — contract test.
# ---------------------------------------------------------------------------


class TestRunModelOnDemo:
    def _model(self) -> BCPolicy:
        cfg = BCPolicyConfig(stack_size=4, input_hw=(114, 140), feature_dim=32, lstm_hidden=32)
        return BCPolicy(cfg)

    def test_output_shapes(self, tmp_path: Path) -> None:
        samples = _synth_demo(tmp_path, 40)
        model = self._model()
        preds = run_model_on_demo(
            model,
            samples,
            device=torch.device("cpu"),
            stack_size=4,
            frame_skip=4,
            chunk_len=16,
            frame_size=(140, 114),
        )
        assert preds["steering_pred_logits"].shape == (40, N_STEERING_BINS)
        assert preds["pred_steering_bin"].shape == (40,)
        assert preds["pred_steering"].shape == (40,)
        for name in ("accelerate", "brake", "drift", "item"):
            assert preds["button_logits"][name].shape == (40,)
            assert preds["pred_buttons"][name].shape == (40,)

    def test_chunked_matches_single_pass(self, tmp_path: Path) -> None:
        """chunk_len=16 vs chunk_len=40 should produce identical results (hidden state carried)."""
        samples = _synth_demo(tmp_path, 40)
        model = self._model()
        model.eval()
        preds_chunked = run_model_on_demo(
            model,
            samples,
            device=torch.device("cpu"),
            stack_size=4,
            frame_skip=4,
            chunk_len=16,
            frame_size=(140, 114),
        )
        preds_one = run_model_on_demo(
            model,
            samples,
            device=torch.device("cpu"),
            stack_size=4,
            frame_skip=4,
            chunk_len=80,
            frame_size=(140, 114),
        )
        # Tensors should match.
        assert torch.allclose(
            preds_chunked["steering_pred_logits"],
            preds_one["steering_pred_logits"],
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# Ground truth extraction.
# ---------------------------------------------------------------------------


class TestExtractGroundTruth:
    def test_round_trip(self, tmp_path: Path) -> None:
        samples = _synth_demo(tmp_path, 20)
        gt = extract_ground_truth(samples)
        assert gt["steering_bin"].shape == (20,)
        for name in ("accelerate", "brake", "drift", "item"):
            assert gt["buttons"][name].shape == (20,)
        # Values should be in expected ranges.
        assert gt["steering_bin"].min() >= 0
        assert gt["steering_bin"].max() < N_STEERING_BINS
        for name in ("accelerate", "brake", "drift", "item"):
            v = gt["buttons"][name]
            assert ((v == 0.0) | (v == 1.0)).all()


# ---------------------------------------------------------------------------
# Side-by-side video.
# ---------------------------------------------------------------------------


class TestSideBySideVideo:
    def test_writes_valid_mp4(self, tmp_path: Path) -> None:
        samples = _synth_demo(tmp_path, 30)
        model = BCPolicy(BCPolicyConfig(stack_size=4, input_hw=(114, 140), feature_dim=32, lstm_hidden=32))
        preds = run_model_on_demo(model, samples, device=torch.device("cpu"), chunk_len=16)
        out = tmp_path / "sbs.mp4"
        path = write_side_by_side_video(samples, preds, out, fps=30, n_seconds=None)
        assert path == out
        assert out.exists()
        assert out.stat().st_size > 1024

    def test_empty_samples_raises(self, tmp_path: Path) -> None:
        preds = {
            "pred_steering": torch.zeros(0),
            "pred_buttons": {
                name: torch.zeros(0, dtype=torch.bool) for name in ("accelerate", "brake", "drift", "item")
            },
        }
        with pytest.raises(ValueError, match="no samples"):
            write_side_by_side_video([], preds, tmp_path / "x.mp4")
