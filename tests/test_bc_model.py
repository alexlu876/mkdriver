"""Tests for src/mkw_rl/bc/model.py."""

from __future__ import annotations

import pytest
import torch

from mkw_rl.bc.model import (
    BCPolicy,
    BCPolicyConfig,
    ImpalaEncoder,
    _ImpalaBlock,
    _ImpalaResBlock,
    bc_loss,
)
from mkw_rl.dtm.action_encoding import N_STEERING_BINS


class TestImpalaComponents:
    def test_resblock_preserves_shape(self) -> None:
        block = _ImpalaResBlock(16)
        x = torch.randn(2, 16, 28, 28)
        assert block(x).shape == x.shape

    def test_impala_block_halves_spatial(self) -> None:
        block = _ImpalaBlock(in_channels=3, out_channels=16)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        # MaxPool(3, stride=2, padding=1) on 32 → 16.
        assert out.shape == (2, 16, 16, 16)


class TestEncoder:
    def test_encoder_output_shape(self) -> None:
        enc = ImpalaEncoder(in_channels=4, feature_dim=256, input_hw=(114, 140))
        x = torch.randn(3, 4, 114, 140)
        out = enc(x)
        assert out.shape == (3, 256)

    def test_encoder_batch_invariance(self) -> None:
        enc = ImpalaEncoder(in_channels=4, feature_dim=256, input_hw=(114, 140))
        enc.eval()
        x = torch.randn(5, 4, 114, 140)
        with torch.no_grad():
            out_full = enc(x)
            out_split = torch.cat([enc(x[:2]), enc(x[2:])], dim=0)
        assert torch.allclose(out_full, out_split, atol=1e-5)


class TestBCPolicyForward:
    def _model(self) -> BCPolicy:
        return BCPolicy(BCPolicyConfig())

    def test_output_shapes(self) -> None:
        model = self._model()
        model.eval()
        B, T = 2, 8
        x = torch.rand(B, T, 4, 114, 140)
        with torch.no_grad():
            logits, (h, c) = model(x)
        assert logits["steering"].shape == (B, T, N_STEERING_BINS)
        for name in ("accelerate", "brake", "drift", "item"):
            assert logits[name].shape == (B, T)
        assert h.shape == (1, B, 512)
        assert c.shape == (1, B, 512)

    def test_hidden_state_roundtrip_determinism(self) -> None:
        """Same input + same starting hidden → same output; second call with
        returned hidden is deterministic across a split sequence."""
        model = self._model()
        model.eval()
        B, T = 2, 8
        x_full = torch.rand(B, T, 4, 114, 140)

        with torch.no_grad():
            # One-shot forward.
            logits_full, hidden_full = model(x_full)
            # Split forward: T/2 + T/2, carrying hidden across.
            logits_a, hidden_mid = model(x_full[:, : T // 2])
            logits_b, hidden_end = model(x_full[:, T // 2 :], hidden_mid)

        # Logits over the full sequence should equal the concatenation of split logits.
        stacked = {k: torch.cat([logits_a[k], logits_b[k]], dim=1) for k in logits_full}
        for k, v in stacked.items():
            assert torch.allclose(logits_full[k], v, atol=1e-5), f"mismatch on {k}"
        # Final hidden state should match.
        assert torch.allclose(hidden_full[0], hidden_end[0], atol=1e-5)
        assert torch.allclose(hidden_full[1], hidden_end[1], atol=1e-5)

    def test_param_count_under_5m(self) -> None:
        model = self._model()
        n = model.param_count()
        assert n < 5_000_000, f"parameter count {n} > 5M"
        # Also confirm it's not absurdly small — there should be a real model.
        assert n > 500_000

    def test_wrong_input_shape_raises(self) -> None:
        model = self._model()
        with pytest.raises(ValueError):
            model(torch.randn(2, 8, 4, 114))  # missing W
        with pytest.raises(ValueError):
            # Wrong stack size.
            cfg = BCPolicyConfig(stack_size=4)
            m = BCPolicy(cfg)
            m(torch.randn(2, 8, 3, 114, 140))

    def test_initial_hidden_shape(self) -> None:
        model = self._model()
        h, c = model.initial_hidden(batch_size=7)
        assert h.shape == (1, 7, 512)
        assert c.shape == (1, 7, 512)
        assert torch.all(h == 0)


class TestBCLoss:
    def test_loss_forward_and_backward(self) -> None:
        model = BCPolicy(BCPolicyConfig())
        B, T = 2, 8
        x = torch.rand(B, T, 4, 114, 140)
        logits, _ = model(x)
        targets = {
            "steering_bin": torch.randint(0, N_STEERING_BINS, (B, T)),
            "accelerate": torch.randint(0, 2, (B, T)).float(),
            "brake": torch.randint(0, 2, (B, T)).float(),
            "drift": torch.randint(0, 2, (B, T)).float(),
            "item": torch.randint(0, 2, (B, T)).float(),
        }
        losses = bc_loss(logits, targets)
        for key in ("total", "steering", "buttons"):
            assert losses[key].ndim == 0
        # Backward works.
        losses["total"].backward()
        # At least the steering head gets gradient.
        assert model.steering_head.weight.grad is not None
        assert model.steering_head.weight.grad.abs().sum() > 0

    def test_loss_weights_zero_means_no_contribution(self) -> None:
        model = BCPolicy(BCPolicyConfig())
        B, T = 1, 2
        x = torch.rand(B, T, 4, 114, 140)
        logits, _ = model(x)
        targets = {
            "steering_bin": torch.zeros(B, T, dtype=torch.long),
            "accelerate": torch.zeros(B, T),
            "brake": torch.zeros(B, T),
            "drift": torch.zeros(B, T),
            "item": torch.zeros(B, T),
        }
        # With button_weight=0, total should equal steering CE.
        losses = bc_loss(logits, targets, steering_weight=1.0, button_weight=0.0)
        assert torch.allclose(losses["total"], losses["steering"])


class TestGradientFlow:
    """Critical diagnostic from spec §2.3: LSTM gradient norm must be non-trivial."""

    def test_lstm_gets_gradient(self) -> None:
        model = BCPolicy(BCPolicyConfig())
        B, T = 2, 4
        x = torch.rand(B, T, 4, 114, 140)
        logits, _ = model(x)
        targets = {
            "steering_bin": torch.randint(0, N_STEERING_BINS, (B, T)),
            "accelerate": torch.zeros(B, T),
            "brake": torch.zeros(B, T),
            "drift": torch.zeros(B, T),
            "item": torch.zeros(B, T),
        }
        bc_loss(logits, targets)["total"].backward()
        lstm_grad_norm = sum(p.grad.norm().item() for p in model.lstm.parameters() if p.grad is not None)
        assert lstm_grad_norm > 1e-4, f"LSTM grad norm {lstm_grad_norm} < 1e-4"
