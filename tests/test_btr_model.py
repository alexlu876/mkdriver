"""Unit tests for BTR policy model (pass 2).

Pure CPU. Covers shape invariants, hidden-state round-trip determinism
(same input + hidden = same output), noise on/off effects, gradient flow
through encoder + LSTM + heads, advantages_only short-circuit, and param
count sanity.

Does NOT exercise the replay buffer, training loop, or live env — those
come in passes 3-6.
"""

from __future__ import annotations

import pytest
import torch

from mkw_rl.rl.model import BTRConfig, BTRPolicy
from mkw_rl.rl.networks import FactorizedNoisyLinear


def _tiny_cfg(**overrides) -> BTRConfig:
    """Smaller model for fast CPU tests. Production defaults stay in BTRConfig()."""
    defaults = dict(
        n_actions=40,
        stack_size=4,
        input_hw=(75, 140),
        feature_dim=64,
        lstm_hidden=64,
        linear_size=64,
        num_tau=8,
        n_cos=64,
        layer_norm=True,
    )
    defaults.update(overrides)
    return BTRConfig(**defaults)


# ---------------------------------------------------------------------------
# Construction + parameter count.
# ---------------------------------------------------------------------------


class TestBTRConstruction:
    def test_constructs_with_default_config(self) -> None:
        policy = BTRPolicy()
        assert policy.cfg.n_actions == 40
        assert policy.cfg.lstm_hidden == 512
        assert policy.cfg.input_hw == (75, 140)

    def test_constructs_with_custom_config(self) -> None:
        cfg = _tiny_cfg(n_actions=21)
        policy = BTRPolicy(cfg)
        assert policy.cfg.n_actions == 21

    def test_param_count_reasonable(self) -> None:
        """Default config should land in the ~1-5M params range per v2 spec
        (paper's ImpalaCNN is ~1M; LSTM + IQN heads add a few)."""
        policy = BTRPolicy()
        n = policy.param_count()
        assert 1_000_000 <= n <= 10_000_000, f"unexpected param count: {n}"

    def test_has_factorized_noisy_linears(self) -> None:
        """Dueling heads should contain NoisyLinear layers (for noisy-nets exploration)."""
        policy = BTRPolicy(_tiny_cfg())
        noisy_layers = [m for m in policy.modules() if isinstance(m, FactorizedNoisyLinear)]
        # 2 branches × 2 NoisyLinear each = 4.
        assert len(noisy_layers) == 4


# ---------------------------------------------------------------------------
# Forward pass shapes.
# ---------------------------------------------------------------------------


class TestBTRForwardShapes:
    def test_output_shapes_for_single_timestep(self) -> None:
        cfg = _tiny_cfg()
        policy = BTRPolicy(cfg)
        policy.eval()
        B, T = 2, 1
        x = torch.rand(B, T, cfg.stack_size, *cfg.input_hw) * 255
        with torch.no_grad():
            quantiles, taus, (h, c) = policy(x)
        assert quantiles.shape == (B, T, cfg.num_tau, cfg.n_actions)
        assert taus.shape == (B, T, cfg.num_tau, 1)
        assert h.shape == (cfg.lstm_layers, B, cfg.lstm_hidden)
        assert c.shape == (cfg.lstm_layers, B, cfg.lstm_hidden)

    def test_output_shapes_for_multi_timestep(self) -> None:
        cfg = _tiny_cfg()
        policy = BTRPolicy(cfg)
        policy.eval()
        B, T = 3, 16
        x = torch.rand(B, T, cfg.stack_size, *cfg.input_hw) * 255
        with torch.no_grad():
            quantiles, _, (h, c) = policy(x)
        assert quantiles.shape == (B, T, cfg.num_tau, cfg.n_actions)
        assert h.shape == (cfg.lstm_layers, B, cfg.lstm_hidden)

    def test_q_values_shape(self) -> None:
        cfg = _tiny_cfg()
        policy = BTRPolicy(cfg)
        policy.eval()
        B, T = 2, 4
        x = torch.rand(B, T, cfg.stack_size, *cfg.input_hw)
        with torch.no_grad():
            q, _ = policy.q_values(x)
        # q_values averages over the num_tau axis.
        assert q.shape == (B, T, cfg.n_actions)

    def test_accepts_uint8_and_float_inputs(self) -> None:
        """Internal /255 normalization should produce the same result whether
        the input is uint8 or float in [0, 255]."""
        cfg = _tiny_cfg()
        policy = BTRPolicy(cfg)
        policy.eval()
        policy.disable_noise()  # determinism for the comparison
        B, T = 1, 2
        # Use seed to match the two IQN τ samples across runs.
        torch.manual_seed(42)
        x_uint = torch.randint(0, 256, (B, T, cfg.stack_size, *cfg.input_hw), dtype=torch.uint8)
        x_float = x_uint.float()
        with torch.no_grad():
            torch.manual_seed(0)
            q_u, _ = policy.q_values(x_uint)
            torch.manual_seed(0)
            q_f, _ = policy.q_values(x_float)
        assert torch.allclose(q_u, q_f, atol=1e-5)

    def test_wrong_input_shape_raises(self) -> None:
        policy = BTRPolicy(_tiny_cfg())
        # Missing time dim.
        with pytest.raises(ValueError, match="expected frames shape"):
            policy(torch.rand(2, 4, 75, 140))

    def test_wrong_stack_size_raises(self) -> None:
        policy = BTRPolicy(_tiny_cfg(stack_size=4))
        with pytest.raises(ValueError, match="stack size mismatch"):
            policy(torch.rand(2, 8, 3, 75, 140))


# ---------------------------------------------------------------------------
# Hidden state round-trip — the LSTM is stateful; split-and-carry must match joint.
# ---------------------------------------------------------------------------


class TestHiddenRoundTrip:
    def test_split_matches_joint(self) -> None:
        """Forward the full sequence in one pass vs in two halves carrying hidden
        state across. Noise is off so the only source of randomness is the IQN
        τ sampling — we compare the LSTM output hidden state (deterministic)."""
        cfg = _tiny_cfg()
        policy = BTRPolicy(cfg)
        policy.eval()
        policy.disable_noise()
        B, T = 2, 8
        x_full = torch.rand(B, T, cfg.stack_size, *cfg.input_hw) * 255

        with torch.no_grad():
            _, _, hidden_full = policy(x_full)
            _, _, hidden_mid = policy(x_full[:, : T // 2])
            _, _, hidden_end = policy(x_full[:, T // 2 :], hidden_mid)

        # Final hidden state should match bit-for-bit (LSTM math is deterministic).
        assert torch.allclose(hidden_full[0], hidden_end[0], atol=1e-5)
        assert torch.allclose(hidden_full[1], hidden_end[1], atol=1e-5)

    def test_initial_hidden_shape(self) -> None:
        policy = BTRPolicy(_tiny_cfg())
        h, c = policy.initial_hidden(batch_size=5)
        assert h.shape == (1, 5, 64)
        assert c.shape == (1, 5, 64)
        assert torch.all(h == 0)
        assert torch.all(c == 0)


# ---------------------------------------------------------------------------
# Noise + determinism.
# ---------------------------------------------------------------------------


class TestNoise:
    def test_disable_noise_gives_deterministic_heads(self) -> None:
        """With noise disabled and IQN τ seeded, two forward passes match exactly."""
        cfg = _tiny_cfg()
        policy = BTRPolicy(cfg)
        policy.eval()
        policy.disable_noise()
        B, T = 1, 1
        x = torch.rand(B, T, cfg.stack_size, *cfg.input_hw) * 255
        with torch.no_grad():
            torch.manual_seed(7)
            q1, _, _ = policy(x)
            torch.manual_seed(7)
            q2, _, _ = policy(x)
        assert torch.equal(q1, q2)

    def test_reset_noise_changes_output(self) -> None:
        """With noise enabled, two forwards with different ε should differ."""
        cfg = _tiny_cfg()
        policy = BTRPolicy(cfg)
        policy.eval()
        B, T = 1, 1
        x = torch.rand(B, T, cfg.stack_size, *cfg.input_hw) * 255
        with torch.no_grad():
            torch.manual_seed(0)
            policy.reset_noise()
            torch.manual_seed(99)  # seed the IQN τ sampling
            q1, _, _ = policy(x)
            torch.manual_seed(1)
            policy.reset_noise()
            torch.manual_seed(99)
            q2, _, _ = policy(x)
        # Different noise → different output.
        assert not torch.equal(q1, q2)


# ---------------------------------------------------------------------------
# Gradient flow.
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradients_flow_through_all_components(self) -> None:
        cfg = _tiny_cfg()
        policy = BTRPolicy(cfg)
        policy.train()
        B, T = 2, 4
        x = torch.rand(B, T, cfg.stack_size, *cfg.input_hw) * 255

        quantiles, _, _ = policy(x)
        # Select one action per (B, T, τ) and sum. Using .sum() over all actions
        # would zero out the advantage branch's gradient because
        # sum_a (V + A - mean(A)) = n_actions * V, so advantages drop out —
        # a known dueling-network quirk. A single-action (or gather) loss
        # keeps both branches in the graph.
        loss = quantiles[..., 0].sum()
        loss.backward()

        # Encoder.
        encoder_grads = [
            p.grad for p in policy.encoder.parameters() if p.grad is not None
        ]
        assert any(g.abs().sum() > 0 for g in encoder_grads)
        # LSTM.
        lstm_grads = [p.grad for p in policy.lstm.parameters() if p.grad is not None]
        assert any(g.abs().sum() > 0 for g in lstm_grads)
        # Dueling heads — every FactorizedNoisyLinear's μ should get gradient.
        head_mu_grads = [
            m.weight_mu.grad
            for m in policy.modules()
            if isinstance(m, FactorizedNoisyLinear)
        ]
        assert all(g is not None and g.abs().sum() > 0 for g in head_mu_grads)


# ---------------------------------------------------------------------------
# advantages_only short-circuit.
# ---------------------------------------------------------------------------


class TestAdvantagesOnly:
    def test_argmax_matches_q_argmax(self) -> None:
        """For action selection we can skip the value branch — Dueling's advantages_only.
        The argmax of advantages should equal argmax of full Q (since Q = V + A - mean(A)
        adds a constant per sample to all actions)."""
        cfg = _tiny_cfg()
        policy = BTRPolicy(cfg)
        policy.eval()
        policy.disable_noise()
        B, T = 3, 1
        x = torch.rand(B, T, cfg.stack_size, *cfg.input_hw) * 255
        with torch.no_grad():
            torch.manual_seed(5)
            q_full, _, _ = policy(x, advantages_only=False)
            torch.manual_seed(5)
            q_adv, _, _ = policy(x, advantages_only=True)
        # Argmax over actions should agree (after averaging over τ).
        assert torch.equal(
            q_full.mean(dim=2).argmax(dim=-1),
            q_adv.mean(dim=2).argmax(dim=-1),
        )
