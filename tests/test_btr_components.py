"""Unit tests for pass-1 BTR components.

Covers:
- FactorizedNoisyLinear: correct output shape, noise on/off determinism,
  parameter count, reset_noise changes output, gradient flow.
- Dueling: Q value decomposition identity (V + A - mean(A)), advantages_only
  short-circuit.
- SumTree: append + find round-trip, prefix-sum semantics, total() correctness.
- PER: append → sample → update_priorities lifecycle, shape invariants,
  n-step discount correctness, terminal handling.

These are pure-CPU tests — no CUDA/MPS required, no env, no Dolphin. Each
one runs in <100ms. The BTR model + training-loop tests come in later passes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mkw_rl.rl.networks import Dueling, FactorizedNoisyLinear
from mkw_rl.rl.replay import PER, SumTree

# ---------------------------------------------------------------------------
# FactorizedNoisyLinear.
# ---------------------------------------------------------------------------


class TestFactorizedNoisyLinear:
    def test_output_shape(self) -> None:
        layer = FactorizedNoisyLinear(64, 32)
        x = torch.randn(5, 64)
        out = layer(x)
        assert out.shape == (5, 32)

    def test_disable_noise_is_deterministic(self) -> None:
        layer = FactorizedNoisyLinear(16, 8)
        layer.disable_noise()
        x = torch.randn(3, 16)
        # Two forward passes with noise disabled — must be identical.
        out1 = layer(x)
        out2 = layer(x)
        assert torch.equal(out1, out2)

    def test_reset_noise_changes_output(self) -> None:
        layer = FactorizedNoisyLinear(16, 8, sigma_0=0.5)
        torch.manual_seed(0)
        layer.reset_noise()
        x = torch.ones(1, 16)
        out_a = layer(x).clone()
        # Re-sample noise — output should change (unless sigma annihilates, which it doesn't at default).
        torch.manual_seed(1)
        layer.reset_noise()
        out_b = layer(x)
        assert not torch.equal(out_a, out_b)

    def test_parameter_count(self) -> None:
        # weight_mu + weight_sigma (in*out each) + bias_mu + bias_sigma (out each).
        layer = FactorizedNoisyLinear(10, 4)
        n_params = sum(p.numel() for p in layer.parameters())
        assert n_params == 2 * (10 * 4) + 2 * 4

    def test_gradient_flows(self) -> None:
        layer = FactorizedNoisyLinear(8, 4)
        x = torch.randn(2, 8, requires_grad=False)
        out = layer(x)
        out.sum().backward()
        assert layer.weight_mu.grad is not None
        assert layer.bias_mu.grad is not None
        # sigma gets grad only if noise is nonzero (it's zero post-disable_noise()).
        # Turn noise on to exercise sigma gradients:
        layer.reset_noise()
        layer.weight_mu.grad = None
        layer.weight_sigma.grad = None
        layer(x).sum().backward()
        assert layer.weight_sigma.grad is not None
        # weight_sigma grad shouldn't be all zeros after a real noisy forward.
        assert layer.weight_sigma.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Dueling.
# ---------------------------------------------------------------------------


class TestDueling:
    def _make(self, feat_dim: int = 8, n_actions: int = 4) -> Dueling:
        value_branch = torch.nn.Linear(feat_dim, 1)
        advantage_branch = torch.nn.Linear(feat_dim, n_actions)
        return Dueling(value_branch, advantage_branch)

    def test_output_shape(self) -> None:
        d = self._make(feat_dim=8, n_actions=4)
        x = torch.randn(3, 8)
        q = d(x)
        assert q.shape == (3, 4)

    def test_advantages_only_short_circuits(self) -> None:
        d = self._make(feat_dim=8, n_actions=4)
        x = torch.randn(3, 8)
        a_only = d(x, advantages_only=True)
        full_q = d(x)
        # advantages_only returns raw advantages; full_q = V + (A - mean(A))
        # so difference is V_scalar + (-mean(A)) per sample, constant across actions.
        diff = full_q - a_only
        # Each row should be constant across actions (same broadcast scalar).
        assert torch.allclose(diff - diff[:, :1], torch.zeros_like(diff), atol=1e-6)

    def test_q_decomposition_identity(self) -> None:
        """Q should equal V + A - mean(A) along the action dim."""
        d = self._make(feat_dim=8, n_actions=4)
        x = torch.randn(5, 8)
        q = d(x)
        # Recompute manually.
        with torch.no_grad():
            v = d.value_branch(x)  # (5, 1)
            a = d.advantage_branch(x)  # (5, 4)
            expected = v + (a - a.mean(dim=1, keepdim=True))
        assert torch.allclose(q, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# SumTree.
# ---------------------------------------------------------------------------


class TestSumTree:
    def test_empty_total_is_zero(self) -> None:
        st = SumTree(size=16)
        assert st.total() == 0.0

    def test_append_and_total(self) -> None:
        st = SumTree(size=8)
        for v in [1.0, 2.0, 3.0, 4.0]:
            st.append(v)
        assert st.total() == pytest.approx(10.0)

    def test_max_tracks_largest(self) -> None:
        st = SumTree(size=8)
        for v in [0.1, 5.0, 2.0]:
            st.append(v)
        assert st.max == 5.0

    def test_find_returns_correct_leaf(self) -> None:
        st = SumTree(size=4)
        # Leaves: priorities [1.0, 2.0, 3.0, 4.0], total 10.
        # _retrieve uses np.greater (strict >), so intervals are right-closed:
        # [0, 1], (1, 3], (3, 6], (6, 10]. A query exactly on a boundary goes left.
        for v in [1.0, 2.0, 3.0, 4.0]:
            st.append(v)
        query = np.array([0.5, 1.5, 4.0, 7.0], dtype=np.float32)
        prios, data_idxs, tree_idxs = st.find(query)
        # Each query should land in the leaf whose interval contains it.
        assert data_idxs[0] == 0  # 0.5 ∈ [0, 1]
        assert data_idxs[1] == 1  # 1.5 ∈ (1, 3]
        assert data_idxs[2] == 2  # 4.0 ∈ (3, 6]
        assert data_idxs[3] == 3  # 7.0 ∈ (6, 10]
        # Returned priorities should equal the leaf values.
        assert np.allclose(prios, [1.0, 2.0, 3.0, 4.0])

    def test_update_changes_sums(self) -> None:
        st = SumTree(size=4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            st.append(v)
        # Get the tree indices for each leaf.
        _, _, tree_idxs = st.find(np.array([0.5, 1.5, 4.0, 7.0], dtype=np.float32))
        # Double every priority.
        st.update(tree_idxs, np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32))
        assert st.total() == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# PER — end-to-end append/sample/update.
# ---------------------------------------------------------------------------


def _random_frame(h: int = 20, w: int = 30) -> np.ndarray:
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)


def _random_stack(framestack: int = 4, h: int = 20, w: int = 30) -> np.ndarray:
    return np.stack([_random_frame(h, w) for _ in range(framestack)])


class TestPER:
    def _make_per(self, size: int = 64, n: int = 3, envs: int = 1) -> PER:
        return PER(
            size=size,
            device="cpu",
            n=n,
            envs=envs,
            gamma=0.99,
            framestack=4,
            imagex=30,
            imagey=20,
        )

    def test_initial_capacity_zero(self) -> None:
        per = self._make_per()
        assert per.capacity == 0
        assert per.st.total() == 0.0

    def test_append_grows_capacity(self) -> None:
        per = self._make_per(size=64, n=3)
        # Append enough transitions for the pointer window to emit.
        # append_pointer emits once we have framestack + n_step = 7 entries
        # in state_buffer, so we need ~10+ appends to see capacity move.
        state = _random_stack()
        n_state = _random_stack()
        for i in range(20):
            per.append(state, action=i % 4, reward=1.0, n_state=n_state, done=False, trun=False, stream=0)
        # After warmup, some pointer entries should exist.
        assert per.capacity > 0

    def test_sample_shapes_after_fill(self) -> None:
        per = self._make_per(size=64, n=3)
        state = _random_stack()
        n_state = _random_stack()
        for i in range(30):
            per.append(state, action=i % 4, reward=0.5, n_state=n_state, done=False, trun=False, stream=0)

        batch_size = 8
        tree_idxs, states, actions, rewards, n_states, dones, weights = per.sample(batch_size)
        assert states.shape == (batch_size, 4, 20, 30)  # (B, framestack, H, W)
        assert n_states.shape == (batch_size, 4, 20, 30)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert dones.shape == (batch_size,)
        assert weights.shape == (batch_size,)
        # Tree indices are numpy ints — the Agent uses these for update_priorities.
        assert tree_idxs.dtype.kind in ("i", "u")

    def test_sampled_states_are_float32_on_device(self) -> None:
        per = self._make_per(size=64, n=3)
        state = _random_stack()
        n_state = _random_stack()
        for _ in range(20):
            per.append(state, action=0, reward=0.1, n_state=n_state, done=False, trun=False, stream=0)
        _, states, *_ = per.sample(4)
        assert states.dtype == torch.float32
        assert states.device.type == "cpu"  # we configured device="cpu" in _make_per

    def test_update_priorities_affects_subsequent_sampling(self) -> None:
        """After boosting one transition's priority so it spans multiple
        stratified-sampling segments, every batch should include it."""
        per = self._make_per(size=32, n=1)
        state = _random_stack()
        n_state = _random_stack()
        for i in range(15):
            per.append(state, action=i % 4, reward=float(i), n_state=n_state, done=False, trun=False, stream=0)

        # Grab a batch to get some tree indices, then boost the first one huge.
        # With alpha=0.2, priority^alpha: boost=1e10 → ~100, vs default ~1.
        # At batch_size=4 over ~115 total, segment length ~29 — boosted mass 100
        # spans ~3 segments, so hits 3-4 times per batch deterministically.
        tree_idxs, *_ = per.sample(2)
        boosted = tree_idxs[0]
        per.update_priorities(np.array([boosted]), np.array([1e10]))

        trials = 20
        for _trial in range(trials):
            tree_idxs, *_ = per.sample(4)
            assert boosted in tree_idxs

    def test_terminal_resets_stream_buffers(self) -> None:
        per = self._make_per(size=32, n=3)
        state = _random_stack()
        n_state = _random_stack()
        for _ in range(10):
            per.append(state, action=0, reward=1.0, n_state=n_state, done=False, trun=False, stream=0)
        # Episode ends.
        per.append(state, action=0, reward=1.0, n_state=n_state, done=True, trun=False, stream=0)
        # After terminal, the stream's buffers are reset.
        assert per.state_buffer[0] == []
        assert per.reward_buffer[0] == []
        assert per.last_terminal[0] is True

    def test_compute_discounted_rewards_n_step(self) -> None:
        per = self._make_per(size=16, n=3)
        rewards = np.array([[1.0, 1.0, 1.0]])
        dones = np.array([[False, False, False]])
        truns = np.array([[False, False, False]])
        discounted, done_out = per.compute_discounted_rewards_batch(rewards, dones, truns)
        # R = 1 + 0.99 + 0.99^2 ≈ 2.9701
        assert discounted[0] == pytest.approx(1.0 + 0.99 + 0.99**2, abs=1e-6)
        assert done_out[0] is np.False_ or not done_out[0]

    def test_sample_shapes_with_n_step_1(self) -> None:
        """Regression: VIPTankz's conditional at BTR.py:599-602 produced a
        (B, 1) shape for actions/rewards/dones when n_step=1 (should be (B,))."""
        per = self._make_per(size=32, n=1)
        state = _random_stack()
        n_state = _random_stack()
        for _ in range(20):
            per.append(state, action=0, reward=0.5, n_state=n_state, done=False, trun=False, stream=0)
        _, _, actions, rewards, _, dones, weights = per.sample(4)
        # All per-sample tensors should be 1-D of length batch_size.
        assert actions.shape == (4,), f"n=1 actions shape drift: {actions.shape}"
        assert rewards.shape == (4,), f"n=1 rewards shape drift: {rewards.shape}"
        assert dones.shape == (4,), f"n=1 dones shape drift: {dones.shape}"
        assert weights.shape == (4,), f"n=1 weights shape drift: {weights.shape}"

    def test_update_priorities_replaces_nans(self) -> None:
        """Regression: raw VIPTankz code fell through to SumTree.update with NaN
        priorities, corrupting the tree and causing subsequent sample() to
        raise OverflowError. We replace NaNs with eps instead."""
        per = self._make_per(size=16, n=1)
        state = _random_stack()
        n_state = _random_stack()
        for _ in range(10):
            per.append(state, action=0, reward=1.0, n_state=n_state, done=False, trun=False, stream=0)
        tree_idxs, *_ = per.sample(2)

        # Inject a NaN priority — should NOT poison the SumTree.
        per.update_priorities(np.array([tree_idxs[0]]), np.array([float("nan")]))

        # Tree should still be sample-able.
        per.sample(2)
        # And total priority should be finite.
        assert np.isfinite(per.st.total())

    def test_sample_on_empty_buffer_raises(self) -> None:
        """Regression: empty-buffer sample triggered NaN retry loop in raw code.
        Now raises a clear RuntimeError."""
        per = self._make_per(size=32, n=3)
        with pytest.raises(RuntimeError, match="empty buffer"):
            per.sample(4)

    def test_compute_discounted_rewards_breaks_on_done(self) -> None:
        per = self._make_per(size=16, n=3)
        rewards = np.array([[1.0, 1.0, 1.0]])
        dones = np.array([[False, True, False]])  # done fires at step 1
        truns = np.array([[False, False, False]])
        discounted, done_out = per.compute_discounted_rewards_batch(rewards, dones, truns)
        # Sum = 1 + 0.99 = 1.99 (stops after including the done step per VIPTankz's loop).
        assert discounted[0] == pytest.approx(1.0 + 0.99, abs=1e-6)
        assert done_out[0]
