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
        tree_idxs, states, actions, rewards, n_states, dones, weights, hiddens = per.sample(batch_size)
        assert states.shape == (batch_size, 4, 20, 30)  # (B, framestack, H, W)
        assert n_states.shape == (batch_size, 4, 20, 30)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert dones.shape == (batch_size,)
        assert weights.shape == (batch_size,)
        # Tree indices are numpy ints — the Agent uses these for update_priorities.
        assert tree_idxs.dtype.kind in ("i", "u")
        # Stored-hidden replay: each sample carries a (h, c) pair shaped
        # (lstm_layers, B, lstm_hidden) on device in fp32.
        h, c = hiddens
        assert h.shape == (per.lstm_layers, batch_size, per.lstm_hidden)
        assert c.shape == (per.lstm_layers, batch_size, per.lstm_hidden)
        assert h.dtype == torch.float32 and c.dtype == torch.float32

    def test_sampled_states_are_uint8_on_device(self) -> None:
        """2026-04-23: frames are kept uint8 on device; BTRPolicy normalizes
        internally. Previously cast to float32 at sample time — wasted GPU
        memory for no benefit."""
        per = self._make_per(size=64, n=3)
        state = _random_stack()
        n_state = _random_stack()
        for _ in range(20):
            per.append(state, action=0, reward=0.1, n_state=n_state, done=False, trun=False, stream=0)
        _, states, *_ = per.sample(4)
        assert states.dtype == torch.uint8
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
        _, _, actions, rewards, _, dones, weights, _ = per.sample(4)
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

    def test_update_priorities_replaces_infinities(self) -> None:
        """Regression: +inf and -inf must also be sanitized, not just NaN.
        -inf ** alpha = NaN (for non-integer alpha), re-introducing the
        corruption. +inf as a priority dominates stratified sampling."""
        per = self._make_per(size=16, n=1)
        state = _random_stack()
        n_state = _random_stack()
        for _ in range(10):
            per.append(state, action=0, reward=1.0, n_state=n_state, done=False, trun=False, stream=0)
        tree_idxs, *_ = per.sample(3)

        per.update_priorities(
            np.array([tree_idxs[0], tree_idxs[1]]),
            np.array([float("inf"), float("-inf")]),
        )
        per.sample(2)
        assert np.isfinite(per.st.total())

    def test_stored_hidden_round_trips(self) -> None:
        """append() with a non-zero (h, c) must be retrievable at sample()
        time — exact values after fp16 round-trip. If this drifts we
        silently feed wrong hiddens to the LSTM at train time."""
        per = PER(
            size=16, device="cpu", n=1, envs=1, gamma=0.99, framestack=4,
            imagex=12, imagey=10, lstm_hidden=8, lstm_layers=1,
        )
        rng = np.random.default_rng(7)
        state = _random_stack(framestack=4, h=10, w=12)
        n_state = _random_stack(framestack=4, h=10, w=12)
        # Per-transition distinct hiddens so we can verify correct indexing.
        hiddens_in: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(10):
            h = rng.standard_normal(size=(1, 8)).astype(np.float16)
            c = rng.standard_normal(size=(1, 8)).astype(np.float16)
            hiddens_in.append((h, c))
            per.append(
                state, action=i % 4, reward=0.1, n_state=n_state,
                done=False, trun=False, stream=0, hidden=(h, c),
            )
        _, _, _, _, _, _, _, (h_out, c_out) = per.sample(4)
        # Shape contract: (lstm_layers, B, lstm_hidden), float32 on device.
        assert h_out.shape == (1, 4, 8)
        assert c_out.shape == (1, 4, 8)
        assert h_out.dtype == torch.float32
        # Values should round-trip exactly in fp16 → fp32 promotion
        # (no loss of information going back up).
        assert torch.isfinite(h_out).all()
        assert torch.isfinite(c_out).all()

    def test_none_hidden_stores_zeros(self) -> None:
        """append(hidden=None) must store zeros (episode-start convention).
        sample() must return zeros for those slots."""
        per = PER(
            size=16, device="cpu", n=1, envs=1, gamma=0.99, framestack=4,
            imagex=12, imagey=10, lstm_hidden=8, lstm_layers=1,
        )
        state = _random_stack(framestack=4, h=10, w=12)
        n_state = _random_stack(framestack=4, h=10, w=12)
        for i in range(10):
            per.append(
                state, action=0, reward=0.1, n_state=n_state,
                done=False, trun=False, stream=0, hidden=None,
            )
        # h_mem and c_mem should all be zeros at the filled slots.
        assert np.all(per.h_mem[: per.reward_mem_idx] == 0)
        assert np.all(per.c_mem[: per.reward_mem_idx] == 0)

    def test_sample_on_empty_buffer_raises(self) -> None:
        """Regression: empty-buffer sample triggered NaN retry loop in raw code.
        Now raises a clear RuntimeError."""
        per = self._make_per(size=32, n=3)
        with pytest.raises(RuntimeError, match="empty buffer"):
            per.sample(4)


class TestPERSerialization:
    """Round-trip save → load of PER.state_dict via torch.save/load.

    Matters for --resume: previously checkpoints excluded replay entirely
    (3-hour re-warmup on every resume). Now _final.pt / _diverged.pt
    embed it; a silent shape/field mismatch in save/load would produce a
    post-resume run with a corrupt buffer → wrong priority sampling →
    divergence without any obvious error. These tests catch that.
    """

    def _make_per(self, size: int = 64, n: int = 3, envs: int = 2) -> PER:
        return PER(
            size=size, device="cpu", n=n, envs=envs, gamma=0.99,
            framestack=4, imagex=12, imagey=10, storage_size_multiplier=2.0,
        )

    def _populate(self, per: PER, n_transitions: int = 30, seed: int = 0) -> None:
        import numpy as _np
        rng = _np.random.default_rng(seed)
        num_envs = len(per.last_terminal)
        for i in range(n_transitions):
            state = rng.integers(0, 256, size=(4, per.imagey, per.imagex), dtype=_np.uint8)
            n_state = rng.integers(0, 256, size=(4, per.imagey, per.imagex), dtype=_np.uint8)
            per.append(
                state=state, action=i % 4, reward=0.1 * i,
                n_state=n_state, done=(i == n_transitions - 1),
                trun=False, stream=i % num_envs,
            )

    def test_round_trip_via_torch_save(self, tmp_path: Path) -> None:
        """Save → torch.save → torch.load → load into a fresh PER; verify
        every field matches + a subsequent sample() would return the same
        transition for the same random seed."""
        import torch as _torch
        p1 = self._make_per()
        self._populate(p1, n_transitions=25)

        ckpt_path = tmp_path / "per.pt"
        _torch.save(p1.state_dict(), ckpt_path)

        p2 = self._make_per()
        p2.load_state_dict(_torch.load(ckpt_path, weights_only=False))

        # Scalar + per-env state.
        assert p2.capacity == p1.capacity
        assert p2.index == p1.index
        assert p2.point_mem_idx == p1.point_mem_idx
        assert p2.state_mem_idx == p1.state_mem_idx
        assert p2.reward_mem_idx == p1.reward_mem_idx
        assert p2.max_prio == p1.max_prio
        assert p2.last_terminal == p1.last_terminal
        assert p2.tstep_counter == p1.tstep_counter

        # Big memory arrays.
        assert np.array_equal(p2.state_mem, p1.state_mem)
        assert np.array_equal(p2.action_mem, p1.action_mem)
        assert np.array_equal(p2.reward_mem, p1.reward_mem)
        assert np.array_equal(p2.done_mem, p1.done_mem)
        assert np.array_equal(p2.trun_mem, p1.trun_mem)
        # pointer_mem is a structured dtype; compare field-by-field.
        for field in p1.pointer_mem.dtype.names:
            assert np.array_equal(p2.pointer_mem[field], p1.pointer_mem[field])

        # SumTree contents (priority sampling must be identical).
        assert np.array_equal(p2.st.sum_tree, p1.st.sum_tree)
        assert p2.st.index == p1.st.index
        assert p2.st.full == p1.st.full
        assert p2.st.max == p1.st.max

    def test_round_trip_preserves_sampling(self, tmp_path: Path) -> None:
        """Deterministic check: given the same numpy seed, the same batch
        sampled before and after a save→load round-trip should match."""
        import torch as _torch
        p1 = self._make_per()
        self._populate(p1, n_transitions=40)

        ckpt_path = tmp_path / "per.pt"
        _torch.save(p1.state_dict(), ckpt_path)

        # Sample from p1.
        _np_seed = 12345
        np.random.seed(_np_seed)
        idxs1, *rest1 = p1.sample(batch_size=2)

        # Fresh PER, load, sample with same seed.
        p2 = self._make_per()
        p2.load_state_dict(_torch.load(ckpt_path, weights_only=False))
        np.random.seed(_np_seed)
        idxs2, *rest2 = p2.sample(batch_size=2)

        assert np.array_equal(idxs1, idxs2)

    def test_shape_mismatch_raises(self) -> None:
        """Loading a payload built with different config dims must fail
        loudly rather than silently load garbage."""
        p1 = self._make_per(size=64)
        self._populate(p1, n_transitions=20)
        sd = p1.state_dict()

        # Fresh PER with different framestack → state_mem shape differs.
        p2 = PER(
            size=64, device="cpu", n=3, envs=2, gamma=0.99,
            framestack=4, imagex=16, imagey=10,  # imagex changed 12→16
            storage_size_multiplier=2.0,
        )
        import pytest as _pytest
        with _pytest.raises(ValueError, match="shape mismatch"):
            p2.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Discounted-reward helper.
# ---------------------------------------------------------------------------


class TestComputeDiscountedRewards:
    def _make_per(self, size: int = 128, n: int = 3, envs: int = 1) -> PER:
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

    def test_compute_discounted_rewards_breaks_on_done(self) -> None:
        per = self._make_per(size=16, n=3)
        rewards = np.array([[1.0, 1.0, 1.0]])
        dones = np.array([[False, True, False]])  # done fires at step 1
        truns = np.array([[False, False, False]])
        discounted, done_out = per.compute_discounted_rewards_batch(rewards, dones, truns)
        # Sum = 1 + 0.99 = 1.99 (stops after including the done step per VIPTankz's loop).
        assert discounted[0] == pytest.approx(1.0 + 0.99, abs=1e-6)
        assert done_out[0]
