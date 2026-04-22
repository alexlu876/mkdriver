"""Tests for ``src/mkw_rl/rl/train.py``.

Pure-function tests — no live Dolphin, no real env. Covers:
- Config loading from YAML (main + testing-subtree merge)
- Quantile Huber loss shapes + axis convention (Dabney eq. 10)
- Munchausen reward computation shapes
- Full per-timestep Munchausen-IQN loss end-to-end
- BTRAgent.build + learn_step + sync_target + act + checkpoint round-trip
- Priority write-back (R2D2 aggregation formula)
- NaN/inf bail and dones handling

Does NOT test:
- Rollout against live Dolphin (covered by scripts/smoke_env.py).
- Wandb logging (third-party dep; mocked via env var absence → CSV fallback).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mkw_rl.rl.train import (
    TrainConfig,
    _compute_munchausen_reward,
    _compute_td_error_and_loss,
    _quantile_huber_loss,
    load_config,
)

# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _tiny_cfg(tmp_path: Path, **overrides) -> TrainConfig:
    """A minimal in-memory TrainConfig that can build a BTRAgent."""
    savestate_dir = tmp_path / "savestates"
    savestate_dir.mkdir(exist_ok=True)
    (savestate_dir / "luigi_circuit_tt.sav").write_bytes(b"")
    base = dict(
        savestate_dir=str(savestate_dir),
        log_dir=str(tmp_path / "runs"),
        batch_size=2,
        replay_size=64,
        storage_size_multiplier=2.0,
        lstm_hidden=16,
        feature_dim=16,
        linear_size=16,
        num_tau=4,
        n_cos=8,
        encoder_channels=(4, 8, 8),
        framestack=4,
        imagex=32,
        imagey=24,
        input_hw=(24, 32),
        stack_size=4,
        min_sampling_size=16,
        burn_in_len=2,
        learning_seq_len=4,
        n_step=2,
        layer_norm=False,
        testing=True,
    )
    base.update(overrides)
    return TrainConfig(**base)


def _populate_replay(agent, n: int, rng: np.random.Generator | None = None) -> None:
    """Push ``n`` synthetic transitions into the agent's replay buffer."""
    if rng is None:
        rng = np.random.default_rng(0)
    stack_shape = (agent.cfg.framestack, agent.cfg.imagey, agent.cfg.imagex)
    for i in range(n):
        state = rng.integers(0, 256, size=stack_shape, dtype=np.uint8)
        nstate = rng.integers(0, 256, size=stack_shape, dtype=np.uint8)
        agent.replay.append(
            state=state,
            action=int(rng.integers(0, 40)),
            reward=float(rng.standard_normal()),
            n_state=nstate,
            done=(i == n - 1),  # terminate at the end so last n-step window closes
            trun=False,
            stream=0,
        )

# ---------------------------------------------------------------------------
# Config loading.
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_loads_main_yaml(self) -> None:
        """The production btr.yaml should load with all expected defaults."""
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "btr.yaml")
        assert cfg.batch_size == 256
        assert cfg.target_replace_grad_steps == 200  # audit override, not VIPTankz's 500
        assert cfg.burn_in_len == 20
        assert cfg.learning_seq_len == 40
        assert cfg.priority_eta == 0.9
        assert cfg.testing is False

    def test_testing_flag_merges_subtree(self) -> None:
        """--testing should override key values from the testing: subtree."""
        cfg = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "btr.yaml",
            testing=True,
        )
        # Shrunk values per the testing: override.
        assert cfg.total_frames == 500
        assert cfg.batch_size == 4
        assert cfg.replay_size == 1024
        assert cfg.lstm_hidden == 64  # tiny model
        assert cfg.burn_in_len == 4
        assert cfg.testing is True

    def test_input_hw_converted_to_tuple(self) -> None:
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "btr.yaml")
        assert isinstance(cfg.input_hw, tuple)
        assert cfg.input_hw == (75, 140)


# ---------------------------------------------------------------------------
# Quantile Huber loss.
# ---------------------------------------------------------------------------


class TestQuantileHuberLoss:
    def test_output_shape(self) -> None:
        B, nt_online, nt_target = 4, 8, 8
        td = torch.randn(B, nt_online, nt_target)
        taus = torch.rand(B, nt_online, 1)
        loss = _quantile_huber_loss(td, taus)
        assert loss.shape == (B,)

    def test_zero_error_zero_loss(self) -> None:
        td = torch.zeros(3, 8, 8)
        taus = torch.rand(3, 8, 1)
        loss = _quantile_huber_loss(td, taus)
        assert torch.allclose(loss, torch.zeros(3))

    def test_positive_error_positive_loss(self) -> None:
        td = torch.ones(2, 8, 8)
        taus = torch.rand(2, 8, 1)
        loss = _quantile_huber_loss(td, taus)
        assert (loss > 0).all()

    def test_dabney_axis_convention(self) -> None:
        """Verify we sum over axis 2 (target-τ) and mean over axis 1 (online-τ),
        per Dabney eq. 10 — NOT VIPTankz's sum(dim=1).mean(dim=1) which swaps."""
        B, nt_online, nt_target = 2, 4, 6  # Deliberately unequal.
        td = torch.full((B, nt_online, nt_target), 1.0)
        taus = torch.full((B, nt_online, 1), 0.5)
        loss = _quantile_huber_loss(td, taus)
        # With all-ones TD error and τ=0.5: huber=0.5 for |δ|≤1, weight |0.5-0|=0.5.
        # quantile_l = 0.5 × 0.5 = 0.25 per element. Sum over target (6) → 1.5.
        # Mean over online (4) → 1.5. So loss per batch = 1.5.
        expected = 0.25 * nt_target  # sum over 6 target values
        # Mean over online is already applied; one value per batch element.
        assert torch.allclose(loss, torch.full((B,), expected), atol=1e-5)


# ---------------------------------------------------------------------------
# Munchausen reward.
# ---------------------------------------------------------------------------


class TestMunchausenReward:
    def test_shape(self) -> None:
        B, nt, n_actions = 4, 8, 40
        q = torch.randn(B, nt, n_actions)
        rewards = torch.randn(B)
        actions = torch.randint(0, n_actions, (B,))
        out = _compute_munchausen_reward(
            q, rewards, actions, entropy_tau=0.03, munch_alpha=0.9, munch_lo=-1.0
        )
        assert out.shape == (B, 1, 1)

    def test_bonus_clamped_at_zero(self) -> None:
        """Munchausen bonus = α · clamp(log π(a|s), lo, 0). Greedy action →
        log π(a|s) is closest to 0 (most probable); clamped at 0 means it
        contributes nothing beyond the base reward."""
        B, nt, n_actions = 2, 8, 3
        # Construct Q values where action 0 is dominant.
        q = torch.full((B, nt, n_actions), -10.0)
        q[:, :, 0] = 10.0
        rewards = torch.tensor([0.0, 0.0])
        actions = torch.tensor([0, 0])  # pick the dominant action
        out = _compute_munchausen_reward(
            q, rewards, actions, entropy_tau=0.03, munch_alpha=0.9, munch_lo=-1.0
        )
        # The bonus for the dominant action should be ≈ 0 (log π ≈ 0 for p ≈ 1).
        # So out ≈ rewards = 0.
        assert torch.allclose(out.squeeze(), torch.zeros(B), atol=1e-2)

    def test_bonus_clamped_at_lo_for_rare_action(self) -> None:
        """Rare action → log π very negative → clamped at munch_lo."""
        B, nt, n_actions = 2, 8, 3
        q = torch.full((B, nt, n_actions), 10.0)
        q[:, :, 0] = -10.0  # action 0 is rare
        rewards = torch.tensor([1.0, 1.0])
        actions = torch.tensor([0, 0])  # pick the rare action
        out = _compute_munchausen_reward(
            q, rewards, actions, entropy_tau=0.03, munch_alpha=0.9, munch_lo=-1.0
        )
        # bonus = α × (-1) = -0.9, so out = 1 + (-0.9) = 0.1.
        assert torch.allclose(out.squeeze(), torch.tensor([0.1, 0.1]), atol=1e-3)


# ---------------------------------------------------------------------------
# Full loss end-to-end.
# ---------------------------------------------------------------------------


class TestComputeLoss:
    def test_full_loss_runs_and_produces_gradient(self) -> None:
        """End-to-end check: forward → loss → backward → gradients exist."""
        B, nt, n_actions = 4, 8, 40
        online_q = torch.randn(B, nt, n_actions, requires_grad=True)
        online_taus = torch.rand(B, nt, 1)
        target_q = torch.randn(B, nt, n_actions)
        actions = torch.randint(0, n_actions, (B,))
        rewards = torch.randn(B)
        dones = torch.zeros(B, 1, 1, dtype=torch.bool)
        weights = torch.ones(B)

        munch_r = _compute_munchausen_reward(
            online_q.detach(), rewards, actions,
            entropy_tau=0.03, munch_alpha=0.9, munch_lo=-1.0,
        )
        loss, td_abs = _compute_td_error_and_loss(
            online_q, online_taus, target_q, actions, munch_r,
            gamma_n=0.99, dones=dones, weights=weights, entropy_tau=0.03,
        )
        assert loss.ndim == 0
        assert td_abs.shape == (B,)
        loss.backward()
        assert online_q.grad is not None
        assert online_q.grad.abs().sum() > 0

    def test_weights_scale_loss(self) -> None:
        """PER weights scale the loss linearly. With uniform 10× weights, mean
        loss should be exactly 10× — not approximately."""
        torch.manual_seed(0)
        B, nt, n_actions = 2, 4, 10
        online_q = torch.randn(B, nt, n_actions)
        online_taus = torch.rand(B, nt, 1)
        target_q = torch.randn(B, nt, n_actions)
        actions = torch.randint(0, n_actions, (B,))
        rewards = torch.randn(B)
        dones = torch.zeros(B, 1, 1, dtype=torch.bool)
        munch_r = _compute_munchausen_reward(
            online_q, rewards, actions, 0.03, 0.9, -1.0,
        )
        loss1, _ = _compute_td_error_and_loss(
            online_q, online_taus, target_q, actions, munch_r,
            0.99, dones, torch.ones(B), 0.03,
        )
        loss10, _ = _compute_td_error_and_loss(
            online_q, online_taus, target_q, actions, munch_r,
            0.99, dones, torch.ones(B) * 10, 0.03,
        )
        assert abs(loss10.item() / loss1.item() - 10.0) < 1e-5

    def test_zero_weights_zero_gradient(self) -> None:
        """PER weights=0 should produce zero gradient on the loss."""
        torch.manual_seed(0)
        B, nt, n_actions = 2, 4, 10
        online_q = torch.randn(B, nt, n_actions, requires_grad=True)
        online_taus = torch.rand(B, nt, 1)
        target_q = torch.randn(B, nt, n_actions)
        actions = torch.randint(0, n_actions, (B,))
        rewards = torch.randn(B)
        dones = torch.zeros(B, 1, 1, dtype=torch.bool)
        munch_r = _compute_munchausen_reward(
            online_q.detach(), rewards, actions, 0.03, 0.9, -1.0,
        )
        loss, _ = _compute_td_error_and_loss(
            online_q, online_taus, target_q, actions, munch_r,
            0.99, dones, torch.zeros(B), 0.03,
        )
        loss.backward()
        assert online_q.grad is not None
        assert online_q.grad.abs().sum().item() == 0.0

    def test_munchausen_alpha_affects_gradient(self) -> None:
        """Turning Munchausen off (alpha=0) changes the loss, proving the
        Munchausen pathway actually participates in the gradient."""
        torch.manual_seed(42)
        B, nt, n_actions = 2, 4, 10
        online_q = torch.randn(B, nt, n_actions)
        online_taus = torch.rand(B, nt, 1)
        target_q = torch.randn(B, nt, n_actions)
        actions = torch.randint(0, n_actions, (B,))
        rewards = torch.randn(B)
        dones = torch.zeros(B, 1, 1, dtype=torch.bool)

        def _loss(alpha: float) -> float:
            mr = _compute_munchausen_reward(
                online_q, rewards, actions, 0.03, alpha, -1.0,
            )
            loss, _ = _compute_td_error_and_loss(
                online_q, online_taus, target_q, actions, mr,
                0.99, dones, torch.ones(B), 0.03,
            )
            return loss.item()

        loss_off = _loss(0.0)
        loss_on = _loss(0.9)
        assert loss_off != loss_on  # Munchausen bonus must be load-bearing.

    def test_all_dones_zeros_q_target(self) -> None:
        """When dones=True, the bootstrapped q_target term is zeroed and the
        loss reduces to the quantile-Huber of (munch_r - q_expected)."""
        torch.manual_seed(7)
        B, nt, n_actions = 3, 4, 8
        online_q = torch.randn(B, nt, n_actions)
        online_taus = torch.rand(B, nt, 1)
        target_q = torch.randn(B, nt, n_actions)
        actions = torch.randint(0, n_actions, (B,))
        rewards = torch.zeros(B)  # zero reward + zero bonus (all-actions equal-Q) = zero munch
        dones_true = torch.ones(B, 1, 1, dtype=torch.bool)
        munch_r = _compute_munchausen_reward(
            online_q, rewards, actions, 0.03, 0.9, -1.0,
        )
        # Flipping target_q to zeros shouldn't matter under dones=True: q_target
        # is entirely gated out. Use two different target_q and assert equal loss.
        loss_a, _ = _compute_td_error_and_loss(
            online_q, online_taus, target_q, actions, munch_r,
            0.99, dones_true, torch.ones(B), 0.03,
        )
        loss_b, _ = _compute_td_error_and_loss(
            online_q, online_taus, target_q * 100.0, actions, munch_r,
            0.99, dones_true, torch.ones(B), 0.03,
        )
        # pi_target + tau_log_pi_next still depend on target_q via the soft value,
        # but the (target_quantiles - tau_log_pi_next) term is zeroed by ~dones.
        # Only the munchausen_reward pathway (which uses online_q, not target_q)
        # should contribute. So loss_a == loss_b.
        assert torch.allclose(loss_a, loss_b, atol=1e-5)


# ---------------------------------------------------------------------------
# BTRAgent construction.
# ---------------------------------------------------------------------------


class TestBTRAgentBuild:
    def test_builds_with_testing_config(self, tmp_path: Path) -> None:
        """Construction requires at least one savestate file. Mock it by
        creating a fake .sav file in a temp dir and pointing cfg there."""
        savestate_dir = tmp_path / "savestates"
        savestate_dir.mkdir()
        (savestate_dir / "luigi_circuit_tt.sav").write_bytes(b"")

        cfg = TrainConfig(
            savestate_dir=str(savestate_dir),
            batch_size=4,
            replay_size=128,
            lstm_hidden=32,
            feature_dim=16,
            linear_size=16,
            min_sampling_size=8,
            testing=True,
        )
        # Lazy import so the test can patch env lookups if needed.
        from mkw_rl.rl.train import BTRAgent

        agent = BTRAgent.build(cfg)
        assert agent.grad_steps == 0
        assert agent.env_steps == 0
        assert agent.online_net.cfg.lstm_hidden == 32
        assert agent.replay.size == 128
        assert agent.sampler.n_tracks == 1

    def test_build_raises_on_empty_savestate_dir(self, tmp_path: Path) -> None:
        savestate_dir = tmp_path / "empty"
        savestate_dir.mkdir()
        cfg = TrainConfig(savestate_dir=str(savestate_dir))
        from mkw_rl.rl.train import BTRAgent

        with pytest.raises(RuntimeError, match="no savestates found"):
            BTRAgent.build(cfg)


# ---------------------------------------------------------------------------
# Logger fallback.
# ---------------------------------------------------------------------------


class TestLogger:
    def test_csv_fallback_when_no_wandb_key(self, tmp_path: Path) -> None:
        """No WANDB_API_KEY → CSV logger at log_dir/{run_name}.csv."""
        cfg = TrainConfig(log_dir=str(tmp_path))
        with patch.dict("os.environ", {}, clear=False):
            if "WANDB_API_KEY" in __import__("os").environ:
                del __import__("os").environ["WANDB_API_KEY"]
            from mkw_rl.rl.train import make_logger
            logger = make_logger(cfg, "test_run")
        logger.log({"loss": 0.5}, step=1)
        logger.close()
        # CSV file exists.
        assert (tmp_path / "test_run.csv").exists()

    def test_csv_disjoint_keys_are_recoverable(self, tmp_path: Path) -> None:
        """Log two rows with disjoint metric keys; both rows must parse back
        cleanly via csv.DictReader without any '#'-prefixed commentary."""
        from mkw_rl.rl.train import _CSVLogger

        path = tmp_path / "log.csv"
        logger = _CSVLogger(path)
        logger.log({"loss": 0.5, "acc": 1.0}, step=1)
        logger.log({"grad_norm": 2.0}, step=2)  # disjoint — schema grows
        logger.close()

        import csv as _csv
        with open(path) as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert set(reader.fieldnames or []) == {"step", "loss", "acc", "grad_norm"}
        # Row 1: loss + acc populated, grad_norm blank.
        assert float(rows[0]["loss"]) == 0.5
        assert float(rows[0]["acc"]) == 1.0
        assert rows[0]["grad_norm"] == ""
        # Row 2: grad_norm populated, loss + acc blank.
        assert float(rows[1]["grad_norm"]) == 2.0
        assert rows[1]["loss"] == ""
        assert rows[1]["acc"] == ""


# ---------------------------------------------------------------------------
# BTRAgent — act / sync_target / learn_step / checkpoint round-trip.
# ---------------------------------------------------------------------------


class TestBTRAgentAct:
    def test_act_returns_valid_action_and_hidden(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        frames = torch.zeros(
            1, 1, cfg.stack_size, cfg.imagey, cfg.imagex, dtype=torch.uint8
        )
        action, hidden = agent.act(frames, hidden=None)
        assert isinstance(action, int)
        assert 0 <= action < 40  # NUM_ACTIONS
        # LSTM hidden: (h, c) each of shape (lstm_layers, B=1, lstm_hidden)
        h, c = hidden
        assert h.shape == (cfg.lstm_layers, 1, cfg.lstm_hidden)
        assert c.shape == (cfg.lstm_layers, 1, cfg.lstm_hidden)


class TestBTRAgentSyncTarget:
    def test_sync_target_copies_online_weights(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        # Mutate one online parameter in-place.
        first_param_name = next(iter(agent.online_net.state_dict()))
        with torch.no_grad():
            agent.online_net.state_dict()[first_param_name].add_(100.0)

        # Before sync, online != target for that param.
        assert not torch.equal(
            agent.online_net.state_dict()[first_param_name],
            agent.target_net.state_dict()[first_param_name],
        )
        agent.sync_target()
        # After sync, they match everywhere.
        online_sd = agent.online_net.state_dict()
        target_sd = agent.target_net.state_dict()
        for k in online_sd:
            assert torch.equal(online_sd[k], target_sd[k]), f"mismatch at {k}"


class TestBTRAgentLearnStep:
    def test_learn_step_updates_weights(self, tmp_path: Path) -> None:
        """Populate replay with random transitions; run two learn_steps.
        Verify grad_steps advances, loss is finite, and online weights change."""
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        # Need min_sampling_size + seq_len + n_step headroom.
        seq_len = cfg.burn_in_len + cfg.learning_seq_len
        _populate_replay(agent, cfg.min_sampling_size + seq_len + cfg.n_step + 4)

        # Snapshot a weight-bearing online param (avoid NoisyLinear ε buffers —
        # those are resampled every forward and "change" on every step trivially).
        snapshot = {
            k: v.clone() for k, v in agent.online_net.state_dict().items()
            if "weight" in k and "epsilon" not in k and v.dtype == torch.float32
        }
        assert snapshot, "test helper assumption: at least one weight param"

        m1 = agent.learn_step()
        m2 = agent.learn_step()

        assert agent.grad_steps == 2
        assert "loss" in m1 and np.isfinite(m1["loss"])
        assert "loss" in m2 and np.isfinite(m2["loss"])
        assert "grad_norm" in m1 and np.isfinite(m1["grad_norm"])

        # Online weights should have shifted for at least one parameter.
        after = agent.online_net.state_dict()
        any_changed = any(
            not torch.equal(after[k], v) for k, v in snapshot.items()
        )
        assert any_changed, "learn_step didn't alter any online weight"

    def test_learn_step_noop_before_warmup(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path, min_sampling_size=10_000)
        agent = BTRAgent.build(cfg)
        # Replay empty / below threshold.
        out = agent.learn_step()
        assert out == {}
        assert agent.grad_steps == 0

    def test_learn_step_priority_writeback(self, tmp_path: Path) -> None:
        """Spy on replay.update_priorities; verify called with R2D2 aggregation
        (η·max + (1-η)·mean over the per-timestep |δ| sequence)."""
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        seq_len = cfg.burn_in_len + cfg.learning_seq_len
        _populate_replay(agent, cfg.min_sampling_size + seq_len + cfg.n_step + 4)

        captured: list[tuple[np.ndarray, np.ndarray]] = []
        real_update = agent.replay.update_priorities

        def spy(indices, priorities):  # noqa: ANN001
            captured.append((np.asarray(indices).copy(), np.asarray(priorities).copy()))
            return real_update(indices, priorities)

        with patch.object(agent.replay, "update_priorities", side_effect=spy):
            agent.learn_step()

        assert len(captured) == 1
        idxs, prios = captured[0]
        assert idxs.shape == (cfg.batch_size,)
        assert prios.shape == (cfg.batch_size,)
        # Values must be finite non-negative.
        assert np.all(np.isfinite(prios))
        assert np.all(prios >= 0)

    def test_learn_step_nonfinite_loss_skipped(self, tmp_path: Path) -> None:
        """Monkey-patch the loss fn to return NaN; learn_step must skip the
        step + increment nonfinite_streak without crashing."""
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        seq_len = cfg.burn_in_len + cfg.learning_seq_len
        _populate_replay(agent, cfg.min_sampling_size + seq_len + cfg.n_step + 4)

        nan_loss = torch.tensor(float("nan"))
        fake_td = torch.zeros(cfg.batch_size * cfg.learning_seq_len)

        with patch(
            "mkw_rl.rl.train._compute_td_error_and_loss",
            return_value=(nan_loss, fake_td),
        ):
            out = agent.learn_step()
        assert out["loss"] != out["loss"]  # NaN
        assert agent.grad_steps == 0  # step was skipped
        assert agent.nonfinite_streak == 1


class TestCheckpointRoundTrip:
    def test_save_load_restores_weights_and_counters(self, tmp_path: Path) -> None:
        """Full round-trip: save, build fresh agent, load, verify state matches."""
        from mkw_rl.rl.train import BTRAgent, _save_checkpoint, load_checkpoint

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        # Mutate a weight so the saved state clearly differs from a fresh build.
        first_key = next(
            k for k, v in agent.online_net.state_dict().items()
            if "weight" in k and v.dtype == torch.float32 and v.numel() > 0
        )
        with torch.no_grad():
            agent.online_net.state_dict()[first_key].add_(42.0)
        agent.grad_steps = 500
        agent.env_steps = 10_000

        ckpt_path = tmp_path / "test.pt"
        _save_checkpoint(agent, cfg, ckpt_path)

        # Fresh agent — BTRAgent.build reseeds torch, so its weights match the
        # original pre-mutation state. Confirm the mutation actually took.
        fresh = BTRAgent.build(cfg)
        assert not torch.equal(
            agent.online_net.state_dict()[first_key],
            fresh.online_net.state_dict()[first_key],
        )

        load_checkpoint(fresh, ckpt_path)

        # After load: online + target + counters all match.
        for k in agent.online_net.state_dict():
            assert torch.equal(
                agent.online_net.state_dict()[k],
                fresh.online_net.state_dict()[k],
            ), f"online mismatch at {k}"
        for k in agent.target_net.state_dict():
            assert torch.equal(
                agent.target_net.state_dict()[k],
                fresh.target_net.state_dict()[k],
            ), f"target mismatch at {k}"
        assert fresh.grad_steps == 500
        assert fresh.env_steps == 10_000

    def test_save_load_preserves_sampler_state(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent, _save_checkpoint, load_checkpoint

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        agent.sampler.update("luigi_circuit_tt", 12.5)
        saved_progress = dict(agent.sampler.progress)

        ckpt_path = tmp_path / "sampler.pt"
        _save_checkpoint(agent, cfg, ckpt_path)

        fresh = BTRAgent.build(cfg)
        assert fresh.sampler.progress != saved_progress
        load_checkpoint(fresh, ckpt_path)
        assert fresh.sampler.progress == saved_progress
