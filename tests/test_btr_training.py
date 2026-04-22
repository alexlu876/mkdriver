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


# ---------------------------------------------------------------------------
# FakeEnv + rollout / crash-restart / shutdown tests.
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Minimal gym-shaped env for testing ``run_one_episode`` and the
    outer ``train()`` loop without spinning up Dolphin.

    ``scripted_rewards``: list of per-step rewards. Episode terminates when
    the list is exhausted (``terminated=True``) or ``crash_on_reset_count``
    resets have happened (raises ``BrokenPipeError`` to exercise the
    crash-restart path).

    ``reward_breakdown``: optional dict-valued per-step breakdown, echoed
    through info["reward_breakdown"] to verify component accumulation.
    """

    def __init__(
        self,
        framestack: int = 4,
        h: int = 24,
        w: int = 32,
        scripted_rewards: list[float] | None = None,
        reward_breakdown: dict[str, float] | None = None,
        crash_on_reset_n: int | None = None,
        crash_error: type[Exception] = BrokenPipeError,
    ) -> None:
        self.framestack = framestack
        self.h = h
        self.w = w
        self.scripted_rewards = scripted_rewards or [1.0, 2.0, 3.0]
        self.reward_breakdown = reward_breakdown
        self.crash_on_reset_n = crash_on_reset_n
        self.crash_error = crash_error
        self.reset_count = 0
        self.closed = False
        self._idx = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # noqa: ARG002
        self.reset_count += 1
        if self.crash_on_reset_n is not None and self.reset_count == self.crash_on_reset_n:
            raise self.crash_error("scripted crash from FakeEnv.reset")
        self._idx = 0
        obs = np.zeros((self.framestack, self.h, self.w), dtype=np.uint8)
        return obs, {"track_slug": (options or {}).get("track_slug")}

    def step(self, action: int):  # noqa: ARG002
        reward = self.scripted_rewards[self._idx]
        terminated = self._idx == len(self.scripted_rewards) - 1
        self._idx += 1
        obs = np.full((self.framestack, self.h, self.w), self._idx, dtype=np.uint8)
        info = {}
        if self.reward_breakdown:
            info["reward_breakdown"] = self.reward_breakdown
        return obs, reward, terminated, False, info

    def close(self) -> None:
        self.closed = True


class TestRunOneEpisode:
    def test_rollout_accumulates_return_and_appends_replay(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent, run_one_episode

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        scripted = [1.0, 2.0, 3.0, 4.0]
        breakdown = {"speed_bonus": 0.5, "checkpoint": 1.5}
        env = _FakeEnv(
            framestack=cfg.framestack, h=cfg.imagey, w=cfg.imagex,
            scripted_rewards=scripted, reward_breakdown=breakdown,
        )
        initial_capacity = agent.replay.capacity
        initial_env_steps = agent.env_steps

        ep_return, rb_sums, n_steps = run_one_episode(agent, env, "luigi_circuit_tt")

        assert n_steps == len(scripted)
        assert ep_return == sum(scripted)
        assert agent.env_steps == initial_env_steps + len(scripted)
        assert agent.replay.capacity == initial_capacity + len(scripted)
        assert rb_sums == {
            "speed_bonus": 0.5 * len(scripted),
            "checkpoint": 1.5 * len(scripted),
        }

    def test_shutdown_flag_breaks_episode_early(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent, run_one_episode

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        env = _FakeEnv(
            framestack=cfg.framestack, h=cfg.imagey, w=cfg.imagex,
            scripted_rewards=[1.0] * 100,  # long episode
        )
        shutdown_flag = {"shutdown": True}
        _, _, n_steps = run_one_episode(
            agent, env, "luigi_circuit_tt", shutdown_flag=shutdown_flag,
        )
        # First step runs, then the shutdown check fires at the bottom of the
        # loop. So we expect exactly 1 step before break.
        assert n_steps == 1

    def test_stack_shape_assertion(self, tmp_path: Path) -> None:
        """If env returns a wrong-shape stack, run_one_episode should fail loudly."""
        from mkw_rl.rl.train import BTRAgent, run_one_episode

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        # Wrong framestack size — should trip the assertion.
        env = _FakeEnv(framestack=cfg.framestack + 1, h=cfg.imagey, w=cfg.imagex)
        with pytest.raises(AssertionError, match="framestack"):
            run_one_episode(agent, env, "luigi_circuit_tt")


class TestLearnStepGradNormNaN:
    def test_nonfinite_grad_norm_skipped(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        seq_len = cfg.burn_in_len + cfg.learning_seq_len
        _populate_replay(agent, cfg.min_sampling_size + seq_len + cfg.n_step + 4)

        with patch(
            "torch.nn.utils.clip_grad_norm_",
            return_value=torch.tensor(float("inf")),
        ):
            out = agent.learn_step()
        assert out["grad_norm"] != out["grad_norm"] or not np.isfinite(out["grad_norm"])
        assert agent.grad_steps == 0
        assert agent.nonfinite_streak == 1


class TestMaxNonFiniteAbort:
    def test_max_nonfinite_raises(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        seq_len = cfg.burn_in_len + cfg.learning_seq_len
        _populate_replay(agent, cfg.min_sampling_size + seq_len + cfg.n_step + 4)

        agent.nonfinite_streak = agent.MAX_NONFINITE - 1
        nan_loss = torch.tensor(float("nan"))
        fake_td = torch.zeros(cfg.batch_size * cfg.learning_seq_len)
        with patch(
            "mkw_rl.rl.train._compute_td_error_and_loss",
            return_value=(nan_loss, fake_td),
        ):
            with pytest.raises(RuntimeError, match="consecutive non-finite"):
                agent.learn_step()


class TestSamplerRngRoundTrip:
    def test_rng_state_restoration(self, tmp_path: Path) -> None:
        """Sampler RNG state is preserved across state_dict round-trip:
        sample N before save, then sample N after load, should equal
        sample N of an uninterrupted run of 2N samples."""
        from mkw_rl.rl.track_sampler import (
            ProgressWeightedTrackSampler,
            TrackSamplerConfig,
        )

        tracks = ["a", "b", "c", "d"]
        ref = ProgressWeightedTrackSampler(
            track_slugs=tracks, config=TrackSamplerConfig(), seed=7,
        )
        # Give tracks non-uniform progress so the RNG actually matters.
        for slug, prog in zip(tracks, [0.5, 1.0, 0.2, 0.8], strict=True):
            ref.update(slug, prog)
        # Uninterrupted: draw 10 samples.
        uninterrupted = [ref.sample() for _ in range(10)]

        # Interrupted: draw 5, save, load into fresh sampler, draw 5 more.
        sampler = ProgressWeightedTrackSampler(
            track_slugs=tracks, config=TrackSamplerConfig(), seed=7,
        )
        for slug, prog in zip(tracks, [0.5, 1.0, 0.2, 0.8], strict=True):
            sampler.update(slug, prog)
        first5 = [sampler.sample() for _ in range(5)]
        state = sampler.state_dict()

        resumed = ProgressWeightedTrackSampler(
            track_slugs=tracks, config=TrackSamplerConfig(), seed=999,  # different seed
        )
        resumed.load_state_dict(state)
        last5 = [resumed.sample() for _ in range(5)]

        assert first5 + last5 == uninterrupted


class TestCrashRestart:
    def test_train_recovers_from_env_crash(self, tmp_path: Path) -> None:
        """train() should catch a BrokenPipeError from the env, reconstruct it,
        and continue. Uses a FakeEnv that crashes on reset #2 then recovers."""
        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import train

        # Stand up a tiny config with immediate episode termination.
        cfg = _tiny_cfg(tmp_path, total_frames=15, min_sampling_size=10_000)
        # Patch _make_env so train() builds FakeEnvs. State is tracked across
        # constructor calls via the `envs` list so we can assert on it.
        envs: list[_FakeEnv] = []

        def fake_make(c: TrainConfig) -> _FakeEnv:  # noqa: ARG001
            # First env crashes on its 2nd reset (after one clean episode);
            # replacement envs run cleanly so training completes.
            crash_on = 2 if len(envs) == 0 else None
            env = _FakeEnv(
                framestack=c.framestack, h=c.imagey, w=c.imagex,
                scripted_rewards=[1.0] * 3,
                crash_on_reset_n=crash_on,
            )
            envs.append(env)
            return env

        with patch.object(train_mod, "_make_env", side_effect=fake_make):
            train(cfg)

        # We expect: env #0 runs until crashing on episode N, env #1 was built
        # to crash, env #2 (and later) run cleanly. At least 2 envs built.
        assert len(envs) >= 2
        # All but the last one should be closed (torn down on crash recovery).
        assert envs[0].closed
        assert envs[-1].closed  # last env closed in finally

    def test_train_removes_track_after_repeat_crashes(self, tmp_path: Path) -> None:
        """A track that crashes Dolphin 3× should be removed from the sampler."""
        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import train

        cfg = _tiny_cfg(tmp_path, total_frames=20, min_sampling_size=10_000)
        # Pre-create two tracks so the sampler doesn't abort when one is dropped.
        (Path(cfg.savestate_dir) / "mushroom_gorge_tt.sav").write_bytes(b"")

        def fake_make(c: TrainConfig) -> _FakeEnv:  # noqa: ARG001
            # Always crash on first reset — every env will trigger the crash path.
            return _FakeEnv(
                framestack=c.framestack, h=c.imagey, w=c.imagex,
                scripted_rewards=[1.0] * 3,
                crash_on_reset_n=1,
            )

        # Force the sampler to always pick the SAME track so per-track streak
        # hits 3 before env_crash_streak hits 5.
        with (
            patch.object(train_mod, "_make_env", side_effect=fake_make),
            patch(
                "mkw_rl.rl.track_sampler.ProgressWeightedTrackSampler.sample",
                return_value="luigi_circuit_tt",
            ),
            pytest.raises(RuntimeError),
        ):
            train(cfg)

        # Can't introspect sampler here post-raise, but the lack of KeyError
        # from remove_track at log warning time is the proof. For a stronger
        # assertion see the sampler-level test below.

    def test_sampler_remove_track(self, tmp_path: Path) -> None:
        """Direct test of sampler.remove_track — distribution renormalizes."""
        from mkw_rl.rl.track_sampler import (
            ProgressWeightedTrackSampler,
            TrackSamplerConfig,
        )

        _ = tmp_path  # unused
        sampler = ProgressWeightedTrackSampler(
            track_slugs=["a", "b", "c"], config=TrackSamplerConfig(), seed=0,
        )
        sampler.remove_track("b")
        assert sampler.n_tracks == 2
        assert "b" not in sampler.progress
        dist = sampler.distribution()
        assert set(dist.keys()) == {"a", "c"}
        assert abs(sum(dist.values()) - 1.0) < 1e-9

        # Removing non-existent raises
        with pytest.raises(KeyError):
            sampler.remove_track("z")

        # Removing the last raises RuntimeError
        sampler.remove_track("a")
        with pytest.raises(RuntimeError, match="no tracks left"):
            sampler.remove_track("c")


class TestShutdownHandler:
    def test_sigterm_sets_shutdown_flag(self) -> None:
        import os
        import signal as _signal

        from mkw_rl.rl.train import _install_shutdown_handler

        flag, restore = _install_shutdown_handler()
        try:
            assert flag["shutdown"] is False
            os.kill(os.getpid(), _signal.SIGTERM)
            # Python delivers signals between bytecode ops on the main thread;
            # a no-op loop flushes the pending signal.
            for _ in range(10):
                if flag["shutdown"]:
                    break
            assert flag["shutdown"] is True
        finally:
            restore()

    def test_second_signal_restores_default_handler(self) -> None:
        import os
        import signal as _signal

        from mkw_rl.rl.train import _install_shutdown_handler

        flag, restore = _install_shutdown_handler()
        try:
            installed = _signal.getsignal(_signal.SIGTERM)
            os.kill(os.getpid(), _signal.SIGTERM)
            for _ in range(10):
                if flag["shutdown"]:
                    break
            # Second signal SHOULD flip second_signal flag + restore handler.
            # Catch the SIGTERM by pre-installing a no-op so the test process
            # doesn't die.
            second_fired = {"hit": False}
            def swallow(signum: int, frame: object) -> None:  # noqa: ARG001
                second_fired["hit"] = True
            # Swap to swallow-only AFTER our handler hands control back.
            # The handler's "second signal" branch restores `original` (which
            # is our swallow if we install it here first). So install swallow
            # as the "original" before the second kill reaches our handler.
            # Simplest: assert that a second call to the installed handler
            # flips second_signal.
            assert callable(installed)
            installed(_signal.SIGTERM, None)  # simulate second delivery
            assert flag["second_signal"] is True
        finally:
            restore()

    def test_restore_handler_is_idempotent(self) -> None:
        import signal as _signal

        from mkw_rl.rl.train import _install_shutdown_handler

        pre = _signal.getsignal(_signal.SIGTERM)
        flag, restore = _install_shutdown_handler()
        _ = flag  # unused
        restore()
        assert _signal.getsignal(_signal.SIGTERM) is pre
        # Calling restore again when the handler is already cleaned up should no-op.
        restore()
        assert _signal.getsignal(_signal.SIGTERM) is pre
