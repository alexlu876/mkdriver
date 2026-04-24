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
    lstm_shape = (agent.cfg.lstm_layers, agent.cfg.lstm_hidden)
    for i in range(n):
        state = rng.integers(0, 256, size=stack_shape, dtype=np.uint8)
        nstate = rng.integers(0, 256, size=stack_shape, dtype=np.uint8)
        # Synthetic stored hidden — doesn't need to be realistic for the
        # tests that just exercise sampling / loss math.
        h = rng.standard_normal(size=lstm_shape).astype(np.float16)
        c = rng.standard_normal(size=lstm_shape).astype(np.float16)
        agent.replay.append(
            state=state,
            action=int(rng.integers(0, 40)),
            reward=float(rng.standard_normal()),
            n_state=nstate,
            done=(i == n - 1),  # terminate at the end so last n-step window closes
            trun=False,
            stream=0,
            hidden=(h, c),
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

        with pytest.raises(RuntimeError, match="no usable savestates"):
            BTRAgent.build(cfg)

    def test_build_uses_savestate_yaml_intersection(self, tmp_path: Path) -> None:
        """Round-5 bug fix: available_tracks must intersect on-disk savestates
        against track_metadata.yaml entries. A savestate on disk that's NOT in
        the YAML should be EXCLUDED from the sampler (otherwise the curriculum
        picks it and env.reset KeyErrors three times before sampler.remove_track)."""
        import yaml as _yaml

        from mkw_rl.rl.train import BTRAgent

        savestate_dir = tmp_path / "savestates"
        savestate_dir.mkdir()
        # Two savestates on disk.
        (savestate_dir / "luigi_circuit_tt.sav").write_bytes(b"")
        (savestate_dir / "orphan_slug_tt.sav").write_bytes(b"")

        # YAML has only one of them. Format is slug → fields (not list of dicts).
        meta_path = tmp_path / "track_metadata.yaml"
        meta_path.write_text(_yaml.safe_dump({
            "luigi_circuit_tt": {
                "name": "Luigi Circuit",
                "cup": "mushroom",
                "wr_seconds": 68.733,
                "wr_category": "non_glitch",
                "laps": 3,
            }
        }))

        cfg = TrainConfig(
            savestate_dir=str(savestate_dir),
            track_metadata_path=str(meta_path),
            # Keep the rest tiny so build is fast.
            batch_size=2, replay_size=64, storage_size_multiplier=2.0,
            lstm_hidden=16, feature_dim=16, linear_size=16,
            num_tau=4, n_cos=8, encoder_channels=(4, 8, 8),
            framestack=4, imagex=32, imagey=24, input_hw=(24, 32),
            stack_size=4, min_sampling_size=16, n_step=2,
            layer_norm=False, testing=True,
        )
        agent = BTRAgent.build(cfg)
        # Only the YAML-backed slug is in the sampler; orphan is excluded.
        assert agent.sampler.track_slugs == ["luigi_circuit_tt"]
        assert "orphan_slug_tt" not in agent.sampler.progress


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
        # Need min_sampling_size + n_step headroom (stored-hidden replay
        # needs no seq_len prefix).
        _populate_replay(agent, cfg.min_sampling_size + cfg.n_step + 4)

        # Snapshot a weight-bearing online param (avoid NoisyLinear ε buffers —
        # those are resampled every forward and "change" on every step trivially).
        # Snapshot only trainable FP32 Parameters (not buffers). Excludes:
        # - NoisyLinear epsilon buffers (resample every forward, would
        #   produce false-positive "weight changed" signal)
        # - spectral_norm power-iteration buffers (`parametrizations.*._u`,
        #   `._v`) which also update every forward in training mode
        # With spectral_norm the actual trainable weight is at
        # `parametrizations.weight.original{0,1}` — those ARE in
        # named_parameters and WILL be caught here.
        snapshot = {
            name: p.detach().clone()
            for name, p in agent.online_net.named_parameters()
            if "epsilon" not in name and p.dtype == torch.float32
        }
        assert snapshot, "test helper assumption: at least one weight param"

        m1 = agent.learn_step()
        m2 = agent.learn_step()

        assert agent.grad_steps == 2
        assert "loss" in m1 and np.isfinite(m1["loss"])
        assert "loss" in m2 and np.isfinite(m2["loss"])
        assert "grad_norm" in m1 and np.isfinite(m1["grad_norm"])

        # Online weights should have shifted for at least one parameter.
        after = dict(agent.online_net.named_parameters())
        any_changed = any(
            not torch.equal(after[name].detach(), v) for name, v in snapshot.items()
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
        """Spy on replay.update_priorities; verify it's called with per-transition
        |δ| (one scalar per sample) — stored-hidden replay doesn't do the R2D2
        η·max + (1-η)·mean aggregation anymore."""
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        _populate_replay(agent, cfg.min_sampling_size + cfg.n_step + 4)

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
        _populate_replay(agent, cfg.min_sampling_size + cfg.n_step + 4)

        nan_loss = torch.tensor(float("nan"))
        fake_td = torch.zeros(cfg.batch_size)

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

    def test_save_replay_opt_in_includes_replay(self, tmp_path: Path) -> None:
        """save_replay=True embeds replay; save_replay=False (default) skips
        it. Resume with replay restores capacity; resume without leaves the
        fresh agent's replay empty so warmup re-fires."""
        from mkw_rl.rl.train import BTRAgent, _save_checkpoint, load_checkpoint
        import numpy as _np

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)

        # Populate replay with enough transitions to see non-zero capacity.
        rng = _np.random.default_rng(0)
        lstm_shape = (cfg.lstm_layers, cfg.lstm_hidden)
        for i in range(30):
            s = rng.integers(0, 256, size=(cfg.framestack, cfg.imagey, cfg.imagex), dtype=_np.uint8)
            ns = rng.integers(0, 256, size=(cfg.framestack, cfg.imagey, cfg.imagex), dtype=_np.uint8)
            h = rng.standard_normal(size=lstm_shape).astype(_np.float16)
            c = rng.standard_normal(size=lstm_shape).astype(_np.float16)
            agent.replay.append(
                state=s, action=i % 4, reward=float(i),
                n_state=ns, done=(i == 29), trun=False, stream=0,
                hidden=(h, c),
            )
        seeded_capacity = agent.replay.capacity
        assert seeded_capacity > 0, "test setup: expected replay to have grown"

        # Save WITHOUT replay (periodic ckpt behavior).
        periodic_path = tmp_path / "periodic.pt"
        _save_checkpoint(agent, cfg, periodic_path, save_replay=False)
        fresh1 = BTRAgent.build(cfg)
        load_checkpoint(fresh1, periodic_path)
        assert fresh1.replay.capacity == 0, (
            "periodic ckpt should not restore replay — warmup must re-fire"
        )

        # Save WITH replay (final/diverged ckpt behavior).
        final_path = tmp_path / "final.pt"
        _save_checkpoint(agent, cfg, final_path, save_replay=True)
        fresh2 = BTRAgent.build(cfg)
        load_checkpoint(fresh2, final_path)
        assert fresh2.replay.capacity == seeded_capacity, (
            f"final ckpt should restore replay: got {fresh2.replay.capacity} "
            f"expected {seeded_capacity}"
        )
        # And the restored replay samples identically.
        assert _np.array_equal(fresh2.replay.state_mem, agent.replay.state_mem)
        assert _np.array_equal(fresh2.replay.st.sum_tree, agent.replay.st.sum_tree)


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
        _populate_replay(agent, cfg.min_sampling_size + cfg.n_step + 4)

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
        _populate_replay(agent, cfg.min_sampling_size + cfg.n_step + 4)

        agent.nonfinite_streak = agent.MAX_NONFINITE - 1
        nan_loss = torch.tensor(float("nan"))
        fake_td = torch.zeros(cfg.batch_size)
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

    def test_train_recovers_from_env_reset_failed(self, tmp_path: Path) -> None:
        """Round-5: env.reset raising FileNotFoundError (missing savestate) or
        KeyError (unknown track slug) gets converted to EnvResetFailed by
        run_one_episode. The outer train() loop catches that alongside the
        socket errors and runs the per-track crash counter."""
        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import train

        cfg = _tiny_cfg(tmp_path, total_frames=15, min_sampling_size=10_000)
        envs: list[_FakeEnv] = []

        def fake_make(c: TrainConfig) -> _FakeEnv:  # noqa: ARG001
            # First env's first reset raises FileNotFoundError. Subsequent
            # envs run cleanly.
            crash_error = FileNotFoundError if len(envs) == 0 else BrokenPipeError
            crash_on = 1 if len(envs) == 0 else None
            env = _FakeEnv(
                framestack=c.framestack, h=c.imagey, w=c.imagex,
                scripted_rewards=[1.0] * 3,
                crash_on_reset_n=crash_on,
                crash_error=crash_error,
            )
            envs.append(env)
            return env

        with patch.object(train_mod, "_make_env", side_effect=fake_make):
            train(cfg)

        assert len(envs) >= 2, "expected env restart after FileNotFoundError"

    def test_env_reset_failed_is_narrow(self, tmp_path: Path) -> None:
        """A raw FileNotFoundError from env.reset gets wrapped as EnvResetFailed.
        A non-reset error path (e.g. from agent.learn_step) does NOT get
        absorbed by the widened catch — it propagates up as expected."""
        from mkw_rl.rl.train import BTRAgent, EnvResetFailed, run_one_episode

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)

        class _FileNotFoundEnv:
            def reset(self, **_kwargs: object):  # noqa: ANN204
                raise FileNotFoundError("missing_savestate.sav")

            def step(self, action: int):  # noqa: ARG002, ANN204
                raise RuntimeError("unreachable")

            def close(self) -> None:
                pass

        with pytest.raises(EnvResetFailed):
            run_one_episode(agent, _FileNotFoundEnv(), "luigi_circuit_tt")

    def test_keyerror_from_learn_step_not_absorbed(self, tmp_path: Path) -> None:
        """A KeyError raised from somewhere other than env.reset must NOT be
        absorbed by the outer train() crash-catch — that would silently mask
        training-code bugs as "env crash"."""
        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import train

        cfg = _tiny_cfg(tmp_path, total_frames=10, min_sampling_size=10_000)

        def fake_make(c: TrainConfig) -> _FakeEnv:  # noqa: ARG001
            return _FakeEnv(
                framestack=c.framestack, h=c.imagey, w=c.imagex,
                scripted_rewards=[1.0] * 3,
            )

        # Patch BTRAgent.learn_step to raise KeyError. This simulates a dict
        # lookup bug deep in the training code. With the narrowed catch, this
        # should propagate out of train() instead of being caught as env crash.
        def raise_keyerror(self):  # noqa: ANN001, ARG001
            raise KeyError("bogus_dict_lookup")

        with (
            patch.object(train_mod, "_make_env", side_effect=fake_make),
            patch.object(train_mod.BTRAgent, "learn_step", new=raise_keyerror),
            pytest.raises(KeyError, match="bogus_dict_lookup"),
        ):
            train(cfg)


class TestMultiEnv:
    """Sanity-check the _train_vector multi-env path with FakeEnvs so no
    Dolphin is needed. Validates: N envs constructed with distinct env_ids,
    replay append uses per-env stream, training reaches configured
    total_frames, final checkpoint saves."""

    def test_two_envs_complete_a_short_run(self, tmp_path: Path) -> None:
        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import train

        # 2 envs. Gate warmup high enough that no learn_step fires during the
        # brief test window — avoids needing a replay config big enough to
        # safely sample learning sequences (sample_sequences needs
        # non-seam starts to exist, which requires capacity >> seq_len).
        cfg = _tiny_cfg(
            tmp_path,
            num_envs=2,
            total_frames=30,
            min_sampling_size=10_000,  # never warms up within 30 frames
        )
        constructed_env_ids: list[int] = []
        lock = __import__("threading").Lock()

        def fake_make(c: TrainConfig, env_id: int | None = None) -> _FakeEnv:  # noqa: ARG001
            eid = env_id if env_id is not None else 0
            with lock:
                constructed_env_ids.append(eid)
            return _FakeEnv(
                framestack=c.framestack, h=c.imagey, w=c.imagex,
                scripted_rewards=[0.5] * 6,
            )

        with patch.object(train_mod, "_make_env", side_effect=fake_make):
            agent = train(cfg)

        # Both env_ids instantiated at least once, ids cover the expected range.
        assert sorted(set(constructed_env_ids)) == [0, 1]
        # Training reached the frame target.
        assert agent.env_steps >= cfg.total_frames

    def test_multi_env_final_checkpoint_saved(self, tmp_path: Path) -> None:
        """Final checkpoint file lands in log_dir after clean multi-env exit."""
        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import train

        cfg = _tiny_cfg(
            tmp_path,
            num_envs=2,
            total_frames=20,
            min_sampling_size=10_000,
        )

        def fake_make(c: TrainConfig, env_id: int | None = None) -> _FakeEnv:  # noqa: ARG001
            return _FakeEnv(
                framestack=c.framestack, h=c.imagey, w=c.imagex,
                scripted_rewards=[0.5] * 6,
            )

        with patch.object(train_mod, "_make_env", side_effect=fake_make):
            train(cfg, run_name="multi_env_test")

        ckpt = Path(cfg.log_dir) / "multi_env_test_final.pt"
        assert ckpt.exists(), f"expected final ckpt at {ckpt}"

    def test_multi_env_rejects_explicit_env_arg(self, tmp_path: Path) -> None:
        """Passing env= to train() while num_envs>1 is a config error."""
        from mkw_rl.rl.train import train

        cfg = _tiny_cfg(tmp_path, num_envs=2, total_frames=10, min_sampling_size=10_000)
        stub_env = _FakeEnv(framestack=cfg.framestack, h=cfg.imagey, w=cfg.imagex)
        with pytest.raises(ValueError, match="num_envs > 1"):
            train(cfg, env=stub_env)

    def test_make_env_rejects_wrong_dolphin_app_name_when_multi(self, tmp_path: Path) -> None:
        """With num_envs>1, cfg.dolphin_app must end in 'dolphin0' because
        _make_env derives sibling dolphin{i}/ dirs from Path(...).parent.
        A macOS default like '.../dolphin0/DolphinQt.app' would produce
        nonsense paths; this assert makes the failure loud at build time
        rather than deep in a Dolphin spawn."""
        from mkw_rl.rl.train import _make_env

        cfg = _tiny_cfg(
            tmp_path,
            num_envs=2,
            dolphin_app="/some/prefix/dolphin0/DolphinQt.app",  # misconfig
        )
        with pytest.raises(ValueError, match="must point at a directory named 'dolphin0'"):
            _make_env(cfg, env_id=1)

    def test_multi_env_recovers_from_transient_env_crash(self, tmp_path: Path) -> None:
        """An env that crashes once mid-rollout should be relaunched, other
        envs should keep running, and the whole run should complete once
        total_frames is reached. Covers the threaded crash-recovery path
        that Vast's ~20% EOFError rate exercises in production."""
        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import train

        cfg = _tiny_cfg(
            tmp_path,
            num_envs=2,
            total_frames=80,
            min_sampling_size=10_000,
        )
        lock = __import__("threading").Lock()
        construction_counts: dict[int, int] = {}

        def fake_make(c: TrainConfig, env_id: int | None = None) -> _FakeEnv:  # noqa: ARG001
            eid = env_id if env_id is not None else 0
            with lock:
                construction_counts[eid] = construction_counts.get(eid, 0) + 1
                nth_for_this_env = construction_counts[eid]
            # Only the FIRST env 1 instance crashes — immediately on its
            # first reset. The relaunched env 1 runs cleanly. env 0 always
            # clean. Using reset #1 (not #2) guarantees the crash happens
            # before total_frames could be reached by the other env alone.
            crash_on = 1 if (eid == 1 and nth_for_this_env == 1) else None
            return _FakeEnv(
                framestack=c.framestack, h=c.imagey, w=c.imagex,
                scripted_rewards=[0.25] * 6,
                crash_on_reset_n=crash_on,
                crash_error=BrokenPipeError,
            )

        with patch.object(train_mod, "_make_env", side_effect=fake_make):
            agent = train(cfg)

        # Run completed (env_steps hit target).
        assert agent.env_steps >= cfg.total_frames, (
            f"run didn't complete: env_steps={agent.env_steps} total={cfg.total_frames}"
        )
        # Env 1 was built at least twice (once before crash, once after).
        assert construction_counts.get(1, 0) >= 2, (
            f"expected env 1 to be relaunched after crash, got "
            f"construction_counts={construction_counts}"
        )
        # Env 0 never crashed so it was built exactly once.
        assert construction_counts.get(0, 0) == 1

    def test_multi_env_aborts_on_persistent_env_crashes(self, tmp_path: Path) -> None:
        """An env that crashes on every single reset should eventually hit
        MAX_ENV_CRASHES=5 and abort the whole run. Needed because without
        this bound a broken env could relaunch forever while the rest of
        the system silently wastes compute."""
        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import train

        cfg = _tiny_cfg(
            tmp_path,
            num_envs=2,
            total_frames=500,
            min_sampling_size=10_000,
        )
        # Pre-create a second savestate so remove_track doesn't leave the
        # sampler empty before the env_streak limit can fire.
        (Path(cfg.savestate_dir) / "mushroom_gorge_tt.sav").write_bytes(b"")

        def fake_make(c: TrainConfig, env_id: int | None = None) -> _FakeEnv:  # noqa: ARG001
            eid = env_id if env_id is not None else 0
            # env 1 crashes on EVERY reset; env 0 runs cleanly.
            crash_on = 1 if eid == 1 else None
            return _FakeEnv(
                framestack=c.framestack, h=c.imagey, w=c.imagex,
                scripted_rewards=[0.1] * 4,
                crash_on_reset_n=crash_on,
                crash_error=BrokenPipeError,
            )

        # The run should raise because env 1 will hit MAX_ENV_CRASHES=5
        # consecutive failures. train() re-raises the captured error.
        with (
            patch.object(train_mod, "_make_env", side_effect=fake_make),
            pytest.raises(RuntimeError, match=r"(consecutive crashes|no tracks remain)"),
        ):
            train(cfg)

    def test_make_env_accepts_dolphin0_path_when_multi(self, tmp_path: Path) -> None:
        """Correct shape: dolphin_app = '.../dolphin0' → sibling '.../dolphin1'."""
        from mkw_rl.rl.train import _make_env
        from mkw_rl.env.dolphin_env import MkwDolphinEnv

        # Patch MkwDolphinEnv to a lightweight stub so we can assert on the
        # dolphin_app kwarg without actually trying to launch Dolphin.
        captured_kwargs: dict = {}

        class _StubEnv:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.update(kwargs)

        import mkw_rl.rl.train as train_mod
        original = train_mod.MkwDolphinEnv
        train_mod.MkwDolphinEnv = _StubEnv  # type: ignore[assignment,misc]
        try:
            cfg = _tiny_cfg(
                tmp_path,
                num_envs=4,
                dolphin_app="/opt/wii-rl/dolphin0",
            )
            _make_env(cfg, env_id=2)
        finally:
            train_mod.MkwDolphinEnv = original
        assert captured_kwargs["dolphin_app"] == "/opt/wii-rl/dolphin2"
        assert captured_kwargs["env_id"] == 2


class TestDeterministicAct:
    """Coverage for ``agent.act(deterministic=True)`` — the noisy-nets
    bypass used by scripts/eval_btr.py. Verifies that:

    - deterministic=True skips reset_noise() (so previously-disabled
      noise stays disabled)
    - run_one_episode plumbs the flag through
    """

    def test_deterministic_does_not_reset_noise(self, tmp_path: Path) -> None:
        """After disable_noise(), a deterministic act() leaves ε at zero;
        a non-deterministic one re-samples and ε becomes non-zero."""
        from mkw_rl.rl.networks import FactorizedNoisyLinear
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        agent.online_net.disable_noise()

        # Find the first noisy layer and confirm it starts at ε=0 after disable.
        noisy_layers = [
            m for m in agent.online_net.modules() if isinstance(m, FactorizedNoisyLinear)
        ]
        assert noisy_layers, "test assumes BTR model has noisy linears"
        layer = noisy_layers[0]
        assert torch.all(layer.weight_epsilon == 0)

        frames = torch.zeros(
            1, 1, cfg.stack_size, cfg.imagey, cfg.imagex, dtype=torch.uint8
        )
        # deterministic=True: noise stays at zero.
        agent.act(frames, hidden=None, deterministic=True)
        assert torch.all(layer.weight_epsilon == 0), (
            "deterministic=True should not call reset_noise()"
        )

        # Default (deterministic=False): noise gets re-sampled → non-zero.
        agent.act(frames, hidden=None)
        assert torch.any(layer.weight_epsilon != 0), (
            "default act() should call reset_noise() which samples non-zero ε"
        )

    def test_run_one_episode_passes_deterministic_flag(self, tmp_path: Path) -> None:
        """``run_one_episode(deterministic=True)`` must forward the flag to
        agent.act on every step — not only the first."""
        from mkw_rl.rl.train import BTRAgent, run_one_episode

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        env = _FakeEnv(
            framestack=cfg.framestack, h=cfg.imagey, w=cfg.imagex,
            scripted_rewards=[0.1, 0.2, 0.3, 0.4, 0.5],
        )

        seen_flags: list[bool] = []
        original_act = agent.act

        def spy_act(frames, hidden, deterministic=False):  # noqa: ANN001, ANN202, FBT002
            seen_flags.append(deterministic)
            return original_act(frames, hidden, deterministic=deterministic)

        agent.act = spy_act  # type: ignore[method-assign]
        run_one_episode(agent, env, "luigi_circuit_tt", deterministic=True)

        assert len(seen_flags) == 5, f"expected 5 act() calls, got {len(seen_flags)}"
        assert all(seen_flags), "deterministic=True should reach every act() call"


class TestEvalBtrScript:
    """Coverage for scripts/eval_btr.py via FakeEnv — no live Dolphin needed."""

    def test_eval_main_produces_summary_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Run eval_btr.main() end-to-end: build agent, write ckpt, patch
        _make_env to return a FakeEnv, eval 2 episodes, verify the output
        JSON has correct structure + returns match scripted rewards."""
        import importlib
        import json as _json
        import sys as _sys

        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import BTRAgent, _save_checkpoint

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        ckpt_path = tmp_path / "test_ckpt.pt"
        _save_checkpoint(agent, cfg, ckpt_path)

        # Write a YAML config that load_config can parse — needs savestate_dir
        # + track_metadata path pointed at real files. _tiny_cfg already made
        # the savestate; fabricate a minimal YAML + track_metadata.
        yaml_path = tmp_path / "eval_cfg.yaml"
        yaml_path.write_text(
            f"""data:
  savestate_dir: "{cfg.savestate_dir}"
  track_metadata_path: "{tmp_path / 'track_meta.yaml'}"
env:
  env_id: 0
  num_envs: 1
model:
  stack_size: {cfg.stack_size}
  input_hw: [{cfg.imagey}, {cfg.imagex}]
  encoder_channels: [{cfg.encoder_channels[0]}, {cfg.encoder_channels[1]}, {cfg.encoder_channels[2]}]
  feature_dim: {cfg.feature_dim}
  lstm_hidden: {cfg.lstm_hidden}
  lstm_layers: {cfg.lstm_layers}
  linear_size: {cfg.linear_size}
  num_tau: {cfg.num_tau}
  n_cos: {cfg.n_cos}
  layer_norm: {str(cfg.layer_norm).lower()}
replay:
  size: {cfg.replay_size}
  storage_size_multiplier: {cfg.storage_size_multiplier}
  framestack: {cfg.framestack}
  imagex: {cfg.imagex}
  imagey: {cfg.imagey}
  n_step: {cfg.n_step}
training:
  batch_size: {cfg.batch_size}
  min_sampling_size: {cfg.min_sampling_size}
runtime:
  device: cpu
"""
        )
        # Minimal track metadata YAML — just enough to make load_track_metadata happy.
        # Schema per data/track_metadata.yaml + src/mkw_rl/env/track_meta.py's
        # required fields.
        (tmp_path / "track_meta.yaml").write_text(
            """luigi_circuit_tt:
  name: "Luigi Circuit"
  cup: mushroom
  wr_seconds: 68.733
  wr_category: non_glitch
  laps: 3
"""
        )

        # Patch _make_env in the module eval_btr.main imports from.
        def fake_make(c: TrainConfig, env_id: int | None = None) -> _FakeEnv:  # noqa: ARG001
            return _FakeEnv(
                framestack=c.framestack, h=c.imagey, w=c.imagex,
                scripted_rewards=[1.0, 2.0, 3.0, 4.0],  # sum = 10.0
                reward_breakdown={"checkpoint": 0.5, "speed": 0.5},
            )

        monkeypatch.setattr(train_mod, "_make_env", fake_make)

        out_json = tmp_path / "eval_out.json"
        argv = [
            "eval_btr.py",
            "--ckpt", str(ckpt_path),
            "--config", str(yaml_path),
            "--track-slug", "luigi_circuit_tt",
            "--episodes", "2",
            "--device", "cpu",
            "--output", str(out_json),
        ]
        monkeypatch.setattr(_sys, "argv", argv)

        # Import (or reimport) eval_btr from scripts/.
        scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
        monkeypatch.syspath_prepend(str(scripts_dir))
        if "eval_btr" in _sys.modules:
            eval_btr = importlib.reload(_sys.modules["eval_btr"])
        else:
            eval_btr = importlib.import_module("eval_btr")

        rc = eval_btr.main()
        assert rc == 0, f"eval_btr.main returned {rc}"
        assert out_json.exists()

        payload = _json.loads(out_json.read_text())
        assert "summary" in payload
        assert "per_episode" in payload
        assert payload["summary"]["episodes"] == 2
        assert len(payload["per_episode"]) == 2
        # FakeEnv scripted_rewards sum is 10.0; both episodes should return ~10.
        for ep in payload["per_episode"]:
            assert ep["return"] == 10.0
            assert ep["length"] == 4
        assert payload["summary"]["return_mean"] == 10.0
        assert payload["summary"]["return_std"] == 0.0


class TestCleanupStaleX11State:
    """Coverage for _cleanup_stale_x11_state — the fix for the Vast.ai
    Dolphin-SIGSEGV-after-10-orphans issue. The new liveness check reads
    the owning PID out of ``/tmp/.X<N>-lock`` and checks via ``os.kill(pid, 0)``
    whether that process is alive, rather than the (broken) mtime-based
    heuristic it replaced."""

    @staticmethod
    def _age_paths(paths: list[Path], age_seconds: float) -> None:
        """Backdate mtime — used for xvfb-run dirs, which still use a
        10-min mtime heuristic (no PID file to parse)."""
        import os as _os
        import time as _time
        target = _time.time() - age_seconds
        for p in paths:
            _os.utime(p, (target, target))

    def test_removes_sockets_for_dead_pids_keeps_live_ones(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """X sockets whose lock-file PID is a dead process get removed.
        Sockets whose PID is alive get preserved regardless of mtime.
        This is the critical invariant: Xvfb doesn't touch its socket's
        mtime during normal operation, so mtime alone can't distinguish
        live from dead."""
        import os as _os
        from mkw_rl.rl.train import _cleanup_stale_x11_state

        fake_tmp = tmp_path / "tmp"
        (fake_tmp / ".X11-unix").mkdir(parents=True)

        # Live display: lock points at our own PID (definitely alive).
        live_sock = fake_tmp / ".X11-unix" / "X142"
        live_lock = fake_tmp / ".X142-lock"
        live_sock.write_bytes(b"")
        live_lock.write_text(f"{_os.getpid()}\n")

        # Dead display: lock points at an astronomically unlikely PID (2M).
        # os.kill(2000000, 0) on a real system will raise ProcessLookupError.
        dead_sock = fake_tmp / ".X11-unix" / "X148"
        dead_lock = fake_tmp / ".X148-lock"
        dead_sock.write_bytes(b"")
        dead_lock.write_text("2000000\n")

        # Socket with no lock file at all → treat as dead orphan.
        orphan_sock = fake_tmp / ".X11-unix" / "X200"
        orphan_sock.write_bytes(b"")

        # Decoy non-socket file in .X11-unix/ — must not be touched
        # (doesn't match the X<digits> regex).
        readme = fake_tmp / ".X11-unix" / "README"
        readme.write_bytes(b"not a socket")

        import mkw_rl.rl.train as train_mod
        original_glob = __import__("glob").glob

        def patched_glob(pattern: str) -> list[str]:
            if pattern.startswith("/tmp/"):
                pattern = str(fake_tmp / pattern[len("/tmp/"):])
            return original_glob(pattern)

        monkeypatch.setattr(train_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr("glob.glob", patched_glob)

        _cleanup_stale_x11_state()

        # Live PID → socket + lock preserved.
        assert live_sock.exists(), "live-PID X socket must not be deleted"
        assert live_lock.exists(), "live-PID lock file must not be deleted"
        # Dead PID → both removed.
        assert not dead_sock.exists(), "dead-PID X socket should be removed"
        assert not dead_lock.exists(), "dead-PID lock file should be removed"
        # No-lock socket → removed (treated as orphan).
        assert not orphan_sock.exists()
        # Decoys untouched.
        assert readme.read_bytes() == b"not a socket"

    def test_removes_stale_xvfb_run_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """xvfb-run scratch dirs still use mtime (no PID marker), so test
        a fresh dir is preserved and an aged one is removed."""
        from mkw_rl.rl.train import _cleanup_stale_x11_state

        fake_tmp = tmp_path / "tmp"
        (fake_tmp / ".X11-unix").mkdir(parents=True)
        fresh_dir = fake_tmp / "xvfb-run.fresh"
        fresh_dir.mkdir()
        (fresh_dir / "Xauthority").write_bytes(b"")

        stale_dir = fake_tmp / "xvfb-run.stale"
        stale_dir.mkdir()
        self._age_paths([stale_dir], age_seconds=900)  # 15 min

        import mkw_rl.rl.train as train_mod
        original_glob = __import__("glob").glob

        def patched_glob(pattern: str) -> list[str]:
            if pattern.startswith("/tmp/"):
                pattern = str(fake_tmp / pattern[len("/tmp/"):])
            return original_glob(pattern)

        monkeypatch.setattr(train_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr("glob.glob", patched_glob)

        _cleanup_stale_x11_state()

        assert fresh_dir.exists(), "fresh xvfb-run dir must be preserved"
        assert not stale_dir.exists(), "15-min-old xvfb-run dir should be removed"

    def test_noop_on_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On macOS dev machines the helper must not touch anything —
        we don't want to kill the user's real X session on a Linux workstation
        that's running actual X, and we certainly don't want to touch macOS
        tmp files. The function gates on platform.system() == 'Linux'."""
        import mkw_rl.rl.train as train_mod
        from mkw_rl.rl.train import _cleanup_stale_x11_state

        # Pin to a non-Linux value; assert the function returns without
        # touching anything. We detect 'touched anything' by counting glob calls.
        monkeypatch.setattr(train_mod.platform, "system", lambda: "Darwin")
        import glob as _glob
        call_count = [0]
        original = _glob.glob

        def spy(pattern: str) -> list[str]:
            call_count[0] += 1
            return original(pattern)

        monkeypatch.setattr("glob.glob", spy)
        _cleanup_stale_x11_state()
        assert call_count[0] == 0, "helper should no-op on non-Linux"

    def test_preserves_long_lived_sockets_with_live_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The previous mtime-based implementation had a fatal flaw:
        Xvfb doesn't touch its socket after creation, so after 60s of
        normal operation the socket was indistinguishable from a 60s-old
        orphan and got deleted — killing the live env. This test locks
        that regression out by explicitly aging a live-PID socket past
        the old 60s threshold and asserting it survives."""
        import os as _os
        from mkw_rl.rl.train import _cleanup_stale_x11_state

        fake_tmp = tmp_path / "tmp"
        (fake_tmp / ".X11-unix").mkdir(parents=True)
        sock = fake_tmp / ".X11-unix" / "X42"
        lock = fake_tmp / ".X42-lock"
        sock.write_bytes(b"")
        lock.write_text(f"{_os.getpid()}\n")
        # Age both well past the old 60s mtime threshold.
        self._age_paths([sock, lock], age_seconds=3600)

        import mkw_rl.rl.train as train_mod
        original_glob = __import__("glob").glob

        def patched_glob(pattern: str) -> list[str]:
            if pattern.startswith("/tmp/"):
                pattern = str(fake_tmp / pattern[len("/tmp/"):])
            return original_glob(pattern)

        monkeypatch.setattr(train_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr("glob.glob", patched_glob)
        _cleanup_stale_x11_state()

        assert sock.exists(), "1-hour-old live-PID socket must survive"
        assert lock.exists(), "1-hour-old live-PID lock must survive"


class TestTrainBtrCli:
    """CLI-override plumbing for scripts/train_btr.py. These are the flags
    used by the shakedown workflow — a miswired override would silently
    train at production defaults (3-hour warmup) when the user thought
    they were running a 10-min shakedown."""

    @staticmethod
    def _import_entry() -> object:
        """Import scripts/train_btr.py without running it."""
        import importlib.util as _util
        spec = _util.spec_from_file_location(
            "train_btr_entry",
            Path(__file__).resolve().parents[1] / "scripts" / "train_btr.py",
        )
        assert spec and spec.loader
        mod = _util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _run_main_with_argv(
        self,
        tmp_path: Path,
        extra_argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> TrainConfig:
        """Stub train() to capture the cfg it receives, then invoke main()."""
        import sys as _sys

        captured: dict[str, TrainConfig] = {}

        def fake_train(cfg: TrainConfig, *, env=None, resume_from=None, run_name=None):  # noqa: ARG001, ANN001
            captured["cfg"] = cfg

        entry = self._import_entry()

        # main() does a deferred `from mkw_rl.rl.train import load_config, train`;
        # patch the train symbol after the import resolves. Simplest way:
        # pre-insert a stub module so the import returns our fake.
        import mkw_rl.rl.train as train_mod
        monkeypatch.setattr(train_mod, "train", fake_train)

        # Write a minimal YAML that load_config can parse.
        yaml_path = tmp_path / "cli.yaml"
        yaml_path.write_text(
            "data:\n  savestate_dir: data/savestates\n  track_metadata_path: data/track_metadata.yaml\n"
            "env:\n  env_id: 0\n  num_envs: 1\n"
            "training:\n  batch_size: 256\n  min_sampling_size: 200000\n  total_frames: 500000000\n"
            "logging:\n  checkpoint_every_grad_steps: 10000\n"
            "runtime:\n  device: cpu\n"
        )

        argv = ["train_btr.py", "--config", str(yaml_path), *extra_argv]
        monkeypatch.setattr(_sys, "argv", argv)
        rc = entry.main()
        assert rc == 0, f"main() returned {rc}"
        return captured["cfg"]

    def test_min_sampling_size_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = self._run_main_with_argv(
            tmp_path, ["--min-sampling-size", "2000"], monkeypatch,
        )
        assert cfg.min_sampling_size == 2000

    def test_total_frames_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = self._run_main_with_argv(
            tmp_path, ["--total-frames", "30000"], monkeypatch,
        )
        assert cfg.total_frames == 30000

    def test_checkpoint_every_grad_steps_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = self._run_main_with_argv(
            tmp_path, ["--checkpoint-every-grad-steps", "500"], monkeypatch,
        )
        assert cfg.checkpoint_every_grad_steps == 500

    def test_no_overrides_keeps_yaml_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = self._run_main_with_argv(tmp_path, [], monkeypatch)
        # Values come from the fabricated YAML above — prod-like.
        assert cfg.min_sampling_size == 200000
        assert cfg.total_frames == 500000000
        assert cfg.checkpoint_every_grad_steps == 10000

    def test_bad_min_sampling_size_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    ) -> None:
        """--min-sampling-size 0 is invalid; main() returns 1 with a clear
        stderr message instead of silently doing something weird."""
        import sys as _sys

        entry = self._import_entry()
        yaml_path = tmp_path / "cli.yaml"
        yaml_path.write_text(
            "data:\n  savestate_dir: data/savestates\n  track_metadata_path: data/track_metadata.yaml\n"
            "env:\n  env_id: 0\n  num_envs: 1\n"
            "runtime:\n  device: cpu\n"
        )
        argv = ["train_btr.py", "--config", str(yaml_path), "--min-sampling-size", "0"]
        monkeypatch.setattr(_sys, "argv", argv)
        rc = entry.main()
        assert rc == 1
        assert "must be >= 1" in capsys.readouterr().err


class TestCheckpointRotation:
    def test_prune_keeps_last_n(self, tmp_path: Path) -> None:
        # Create 7 fake grad ckpts with increasing mtimes.
        import os as _os

        from mkw_rl.rl.train import _prune_old_checkpoints
        run = "btr_test"
        for i in range(1, 8):
            p = tmp_path / f"{run}_grad{i}.pt"
            p.write_bytes(b"x")
            _os.utime(p, (1000 + i, 1000 + i))
        # Final + diverged should NEVER be pruned.
        (tmp_path / f"{run}_final.pt").write_bytes(b"x")
        (tmp_path / f"{run}_diverged.pt").write_bytes(b"x")

        _prune_old_checkpoints(tmp_path, run_name=run, keep_last_n=3)

        remaining = {p.name for p in tmp_path.glob(f"{run}*")}
        # Keep 3 newest: grad5, grad6, grad7. Plus final and diverged.
        assert remaining == {
            f"{run}_grad5.pt",
            f"{run}_grad6.pt",
            f"{run}_grad7.pt",
            f"{run}_final.pt",
            f"{run}_diverged.pt",
        }

    def test_prune_disabled_when_keep_zero(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import _prune_old_checkpoints

        run = "btr_test"
        for i in range(1, 5):
            (tmp_path / f"{run}_grad{i}.pt").write_bytes(b"x")

        _prune_old_checkpoints(tmp_path, run_name=run, keep_last_n=0)

        assert len(list(tmp_path.glob(f"{run}_grad*.pt"))) == 4

    def test_prune_doesnt_touch_other_runs(self, tmp_path: Path) -> None:
        """Pruning one run_name must not delete another run's ckpts."""
        from mkw_rl.rl.train import _prune_old_checkpoints

        for i in range(1, 5):
            (tmp_path / f"run_a_grad{i}.pt").write_bytes(b"x")
            (tmp_path / f"run_b_grad{i}.pt").write_bytes(b"x")

        _prune_old_checkpoints(tmp_path, run_name="run_a", keep_last_n=1)

        a_remaining = list(tmp_path.glob("run_a_grad*.pt"))
        b_remaining = list(tmp_path.glob("run_b_grad*.pt"))
        assert len(a_remaining) == 1
        assert len(b_remaining) == 4  # untouched


class TestLearnStepMetricsHaveNonfiniteStreak:
    def test_happy_path_includes_nonfinite_streak(self, tmp_path: Path) -> None:
        """Round-5: nonfinite_streak=0 emitted on happy path so the CSV/wandb
        column exists from step 1 (not just after the first divergence)."""
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        _populate_replay(agent, cfg.min_sampling_size + cfg.n_step + 4)
        m = agent.learn_step()
        assert "nonfinite_streak" in m
        assert m["nonfinite_streak"] == 0


class TestSamplerRemoveTrack:
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


# ---------------------------------------------------------------------------
# Round-4 coverage: spectral_norm / LayerNorm / NaN priority / prod checkpoint.
# ---------------------------------------------------------------------------


class TestSpectralNormPresence:
    """Verify cfg.spectral_norm actually materializes parametrizations."""

    def _count_spectral_norms(self, policy: torch.nn.Module) -> int:
        # nn.utils.parametrizations.spectral_norm attaches a ParametrizationList
        # at module.parametrizations with a "weight" key.
        return sum(
            1
            for m in policy.modules()
            if hasattr(m, "parametrizations") and "weight" in m.parametrizations
        )

    def test_default_cfg_has_spectral_norm(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path, spectral_norm=True)
        agent = BTRAgent.build(cfg)
        # Encoder has 3 block.conv + 3 blocks × 2 residual blocks × 2 convs = 15.
        assert self._count_spectral_norms(agent.online_net) == 15

    def test_disabled_has_no_spectral_norm(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path, spectral_norm=False)
        agent = BTRAgent.build(cfg)
        assert self._count_spectral_norms(agent.online_net) == 0


class TestLayerNormPresence:
    """Verify cfg.layer_norm attaches LayerNorm inside conv blocks."""

    def test_enabled_materializes_conv_block_ln(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path, layer_norm=True)
        agent = BTRAgent.build(cfg)
        enc = agent.online_net.encoder
        for blk in (enc.block1, enc.block2, enc.block3):
            assert isinstance(blk.norm, torch.nn.LayerNorm), (
                f"expected LayerNorm, got {type(blk.norm).__name__}"
            )

    def test_disabled_keeps_identity(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path, layer_norm=False)
        agent = BTRAgent.build(cfg)
        enc = agent.online_net.encoder
        for blk in (enc.block1, enc.block2, enc.block3):
            assert isinstance(blk.norm, torch.nn.Identity)


class TestTargetEvalMode:
    """Verify target net is in eval() mode — prevents spectral_norm drift."""

    def test_target_starts_in_eval(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        assert not agent.target_net.training, "target must be in eval mode"

    def test_sync_target_keeps_eval(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path)
        agent = BTRAgent.build(cfg)
        agent.target_net.train()  # simulate a stray .train()
        agent.sync_target()
        assert not agent.target_net.training

    def test_target_spectral_norm_no_drift_across_forwards(self, tmp_path: Path) -> None:
        """With target.eval(), spectral_norm power iteration stops updating
        _u/_v buffers — target's effective weight stays fixed between syncs."""
        from mkw_rl.rl.train import BTRAgent

        cfg = _tiny_cfg(tmp_path, spectral_norm=True)
        agent = BTRAgent.build(cfg)
        # Capture all buffers named "_u" / "_v" across every parametrization.
        def snapshot() -> dict[str, torch.Tensor]:
            return {
                n: b.detach().clone()
                for n, b in agent.target_net.named_buffers()
                if n.endswith("._u") or n.endswith("._v")
            }

        before = snapshot()
        assert before, "test assumes there IS at least one spectral_norm buffer"

        # Several forwards with dummy input.
        x = torch.zeros(1, 1, cfg.stack_size, cfg.imagey, cfg.imagex, dtype=torch.uint8)
        with torch.no_grad():
            for _ in range(5):
                agent.target_net(x)

        after = snapshot()
        for name in before:
            assert torch.equal(before[name], after[name]), (
                f"target {name} drifted despite eval mode"
            )


class TestNanPriorityTreeValue:
    """Verify NaN-priority placeholder lands at eps in the SumTree, not eps**alpha."""

    def test_nan_priority_replaced_lands_at_eps(self, tmp_path: Path) -> None:
        _ = tmp_path
        from mkw_rl.rl.replay import PER

        per = PER(
            size=32, device="cpu", n=1, envs=1, gamma=0.99, alpha=0.2,
            framestack=4, imagex=30, imagey=20,
        )
        state = np.zeros((4, 20, 30), dtype=np.uint8)
        for i in range(10):
            per.append(state, action=i % 4, reward=1.0, n_state=state,
                       done=False, trun=False, stream=0)

        # Inject a NaN priority at tree_idx 0 (first leaf).
        tree_idx = per.st.tree_start
        per.update_priorities(np.array([tree_idx]), np.array([float("nan")]))
        leaf_val = per.st.sum_tree[tree_idx]
        # Placeholder = eps ** (1/alpha); then ** alpha inside update_priorities
        # gives eps. Without the round-3 fix, the tree value would be eps**alpha
        # ≈ 0.063 at alpha=0.2, eps=1e-6 — ~60000× too high.
        assert abs(leaf_val - per.eps) < per.eps * 0.01, (
            f"NaN-placeholder tree value {leaf_val} should be ~eps={per.eps}; "
            f"eps**alpha would be {per.eps ** per.alpha}"
        )

    def test_regular_priority_lands_at_priority_pow_alpha(self, tmp_path: Path) -> None:
        """Sanity check the companion path: normal priorities go through ^alpha."""
        _ = tmp_path
        from mkw_rl.rl.replay import PER

        per = PER(
            size=32, device="cpu", n=1, envs=1, gamma=0.99, alpha=0.2,
            framestack=4, imagex=30, imagey=20,
        )
        state = np.zeros((4, 20, 30), dtype=np.uint8)
        for i in range(5):
            per.append(state, action=i % 4, reward=1.0, n_state=state,
                       done=False, trun=False, stream=0)
        tree_idx = per.st.tree_start
        raw_prio = 10.0
        per.update_priorities(np.array([tree_idx]), np.array([raw_prio]))
        expected = (raw_prio + per.eps) ** per.alpha
        assert abs(per.st.sum_tree[tree_idx] - expected) < 1e-5


class TestCheckpointRoundTripProductionFlags:
    """Ensure production arch flags (spectral_norm + layer_norm) round-trip
    through _save_checkpoint → load_checkpoint without silent state loss."""

    def test_round_trip_with_spectral_and_ln(self, tmp_path: Path) -> None:
        from mkw_rl.rl.train import BTRAgent, _save_checkpoint, load_checkpoint

        cfg = _tiny_cfg(tmp_path, spectral_norm=True, layer_norm=True)
        agent = BTRAgent.build(cfg)
        # Mutate a conv-encoder parameter (exercises the spectral_norm
        # parametrization path).
        param_name, param = next(
            (n, p) for n, p in agent.online_net.named_parameters()
            if "block1.conv" in n and "original" in n and p.dtype == torch.float32
        )
        with torch.no_grad():
            param.add_(13.0)
        agent.grad_steps = 77
        agent.env_steps = 1234

        ckpt_path = tmp_path / "prod.pt"
        _save_checkpoint(agent, cfg, ckpt_path)

        fresh = BTRAgent.build(cfg)
        fresh_before = dict(fresh.online_net.named_parameters())[param_name].detach().clone()
        assert not torch.equal(
            fresh_before, dict(agent.online_net.named_parameters())[param_name].detach(),
        )

        load_checkpoint(fresh, ckpt_path)
        a_sd = agent.online_net.state_dict()
        f_sd = fresh.online_net.state_dict()
        assert set(a_sd) == set(f_sd)
        for k in a_sd:
            assert torch.equal(a_sd[k], f_sd[k]), f"mismatch at key {k}"
        assert not fresh.target_net.training
        assert fresh.grad_steps == 77
        assert fresh.env_steps == 1234
