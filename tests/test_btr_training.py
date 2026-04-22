"""Tests for ``src/mkw_rl/rl/train.py``.

Pure-function tests — no live Dolphin, no real env. Covers:
- Config loading from YAML (main + testing-subtree merge)
- Quantile Huber loss shapes + axis convention (Dabney eq. 10)
- Munchausen reward computation shapes
- Full per-timestep Munchausen-IQN loss end-to-end
- BTRAgent.build constructs successfully with a tiny config

Does NOT test:
- Rollout or env integration (requires live Dolphin — scripts/smoke_env.py
  pattern; this is the pass-5 smoke-test scope).
- Checkpoint save/load roundtrip (covered by torch's own serialization tests).
- Wandb logging (third-party dep; mocked via env var absence → CSV fallback).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
        """Higher PER weights should produce proportionally larger loss."""
        torch.manual_seed(0)
        B, nt, n_actions = 2, 4, 10
        args = dict(
            online_q=torch.randn(B, nt, n_actions),
            online_taus=torch.rand(B, nt, 1),
            target_q=torch.randn(B, nt, n_actions),
            actions=torch.randint(0, n_actions, (B,)),
            rewards=torch.randn(B),
            dones=torch.zeros(B, 1, 1, dtype=torch.bool),
            entropy_tau=0.03, munch_alpha=0.9, munch_lo=-1.0,
        )
        munch_r = _compute_munchausen_reward(
            args["online_q"], args["rewards"], args["actions"],
            args["entropy_tau"], args["munch_alpha"], args["munch_lo"],
        )

        loss1, _ = _compute_td_error_and_loss(
            args["online_q"], args["online_taus"], args["target_q"],
            args["actions"], munch_r, 0.99, args["dones"], torch.ones(B), 0.03,
        )
        loss10, _ = _compute_td_error_and_loss(
            args["online_q"], args["online_taus"], args["target_q"],
            args["actions"], munch_r, 0.99, args["dones"], torch.ones(B) * 10, 0.03,
        )
        # With 10× weights, mean loss should be ~10× larger.
        assert abs(loss10.item() / loss1.item() - 10.0) < 0.1


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
