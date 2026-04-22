"""BTR training: recurrent IQN-Munchausen-PER on multi-track MKWii.

Orchestrates the components built in passes 1-4:
- ``BTRPolicy`` (IMPALA+LSTM+IQN heads, pass 2)
- ``PER`` with ``sample_sequences()`` (R2D2 recurrent replay, passes 1+3)
- ``ProgressWeightedTrackSampler`` (pass 4)
- ``MkwDolphinEnv`` (phase 2.1)

Ports VIPTankz's ``Agent.learn_call`` (BTR.py:976-1090) with the HIGH
findings from the 2026-04-21 forensic audit applied inline (see the
``_quantile_huber_loss`` and ``_compute_priority`` docstrings):

1. Quantile-Huber axis convention follows Dabney et al. 2018 eq. 10
   (sum over target-tau, mean over online-tau) rather than VIPTankz's
   swapped axes. With ``num_tau_online == num_tau_target`` this is
   mathematically identical; the correct convention matters if we ever
   decouple the two tau counts.
2. Priority signal uses ``mean(dim=tau).mean(dim=tau)`` (scale-invariant)
   rather than ``sum(dim=tau).mean(dim=tau)`` which scales with num_tau.
3. Sequence-level priority aggregation follows R2D2 §2.3 eq. 1:
   ``η·max_t|δ_t| + (1-η)·mean_t|δ_t|`` with ``η=0.9``.
4. Target-net update cadence defaulted to 200 grad steps (MKWii is
   non-stationary; VIPTankz's 500 was Atari-tuned).
5. Exploration: we drop the 100M-frame ε-disable schedule — noisy-nets
   handles exploration by construction.
"""

from __future__ import annotations

import copy
import csv
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from mkw_rl.env.dolphin_env import MkwDolphinEnv, available_tracks
from mkw_rl.rl.model import BTRConfig, BTRPolicy
from mkw_rl.rl.replay import PER
from mkw_rl.rl.track_sampler import (
    ProgressWeightedTrackSampler,
    TrackSamplerConfig,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Flattened training config — one dataclass per section of btr.yaml."""

    # data
    savestate_dir: str = "data/savestates"
    track_metadata_path: str = "data/track_metadata.yaml"

    # env
    env_id: int = 0

    # model (forwarded into BTRConfig)
    stack_size: int = 4
    input_hw: tuple[int, int] = (75, 140)
    encoder_channels: tuple[int, int, int] = (16, 32, 32)
    feature_dim: int = 256
    lstm_hidden: int = 512
    lstm_layers: int = 1
    linear_size: int = 512
    num_tau: int = 8
    n_cos: int = 64
    layer_norm: bool = True

    # replay
    replay_size: int = 1_048_576
    storage_size_multiplier: float = 1.75
    n_step: int = 3
    gamma: float = 0.997
    per_alpha: float = 0.2
    per_beta: float = 0.4
    framestack: int = 4
    imagex: int = 140
    imagey: int = 75

    # training
    batch_size: int = 256
    lr: float = 1e-4
    grad_clip: float = 10.0
    replay_ratio: int = 1
    target_replace_grad_steps: int = 200
    min_sampling_size: int = 200_000
    total_frames: int = 500_000_000
    burn_in_len: int = 20
    learning_seq_len: int = 40
    priority_eta: float = 0.9
    entropy_tau: float = 0.03
    munch_alpha: float = 0.9
    munch_lo: float = -1.0

    # sampler
    sampler_ema_alpha: float = 0.05
    sampler_epsilon: float = 0.1
    sampler_cold_start_progress: float = 0.0

    # logging
    wandb_project: str = "mkw-rl"
    log_dir: str = "runs/btr"
    log_every_grad_steps: int = 100
    checkpoint_every_grad_steps: int = 10_000

    # runtime
    device: str = "cpu"
    seed: int = 0

    # testing flag — not loaded from YAML directly; set by CLI.
    testing: bool = False


def _deep_update(base: dict, overrides: dict) -> dict:
    """Nested dict merge — overrides take precedence."""
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, testing: bool = False) -> TrainConfig:
    """Load btr.yaml and flatten into a ``TrainConfig``.

    If ``testing`` is True, merge the ``testing:`` subtree on top of the
    main sections before flattening — enables small/fast config via
    ``scripts/train_btr.py --testing`` without duplicating the full YAML.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if testing and "testing" in raw:
        for section in ("model", "replay", "training", "logging", "runtime"):
            if section in raw["testing"]:
                raw[section] = _deep_update(raw.get(section, {}), raw["testing"][section])

    # Flatten the sections into kwargs for TrainConfig.
    kw: dict[str, Any] = {}
    for src, dst in (
        ("savestate_dir", "savestate_dir"),
        ("track_metadata_path", "track_metadata_path"),
    ):
        if src in raw.get("data", {}):
            kw[dst] = raw["data"][src]
    if "env" in raw:
        for k in ("env_id",):
            if k in raw["env"]:
                kw[k] = raw["env"][k]
    if "model" in raw:
        for k in (
            "stack_size", "feature_dim", "lstm_hidden", "lstm_layers",
            "linear_size", "num_tau", "n_cos", "layer_norm",
        ):
            if k in raw["model"]:
                kw[k] = raw["model"][k]
        if "input_hw" in raw["model"]:
            kw["input_hw"] = tuple(raw["model"]["input_hw"])
        if "encoder_channels" in raw["model"]:
            kw["encoder_channels"] = tuple(raw["model"]["encoder_channels"])
    if "replay" in raw:
        for src_k, dst_k in (
            ("size", "replay_size"),
            ("storage_size_multiplier", "storage_size_multiplier"),
            ("n_step", "n_step"),
            ("gamma", "gamma"),
            ("per_alpha", "per_alpha"),
            ("per_beta", "per_beta"),
            ("framestack", "framestack"),
            ("imagex", "imagex"),
            ("imagey", "imagey"),
        ):
            if src_k in raw["replay"]:
                kw[dst_k] = raw["replay"][src_k]
    if "training" in raw:
        for k in (
            "batch_size", "lr", "grad_clip", "replay_ratio",
            "target_replace_grad_steps", "min_sampling_size", "total_frames",
            "burn_in_len", "learning_seq_len", "priority_eta",
            "entropy_tau", "munch_alpha", "munch_lo",
        ):
            if k in raw["training"]:
                kw[k] = raw["training"][k]
    if "sampler" in raw:
        for src_k, dst_k in (
            ("ema_alpha", "sampler_ema_alpha"),
            ("epsilon", "sampler_epsilon"),
            ("cold_start_progress", "sampler_cold_start_progress"),
        ):
            if src_k in raw["sampler"]:
                kw[dst_k] = raw["sampler"][src_k]
    if "logging" in raw:
        for k in ("wandb_project", "log_dir", "log_every_grad_steps", "checkpoint_every_grad_steps"):
            if k in raw["logging"]:
                kw[k] = raw["logging"][k]
    if "runtime" in raw:
        for k in ("device", "seed"):
            if k in raw["runtime"]:
                kw[k] = raw["runtime"][k]

    kw["testing"] = testing
    return TrainConfig(**kw)


# ---------------------------------------------------------------------------
# Logger — wandb if WANDB_API_KEY is set, else CSV.
# ---------------------------------------------------------------------------


class Logger(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...


class _CSVLogger:
    """Simple append-only CSV logger. One row per log() call, columns grow
    as new metric keys appear (older rows stay NaN-padded on read)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._columns: list[str] = []
        self._fh = open(path, "a", buffering=1)  # line-buffered
        self._writer: csv.writer | None = None

    def log(self, metrics: dict[str, float], step: int) -> None:
        row = {"step": step, **metrics}
        new_cols = [k for k in row if k not in self._columns]
        if new_cols:
            self._columns.extend(new_cols)
            # Rewrite header — acceptable since this is append-only for small runs.
            self._fh.write("# " + ",".join(self._columns) + "\n")
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=self._columns)
        self._writer.writerow({k: row.get(k, "") for k in self._columns})

    def close(self) -> None:
        self._fh.close()


class _WandbLogger:
    """Thin wandb wrapper — caller provides project/run_name via config."""

    def __init__(self, project: str, run_name: str, config: dict) -> None:
        import wandb  # noqa: PLC0415 — optional dep
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=config)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._run.finish()


def make_logger(cfg: TrainConfig, run_name: str) -> Logger:
    if os.environ.get("WANDB_API_KEY"):
        try:
            return _WandbLogger(cfg.wandb_project, run_name, cfg.__dict__)
        except Exception as e:  # noqa: BLE001
            log.warning("wandb init failed (%s); falling back to CSV", e)
    csv_path = Path(cfg.log_dir) / f"{run_name}.csv"
    log.info("using CSV logger at %s", csv_path)
    return _CSVLogger(csv_path)


# ---------------------------------------------------------------------------
# Loss: Munchausen-IQN with quantile Huber.
# ---------------------------------------------------------------------------


def _quantile_huber_loss(
    td_error: torch.Tensor,
    taus_online: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss per Dabney et al. 2018 eq. 10.

    Args:
        td_error: shape ``(B, num_tau_online, num_tau_target)``.
            δ_ij = target_j - expected_i (broadcast difference).
        taus_online: shape ``(B, num_tau_online, 1)``. τ values of the online
            quantiles — used to weight the asymmetric Huber.
        kappa: Huber threshold (default 1.0). Divisor stays 1.0 when kappa=1;
            if kappa is ever changed, the caller needs to divide by kappa
            to keep the paper's normalization.

    Returns:
        Per-batch loss, shape ``(B,)``.

    Axis convention (fixed from VIPTankz audit): sum over the target-τ axis,
    mean over the online-τ axis. VIPTankz's BTR.py:1076 has these swapped
    — mathematically identical when num_tau_online == num_tau_target but
    breaks if they're ever decoupled (as in some IQN variants using
    N_online < N_target).
    """
    abs_delta = td_error.abs()
    huber = torch.where(
        abs_delta <= kappa,
        0.5 * td_error.pow(2),
        kappa * (abs_delta - 0.5 * kappa),
    )
    # Asymmetric weighting from the quantile regression loss.
    indicator = (td_error.detach() < 0).float()
    weight = (taus_online - indicator).abs()
    quantile_l = weight * huber / kappa
    # Per Dabney eq. 10: (1/N) · sum_j sum_i ρ_τ_i(δ_ij). Sum over target (axis 2),
    # mean over online (axis 1). VIPTankz's code does the reverse; with
    # N_online == N_target the results match, but we use Dabney's convention
    # for future-proofing.
    return quantile_l.sum(dim=2).mean(dim=1)  # (B,)


def _compute_td_error_and_loss(
    online_quantiles: torch.Tensor,
    online_taus: torch.Tensor,
    target_quantiles: torch.Tensor,
    actions: torch.Tensor,
    munchausen_reward: torch.Tensor,
    gamma_n: float,
    dones: torch.Tensor,
    weights: torch.Tensor,
    entropy_tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full Munchausen-IQN loss for one timestep of a batch.

    Shapes:
        online_quantiles: (B, num_tau, n_actions)
        online_taus: (B, num_tau, 1)
        target_quantiles: (B, num_tau, n_actions) — from target net at n_state
        actions: (B,) — the action taken at this timestep
        munchausen_reward: (B, 1, 1) — reward + α · clamp(log π(a|s))
        gamma_n: γ^n_step scalar
        dones: (B, 1, 1) — whether the n-step window terminates
        weights: (B,) — PER IS weights
        entropy_tau: Munchausen τ (softmax temperature)

    Returns:
        (loss_scalar, per_sample_td_abs_mean) — loss for backward, |δ|
        aggregated per sample for priority update.
    """
    # Target net's soft value at next-state.
    q_t_n = target_quantiles.mean(dim=1)  # (B, n_actions)
    v_next = q_t_n.max(1)[0].unsqueeze(-1)
    log_sum = torch.logsumexp((q_t_n - v_next) / entropy_tau, dim=1).unsqueeze(-1)
    tau_log_pi_next = (q_t_n - v_next - entropy_tau * log_sum).unsqueeze(1)
    # (B, 1, n_actions)
    pi_target = F.softmax(q_t_n / entropy_tau, dim=1).unsqueeze(1)
    # (B, 1, n_actions)

    # Q_target: E_{a'}[π(a'|s') · (Q(s',a') - τ log π(a'|s'))], discounted.
    q_target = (
        gamma_n
        * (pi_target * (target_quantiles - tau_log_pi_next) * (~dones)).sum(2)
    ).unsqueeze(1)
    # (B, 1, num_tau)

    q_targets = munchausen_reward + q_target  # (B, 1, num_tau)

    # Expected Q at the action we took.
    actions_b = actions.unsqueeze(1).unsqueeze(-1)  # (B, 1, 1)
    q_expected = online_quantiles.gather(
        2, actions_b.expand(-1, online_quantiles.shape[1], 1)
    )
    # (B, num_tau, 1)

    td_error = q_targets - q_expected  # (B, num_tau, num_tau) via broadcast

    # Per-sample abs-TD for priority (scale-invariant mean-mean instead of
    # VIPTankz's sum-mean which scales with num_tau).
    loss_v = td_error.abs().mean(dim=1).mean(dim=1).detach()  # (B,)

    # Quantile-Huber loss, PER-weighted.
    loss = _quantile_huber_loss(td_error, online_taus) * weights  # (B,)
    return loss.mean(), loss_v


def _compute_munchausen_reward(
    online_quantiles_detached: torch.Tensor,
    rewards: torch.Tensor,
    actions: torch.Tensor,
    entropy_tau: float,
    munch_alpha: float,
    munch_lo: float,
) -> torch.Tensor:
    """Munchausen reward bonus: r + α · clamp(τ · log π(a|s), lo, 0).

    See Vieillard et al. 2020 "Munchausen Reinforcement Learning".
    Here π is the soft Boltzmann policy of the ONLINE network at the
    current state (not target) — this is the Munchausen trick.

    Args:
        online_quantiles_detached: (B, num_tau, n_actions) — online Q at s, no grad.
        rewards: (B,) — raw environment reward (n-step discounted return).
        actions: (B,) — taken action.
        entropy_tau, munch_alpha, munch_lo: Munchausen hyperparameters.

    Returns: shape (B, 1, 1), added to target via broadcast.
    """
    q_mean = online_quantiles_detached.mean(dim=1)  # (B, n_actions)
    v = q_mean.max(1)[0].unsqueeze(-1)
    tau_log_pik = q_mean - v - entropy_tau * torch.logsumexp(
        (q_mean - v) / entropy_tau, dim=1
    ).unsqueeze(-1)
    # (B, n_actions)
    log_pi_a = tau_log_pik.gather(1, actions.unsqueeze(1))  # (B, 1)
    bonus = torch.clamp(log_pi_a, min=munch_lo, max=0.0)
    return (rewards.unsqueeze(-1) + munch_alpha * bonus).unsqueeze(-1)  # (B, 1, 1)


# ---------------------------------------------------------------------------
# Agent: policy + target + optimizer bundle.
# ---------------------------------------------------------------------------


@dataclass
class BTRAgent:
    """Training-time wrapper around policy + target + optimizer + replay."""

    cfg: TrainConfig
    online_net: BTRPolicy
    target_net: BTRPolicy
    optimizer: torch.optim.Optimizer
    replay: PER
    sampler: ProgressWeightedTrackSampler
    device: torch.device

    grad_steps: int = field(default=0)
    env_steps: int = field(default=0)

    @classmethod
    def build(cls, cfg: TrainConfig) -> BTRAgent:
        device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        model_cfg = BTRConfig(
            stack_size=cfg.stack_size,
            input_hw=cfg.input_hw,
            encoder_channels=cfg.encoder_channels,
            feature_dim=cfg.feature_dim,
            lstm_hidden=cfg.lstm_hidden,
            lstm_layers=cfg.lstm_layers,
            linear_size=cfg.linear_size,
            num_tau=cfg.num_tau,
            n_cos=cfg.n_cos,
            layer_norm=cfg.layer_norm,
        )
        online = BTRPolicy(model_cfg).to(device)
        target = copy.deepcopy(online)
        for p in target.parameters():
            p.requires_grad = False
        target.disable_noise()  # target uses greedy (non-noisy) evaluation

        optimizer = torch.optim.Adam(
            online.parameters(),
            lr=cfg.lr,
            eps=0.005 / cfg.batch_size,
        )

        replay = PER(
            size=cfg.replay_size,
            device=device,
            n=cfg.n_step,
            envs=1,
            gamma=cfg.gamma,
            alpha=cfg.per_alpha,
            beta=cfg.per_beta,
            framestack=cfg.framestack,
            imagex=cfg.imagex,
            imagey=cfg.imagey,
            storage_size_multiplier=cfg.storage_size_multiplier,
        )

        tracks = available_tracks(cfg.savestate_dir)
        if not tracks:
            raise RuntimeError(
                f"no savestates found at {cfg.savestate_dir!r}; "
                "record some via scripts/record_savestates.py first"
            )
        sampler = ProgressWeightedTrackSampler(
            track_slugs=tracks,
            config=TrackSamplerConfig(
                ema_alpha=cfg.sampler_ema_alpha,
                epsilon=cfg.sampler_epsilon,
                cold_start_progress=cfg.sampler_cold_start_progress,
            ),
            seed=cfg.seed,
        )

        return cls(cfg, online, target, optimizer, replay, sampler, device)

    # ------------------------------------------------------------------
    # Target sync + action selection.
    # ------------------------------------------------------------------

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.disable_noise()

    def act(
        self,
        frames: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[int, tuple[torch.Tensor, torch.Tensor]]:
        """Greedy (noisy-nets) action selection for rollout. ``frames`` is a
        single step ``(1, 1, stack, H, W)`` uint8. Hidden state is carried
        by the caller across the episode and reset at episode boundaries."""
        with torch.no_grad():
            self.online_net.reset_noise()
            q, new_hidden = self.online_net.q_values(
                frames.to(self.device), hidden=hidden, advantages_only=True
            )
        action = int(q.argmax(dim=-1).item())
        return action, new_hidden

    # ------------------------------------------------------------------
    # Learn step.
    # ------------------------------------------------------------------

    def learn_step(self) -> dict[str, float]:
        """One gradient step from a sequence sampled out of the recurrent replay.

        Returns scalar metrics for logging (loss, td abs mean, grad norm, etc.).
        No-op if replay hasn't warmed up yet.
        """
        if self.replay.capacity < self.cfg.min_sampling_size:
            return {}

        seq_len = self.cfg.burn_in_len + self.cfg.learning_seq_len
        tree_idxs, states, actions, rewards, n_states, dones, weights = (
            self.replay.sample_sequences(self.cfg.batch_size, seq_len)
        )
        # states/n_states: (B, seq_len, framestack, H, W) float32 on device
        # actions: (B, seq_len) int64, rewards/dones: (B, seq_len), weights: (B,)

        burn_in = self.cfg.burn_in_len
        burn_states = states[:, :burn_in]
        learn_states = states[:, burn_in:]
        learn_n_states = n_states[:, burn_in:]
        learn_actions = actions[:, burn_in:]
        learn_rewards = rewards[:, burn_in:]
        learn_dones = dones[:, burn_in:]

        # Burn-in forward on online + target. No grad; purpose is only to
        # warm up the LSTM hidden state so the learning-window forward starts
        # from roughly the collection-time hidden.
        with torch.no_grad():
            self.target_net.reset_noise()
            _, _, hidden_online = self.online_net(burn_states)
            _, _, hidden_target = self.target_net(burn_states)

        # Learning-window forward on online (with grad).
        self.online_net.reset_noise()
        online_quantiles_seq, online_taus_seq, _ = self.online_net(
            learn_states, hidden=hidden_online
        )
        # (B, T, num_tau, n_actions), (B, T, num_tau, 1)

        # Target net forward on n-state sequence (no grad).
        with torch.no_grad():
            target_quantiles_seq, _, _ = self.target_net(
                learn_n_states, hidden=hidden_target
            )

        # Flatten (B, T) → (B*T) so per-timestep loss math mirrors the
        # original single-step pattern in VIPTankz.
        B, T = learn_actions.shape
        online_q_flat = online_quantiles_seq.reshape(B * T, self.cfg.num_tau, -1)
        online_taus_flat = online_taus_seq.reshape(B * T, self.cfg.num_tau, 1)
        target_q_flat = target_quantiles_seq.reshape(B * T, self.cfg.num_tau, -1)
        actions_flat = learn_actions.reshape(B * T)
        rewards_flat = learn_rewards.reshape(B * T)
        dones_flat = learn_dones.reshape(B * T, 1, 1)
        # Broadcast per-sequence weights across T so each element gets the same weight.
        weights_flat = weights.unsqueeze(1).expand(-1, T).reshape(B * T)

        # Munchausen reward (needs online Q at current state, detached).
        munch_reward = _compute_munchausen_reward(
            online_q_flat.detach(),
            rewards_flat,
            actions_flat,
            self.cfg.entropy_tau,
            self.cfg.munch_alpha,
            self.cfg.munch_lo,
        )

        loss, td_abs_flat = _compute_td_error_and_loss(
            online_q_flat,
            online_taus_flat,
            target_q_flat,
            actions_flat,
            munch_reward,
            gamma_n=self.cfg.gamma**self.cfg.n_step,
            dones=dones_flat,
            weights=weights_flat,
            entropy_tau=self.cfg.entropy_tau,
        )

        # Backward + clip + step.
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.cfg.grad_clip
        ).item()
        self.optimizer.step()

        # Priority update: aggregate per-sequence via R2D2's η·max + (1-η)·mean.
        td_abs_bt = td_abs_flat.reshape(B, T).cpu().numpy()  # (B, T)
        eta = self.cfg.priority_eta
        seq_priorities = eta * td_abs_bt.max(axis=1) + (1 - eta) * td_abs_bt.mean(axis=1)
        self.replay.update_priorities(tree_idxs, seq_priorities)

        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_replace_grad_steps == 0:
            self.sync_target()

        return {
            "loss": float(loss.item()),
            "td_abs_mean": float(td_abs_bt.mean()),
            "grad_norm": float(grad_norm),
            "grad_steps": self.grad_steps,
        }


# ---------------------------------------------------------------------------
# Rollout.
# ---------------------------------------------------------------------------


def run_one_episode(
    agent: BTRAgent,
    env: MkwDolphinEnv,
    track_slug: str,
) -> tuple[float, dict[str, float], int]:
    """Run a single episode. Transitions are appended to replay; one learn
    step per env step (replay_ratio=1) fires when replay is warmed up.

    Returns (episode_return, reward_component_sums, n_steps).
    """
    obs, info = env.reset(options={"track_slug": track_slug})
    prev_obs = obs.copy()  # frame_stack at t for replay append (state_t)
    hidden = None
    episode_return = 0.0
    reward_components_sum: dict[str, float] = {}
    step = 0

    while True:
        # To tensor for action selection: (1, 1, stack, H, W).
        obs_t = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, H, W)
        action, hidden = agent.act(obs_t, hidden)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Append the transition.
        # PER.append expects: state (framestack), action, reward, n_state (framestack), done, trun, stream
        agent.replay.append(
            state=prev_obs,
            action=action,
            reward=reward,
            n_state=next_obs,
            done=bool(terminated),
            trun=bool(truncated),
            stream=0,
        )
        agent.env_steps += 1
        episode_return += reward

        rb = info.get("reward_breakdown", {})
        for k, v in rb.items():
            reward_components_sum[k] = reward_components_sum.get(k, 0.0) + float(v)

        # Learn step (once per env step at replay_ratio=1).
        for _ in range(agent.cfg.replay_ratio):
            agent.learn_step()

        prev_obs = next_obs.copy()
        obs = next_obs
        step += 1

        if done:
            break

    return episode_return, reward_components_sum, step


# ---------------------------------------------------------------------------
# Outer training loop.
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig, env: MkwDolphinEnv | None = None) -> BTRAgent:
    """Main training driver. Constructs agent, runs episodes until
    ``total_frames`` env steps, logs + checkpoints per config cadence."""
    agent = BTRAgent.build(cfg)
    run_name = time.strftime("btr_%Y%m%d_%H%M%S")
    logger = make_logger(cfg, run_name)
    log.info("starting training — run=%s device=%s", run_name, cfg.device)

    if env is None:
        env = MkwDolphinEnv(
            env_id=cfg.env_id,
            savestate_dir=cfg.savestate_dir,
            track_metadata_path=cfg.track_metadata_path,
        )

    episode_idx = 0
    try:
        while agent.env_steps < cfg.total_frames:
            track_slug = agent.sampler.sample()
            ep_return, rb_sums, n_steps = run_one_episode(agent, env, track_slug)
            agent.sampler.update(track_slug, ep_return)
            episode_idx += 1

            # Logging: per-episode summary + per-track marker + sampler dist.
            metrics: dict[str, float] = {
                "episode/return": ep_return,
                "episode/length": n_steps,
                f"track/{track_slug}/episode_return": ep_return,
                f"track/{track_slug}/episode_length": n_steps,
                "env_steps": agent.env_steps,
            }
            for comp, val in rb_sums.items():
                metrics[f"reward/{comp}"] = val
            for slug, weight in agent.sampler.distribution().items():
                metrics[f"track_sampler/{slug}/weight"] = weight
            logger.log(metrics, step=agent.env_steps)

            # Checkpoint.
            if (
                agent.grad_steps > 0
                and agent.grad_steps % cfg.checkpoint_every_grad_steps == 0
            ):
                ckpt_path = Path(cfg.log_dir) / f"{run_name}_grad{agent.grad_steps}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "online": agent.online_net.state_dict(),
                        "target": agent.target_net.state_dict(),
                        "optimizer": agent.optimizer.state_dict(),
                        "grad_steps": agent.grad_steps,
                        "env_steps": agent.env_steps,
                        "config": cfg.__dict__,
                    },
                    ckpt_path,
                )
                log.info("saved checkpoint %s", ckpt_path)
    finally:
        env.close()
        logger.close()

    return agent
