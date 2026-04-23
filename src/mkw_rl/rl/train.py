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
6. Target net has noise permanently disabled (VIPTankz resamples per
   learn step). Keeps Munchausen's `π_target = softmax(Q_target/τ)`
   deterministic within a batch.
7. Target LSTM burn-in runs on `n_states[:, :burn_in]` (not `states[...]`)
   so the target's hidden is aligned to its learning-window inputs,
   which come from the n-step-shifted state sequence.
8. Scalar `γ^n_step` bootstrap discount — over-discounts n-step windows
   clipped by `trun` (without `done`). Impact is small at n=3 for
   MKWii's ~1000-frame episodes; matches VIPTankz.

See `docs/TRAINING_METHODOLOGY.md` "Inherited implementation quirks" for
the full list (items 7-15) with paper references.
"""

from __future__ import annotations

import copy
import csv
import logging
import os
import platform
import random
import re
import signal
import threading
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
    # Number of parallel Dolphin instances. 1 keeps the legacy single-env path.
    # >1 enables the multi-env rollout: each instance runs in its own thread,
    # appends to the shared replay via stream=env_id, and uses its own Dolphin
    # binary directory (must be pre-created — see scripts/setup_dolphin_instances.py).
    # Socket ports are BASE_PORT + env_id so they don't collide.
    num_envs: int = 1
    # Optional — override MkwDolphinEnv's dev-machine defaults. Required for
    # any host where Dolphin / the ISO don't live at the author's local paths
    # (e.g., Vast.ai). Leave as None to use MkwDolphinEnv's defaults.
    # With num_envs > 1, dolphin_app must point at the PARENT directory that
    # contains dolphin0/, dolphin1/, ... (not a specific dolphin{i}/).
    dolphin_app: str | None = None
    iso: str | None = None
    mkw_rl_src: str | None = None

    # model (forwarded into BTRConfig)
    stack_size: int = 4
    input_hw: tuple[int, int] = (75, 140)
    encoder_channels: tuple[int, int, int] = (32, 64, 64)  # VIPTankz model_size=2 default
    feature_dim: int = 256
    lstm_hidden: int = 512
    lstm_layers: int = 1
    linear_size: int = 512
    num_tau: int = 8
    n_cos: int = 64
    layer_norm: bool = True
    spectral_norm: bool = True  # wraps every conv in spectral_norm (VIPTankz default)

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
    # Max number of periodic ``{run_name}_grad{N}.pt`` ckpts to retain.
    # Older ones are pruned as new ones land. ``_final.pt`` and ``_diverged.pt``
    # are never pruned (they mark run end-states). Set to 0 to disable rotation.
    keep_last_n_checkpoints: int = 5

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
        for k in ("env_id", "num_envs", "dolphin_app", "iso", "mkw_rl_src"):
            if k in raw["env"]:
                kw[k] = raw["env"][k]
    if "model" in raw:
        for k in (
            "stack_size", "feature_dim", "lstm_hidden", "lstm_layers",
            "linear_size", "num_tau", "n_cos", "layer_norm", "spectral_norm",
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
        for k in (
            "wandb_project", "log_dir", "log_every_grad_steps",
            "checkpoint_every_grad_steps", "keep_last_n_checkpoints",
        ):
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
    """RFC-4180-compatible CSV logger. One row per log() call. Columns may
    grow as new metric keys appear — when they do, the whole file is rewritten
    with the extended header and all previous rows (padded). The file is
    always readable by ``pandas.read_csv`` at any point.

    Resume-safe: if a CSV already exists at ``path``, the existing header and
    rows are parsed in and new rows append in the same file. This means a
    resumed training run (which reuses ``run_name``) writes a continuous log
    instead of creating a fresh file per resume.

    Small per-run log volume makes the rewrite cost negligible vs the
    alternative (malformed headers or silently-dropped columns)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._columns: list[str] = ["step"]
        self._rows: list[dict[str, Any]] = []

        # Resume-safe: preload any existing rows + columns. If the file is
        # empty or unreadable as CSV, start fresh.
        existing = path.exists() and path.stat().st_size > 0
        if existing:
            try:
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        self._columns = list(reader.fieldnames)
                        self._rows = list(reader)
            except (OSError, csv.Error) as e:
                log.warning("could not parse existing CSV at %s (%s); starting fresh", path, e)
                existing = False

        # Always open in append so partial-write safety is better than
        # truncate-on-startup. The rewrite path still truncates, but only when
        # a new column forces it.
        self._fh = open(path, "a", buffering=1)  # line-buffered
        self._writer = csv.DictWriter(self._fh, fieldnames=self._columns)
        if not existing:
            self._writer.writeheader()
            self._fh.flush()

    def _rewrite(self) -> None:
        """Rewrite the whole file with the current ``_columns`` + ``_rows``.
        Triggered when a new metric key appears."""
        self._fh.close()
        self._fh = open(self.path, "w", buffering=1)
        self._writer = csv.DictWriter(self._fh, fieldnames=self._columns)
        self._writer.writeheader()
        for row in self._rows:
            self._writer.writerow({k: row.get(k, "") for k in self._columns})
        self._fh.flush()

    def log(self, metrics: dict[str, float], step: int) -> None:
        row: dict[str, Any] = {"step": step, **metrics}
        new_cols = [k for k in row if k not in self._columns]
        if new_cols:
            self._columns.extend(new_cols)
            self._rows.append(row)
            self._rewrite()
            return
        self._rows.append(row)
        self._writer.writerow({k: row.get(k, "") for k in self._columns})
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()


class _WandbLogger:
    """Wandb wrapper with transient-failure tolerance + CSV fallback.

    - Uses ``resume="allow"`` + ``id=run_name`` so a resumed training run
      (same ``run_name``) stitches onto the original wandb run rather than
      starting a new one. Charts stay continuous across Vast.ai preemptions.
    - Individual ``log()`` calls swallow exceptions and count consecutive
      failures. After ``MAX_FAILURES`` in a row (network blip, wandb
      maintenance, etc.), we permanently switch to a CSV fallback so an
      8-hour training run doesn't die to a 30-second network hiccup.
    """

    MAX_FAILURES = 5

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict,
        fallback_csv_path: Path,
    ) -> None:
        import wandb  # noqa: PLC0415 — optional dep
        self._wandb = wandb
        # id must match run_name for resume stitching; wandb accepts any
        # alphanumeric-underscore string.
        self._run = wandb.init(
            project=project, name=run_name, id=run_name,
            resume="allow", config=config,
        )
        self._fail_count = 0
        self._fallback_csv_path = fallback_csv_path
        self._fallback: _CSVLogger | None = None

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._fallback is not None:
            self._fallback.log(metrics, step)
            return
        try:
            self._wandb.log(metrics, step=step)
            self._fail_count = 0
        except Exception as e:  # noqa: BLE001 - any wandb internal failure
            self._fail_count += 1
            log.warning(
                "wandb.log failed (%s); fail_count=%d/%d",
                e, self._fail_count, self.MAX_FAILURES,
            )
            if self._fail_count >= self.MAX_FAILURES:
                log.error(
                    "wandb.log failed %d× consecutively; demoting to CSV at %s",
                    self.MAX_FAILURES, self._fallback_csv_path,
                )
                self._fallback = _CSVLogger(self._fallback_csv_path)
                self._fallback.log(metrics, step)

    def close(self) -> None:
        if self._fallback is not None:
            try:
                self._fallback.close()
            except Exception:  # noqa: BLE001
                log.exception("error closing CSV fallback")
        try:
            self._run.finish()
        except Exception:  # noqa: BLE001
            log.exception("error finishing wandb run")


def make_logger(cfg: TrainConfig, run_name: str) -> Logger:
    csv_path = Path(cfg.log_dir) / f"{run_name}.csv"
    if os.environ.get("WANDB_API_KEY"):
        try:
            return _WandbLogger(cfg.wandb_project, run_name, cfg.__dict__, csv_path)
        except Exception as e:  # noqa: BLE001
            log.warning("wandb init failed (%s); falling back to CSV", e)
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
    # Consecutive non-finite loss/grad events — abort after MAX_NONFINITE to
    # avoid silently training on NaN weights. Reset by any finite step.
    # With entropy_tau=0.03, one Q-overflow can poison target-sync in ~200
    # steps; 10 is forgiving enough for transients while still bailing before
    # the Adam moments go NaN across the full net.
    nonfinite_streak: int = field(default=0)

    MAX_NONFINITE: int = 10

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
            spectral_norm=cfg.spectral_norm,
        )
        online = BTRPolicy(model_cfg).to(device)
        target = copy.deepcopy(online)
        for p in target.parameters():
            p.requires_grad = False
        target.disable_noise()  # target uses greedy (non-noisy) evaluation
        # Eval mode stops ``nn.utils.parametrizations.spectral_norm`` from
        # running power iteration + mutating ``_u``/``_v`` buffers on every
        # forward. Without this, target's spectrally-normalized weight drifts
        # (~6e-5 per forward, compounding) between ``sync_target`` calls even
        # though its underlying ``.original`` weight is frozen — which
        # contaminates the Munchausen `π_target = softmax(Q_target/τ)` at
        # `entropy_tau=0.03` where tiny Q shifts matter.
        target.eval()

        optimizer = torch.optim.Adam(
            online.parameters(),
            lr=cfg.lr,
            eps=0.005 / cfg.batch_size,
        )

        replay = PER(
            size=cfg.replay_size,
            device=device,
            n=cfg.n_step,
            envs=cfg.num_envs,
            gamma=cfg.gamma,
            alpha=cfg.per_alpha,
            beta=cfg.per_beta,
            framestack=cfg.framestack,
            imagex=cfg.imagex,
            imagey=cfg.imagey,
            storage_size_multiplier=cfg.storage_size_multiplier,
        )

        # Pass the metadata path so ``available_tracks`` intersects on-disk
        # savestates with YAML entries — any slug on disk but missing from
        # the YAML would crash ``env.reset`` with KeyError, burning 3 Dolphin
        # boots per slug before the per-track crash counter kicks in. The
        # intersection prevents that silent cost.
        tracks = available_tracks(cfg.savestate_dir, cfg.track_metadata_path)
        if not tracks:
            raise RuntimeError(
                f"no usable savestates — disk={cfg.savestate_dir!r} "
                f"intersected with track_metadata at {cfg.track_metadata_path!r} "
                "is empty. Either record savestates via scripts/record_savestates.py, "
                "or add matching entries to track_metadata.yaml."
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
        # Re-assert eval mode after load_state_dict — load_state_dict doesn't
        # touch the training flag, but belt-and-braces against any future
        # refactor that toggles it.
        self.target_net.eval()

    def act(
        self,
        frames: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        deterministic: bool = False,
    ) -> tuple[int, tuple[torch.Tensor, torch.Tensor]]:
        """Greedy (noisy-nets) action selection for rollout. ``frames`` is a
        single step ``(1, 1, stack, H, W)`` uint8. Hidden state is carried
        by the caller across the episode and reset at episode boundaries.

        ``deterministic=True``: skips ``reset_noise()`` so noisy-nets stays
        at whatever state the caller put it in (typically 0'd via
        ``disable_noise()``). Used by ``scripts/eval_btr.py`` to get
        reproducible greedy rollouts against a frozen checkpoint.
        """
        with torch.no_grad():
            if not deterministic:
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
        burn_n_states = n_states[:, :burn_in]
        learn_states = states[:, burn_in:]
        learn_n_states = n_states[:, burn_in:]
        learn_actions = actions[:, burn_in:]
        learn_rewards = rewards[:, burn_in:]
        learn_dones = dones[:, burn_in:]

        # Warm-up noise: resample before BOTH burn-in and learning forward so
        # the LSTM hidden state is computed under the same noise realization
        # that the subsequent gradient step will use. Target stays deterministic
        # (disable_noise at build + sync); do NOT reset_noise() on target here.
        self.online_net.reset_noise()

        # Burn-in forward on online + target. No grad; warms up the LSTM hidden.
        # Target burns in on n_states so its hidden corresponds to n_states[:, burn_in]
        # — this is the state the learning-window target forward actually consumes.
        with torch.no_grad():
            _, _, hidden_online = self.online_net(burn_states)
            _, _, hidden_target = self.target_net(burn_n_states)

        # Learning-window forward on online (with grad).
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

        # NaN/inf guard: entropy_tau=0.03 + outlier Q can overflow logsumexp.
        # Skip the step rather than poisoning weights with a NaN Adam moment.
        if not torch.isfinite(loss):
            self.nonfinite_streak += 1
            log.warning(
                "non-finite loss at grad_step=%d (streak=%d/%d); skipping step",
                self.grad_steps, self.nonfinite_streak, self.MAX_NONFINITE,
            )
            if self.nonfinite_streak >= self.MAX_NONFINITE:
                raise RuntimeError(
                    f"aborting: {self.MAX_NONFINITE} consecutive non-finite losses. "
                    "Training likely diverged; inspect recent metrics + replay state."
                )
            return {"loss": float("nan"), "grad_norm": float("nan"),
                    "grad_steps": self.grad_steps, "nonfinite_streak": self.nonfinite_streak}

        # Backward + clip + step.
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.cfg.grad_clip
        ).item()
        if not np.isfinite(grad_norm):
            self.nonfinite_streak += 1
            log.warning(
                "non-finite grad_norm at grad_step=%d (streak=%d/%d); skipping step",
                self.grad_steps, self.nonfinite_streak, self.MAX_NONFINITE,
            )
            if self.nonfinite_streak >= self.MAX_NONFINITE:
                raise RuntimeError(
                    f"aborting: {self.MAX_NONFINITE} consecutive non-finite grad_norms"
                )
            # Grads are NaN; zero them so the Adam moments don't get poisoned.
            self.optimizer.zero_grad()
            return {"loss": float(loss.item()), "grad_norm": float("nan"),
                    "grad_steps": self.grad_steps, "nonfinite_streak": self.nonfinite_streak}
        self.nonfinite_streak = 0
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
            # Include on the happy path (always 0 here) so the CSV/wandb
            # column exists from step 1. Without this, the column only
            # appears after the first non-finite event, breaking chart
            # continuity when trying to diagnose divergence retrospectively.
            "nonfinite_streak": self.nonfinite_streak,
        }


# ---------------------------------------------------------------------------
# Rollout.
# ---------------------------------------------------------------------------


def run_one_episode(
    agent: BTRAgent,
    env: MkwDolphinEnv,
    track_slug: str,
    logger: Logger | None = None,
    shutdown_flag: dict[str, bool] | None = None,
    stream: int = 0,
    agent_lock: "threading.Lock | None" = None,
    skip_learn: bool = False,
    deterministic: bool = False,
) -> tuple[float, dict[str, float], int]:
    """Run a single episode. Transitions are appended to replay; one learn
    step per env step (replay_ratio=1) fires when replay is warmed up.

    If ``logger`` is provided, learn-step metrics are logged every
    ``cfg.log_every_grad_steps`` grad steps (mid-episode, not just at
    episode end). ``shutdown_flag["shutdown"]`` is checked after each step
    for graceful termination.

    Multi-env plumbing:
    - ``stream`` is the replay-buffer stream ID = env_id. For single-env runs
      (legacy path) stream=0 matches the old hardcoded value.
    - ``agent_lock``, if provided, serializes all agent state mutations
      (``act``, ``learn_step``, ``replay.append``, env_steps increment). Set
      this in multi-env runs where N rollout threads share one agent.
    - ``skip_learn``: in multi-env runs the rollout threads only collect
      transitions; the main thread owns the learn-step cadence. Setting this
      to True makes the function pure rollout.
    - ``deterministic``: passed through to ``agent.act``. True = noisy-nets
      frozen (caller must have called ``disable_noise()``), used for eval.

    Returns (episode_return, reward_component_sums, n_steps).
    """
    # Convert per-track-failure exceptions from env.reset into the narrow
    # EnvResetFailed class so the outer train loop catches them alongside
    # socket errors. Keeps the main exception net focused on "env is broken
    # in a track-specific way" without swallowing generic bugs from deeper
    # in run_one_episode / learn_step.
    try:
        obs, info = env.reset(options={"track_slug": track_slug})
    except (FileNotFoundError, KeyError) as exc:
        raise EnvResetFailed(f"reset failed for {track_slug!r}: {exc}") from exc
    # Env and replay both maintain framestacks. Sanity-check the env's stack
    # shape so a silent regression where the env starts returning single
    # frames (or differently-ordered stacks) would fail loudly here.
    assert obs.shape == (agent.cfg.framestack, agent.cfg.imagey, agent.cfg.imagex), (
        f"env obs shape {obs.shape} doesn't match "
        f"(framestack={agent.cfg.framestack}, H={agent.cfg.imagey}, W={agent.cfg.imagex})"
    )
    prev_obs = obs.copy()  # frame_stack at t for replay append (state_t)
    hidden = None
    episode_return = 0.0
    reward_components_sum: dict[str, float] = {}
    step = 0
    log_cadence = agent.cfg.log_every_grad_steps

    # Dummy context manager so the main loop stays readable whether or not
    # a lock is passed (single-env: nullcontext; multi-env: real Lock).
    import contextlib  # noqa: PLC0415
    lock = agent_lock if agent_lock is not None else contextlib.nullcontext()

    while True:
        # To tensor for action selection: (1, 1, stack, H, W).
        obs_t = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)
        with lock:
            action, hidden = agent.act(obs_t, hidden, deterministic=deterministic)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        with lock:
            agent.replay.append(
                state=prev_obs,
                action=action,
                reward=reward,
                n_state=next_obs,
                done=bool(terminated),
                trun=bool(truncated),
                stream=stream,
            )
            agent.env_steps += 1
        episode_return += reward

        rb = info.get("reward_breakdown", {})
        for k, v in rb.items():
            reward_components_sum[k] = reward_components_sum.get(k, 0.0) + float(v)

        # Learn step (once per env step at replay_ratio=1). Emit learn metrics
        # at configured grad-step cadence so mid-training loss spikes are
        # visible without waiting for the episode boundary. In multi-env
        # runs we skip this — the main thread drives learning on its own
        # cadence synced to total env_steps across all envs.
        if not skip_learn:
            for _ in range(agent.cfg.replay_ratio):
                learn_metrics = agent.learn_step()
                if (
                    logger is not None
                    and learn_metrics
                    and log_cadence > 0
                    and agent.grad_steps > 0
                    and agent.grad_steps % log_cadence == 0
                ):
                    learn_log = {f"learn/{k}": v for k, v in learn_metrics.items()}
                    logger.log(learn_log, step=agent.env_steps)

        prev_obs = next_obs.copy()
        obs = next_obs
        step += 1

        if done or (shutdown_flag is not None and shutdown_flag.get("shutdown")):
            break

    return episode_return, reward_components_sum, step


# ---------------------------------------------------------------------------
# Checkpoint save + load.
# ---------------------------------------------------------------------------


def _save_checkpoint(agent: BTRAgent, cfg: TrainConfig, path: Path) -> None:
    """Serialize agent state to ``path``. Replay buffer is NOT stored —
    re-warmup on resume is ~200K steps (<0.1% of a 500M-frame run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "online": agent.online_net.state_dict(),
            "target": agent.target_net.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "grad_steps": agent.grad_steps,
            "env_steps": agent.env_steps,
            "nonfinite_streak": agent.nonfinite_streak,
            "sampler_state": agent.sampler.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )
    log.info("saved checkpoint %s (grad=%d env=%d)", path, agent.grad_steps, agent.env_steps)


def _prune_old_checkpoints(log_dir: Path, run_name: str, keep_last_n: int) -> None:
    """Delete all but the ``keep_last_n`` newest ``{run_name}_grad{N}.pt``
    periodic checkpoints. Never touches ``_final.pt`` or ``_diverged.pt``.

    A multi-day run with ``checkpoint_every_grad_steps=10_000`` and a ~75 MB
    ckpt would otherwise fill ``log_dir`` unboundedly — typical Vast.ai
    instances ship with 50-100 GB scratch. Rotation keeps disk pressure
    bounded without requiring user intervention.
    """
    if keep_last_n <= 0:
        return
    pattern = f"{run_name}_grad*.pt"
    candidates = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if len(candidates) <= keep_last_n:
        return
    to_delete = candidates[:-keep_last_n]
    for p in to_delete:
        try:
            p.unlink()
            log.info("pruned old checkpoint %s", p)
        except OSError as exc:  # noqa: BLE001 — best-effort cleanup
            log.warning("failed to prune %s: %s", p, exc)


def load_checkpoint(agent: BTRAgent, path: Path | str) -> None:
    """Restore agent state in place from a ``_save_checkpoint`` output.

    Replay is not saved/loaded — the outer train loop re-warms from scratch.
    This means the first ~200K steps post-resume are random-policy exploration
    (matching min_sampling_size) regardless of the resumed grad_step count.
    """
    ckpt = torch.load(path, map_location=agent.device)
    agent.online_net.load_state_dict(ckpt["online"])
    agent.target_net.load_state_dict(ckpt["target"])
    agent.optimizer.load_state_dict(ckpt["optimizer"])
    agent.grad_steps = int(ckpt["grad_steps"])
    agent.env_steps = int(ckpt["env_steps"])
    agent.nonfinite_streak = int(ckpt.get("nonfinite_streak", 0))
    if "sampler_state" in ckpt:
        agent.sampler.load_state_dict(ckpt["sampler_state"])
    # load_state_dict preserves the source module's training flag via its
    # values but doesn't set the destination's mode. Re-assert target.eval()
    # to keep spectral_norm power iteration + noise disabled on target.
    agent.target_net.eval()
    log.info(
        "resumed from %s: grad_steps=%d env_steps=%d",
        path, agent.grad_steps, agent.env_steps,
    )


# ---------------------------------------------------------------------------
# Outer training loop.
# ---------------------------------------------------------------------------


class EnvResetFailed(Exception):  # noqa: N818 — keeps legacy callers' match strings readable
    """Raised by ``run_one_episode`` when ``env.reset`` fails in a way that
    should be handled per-track (missing savestate / unknown slug) rather
    than as a generic env crash. The outer ``train()`` loop catches this
    alongside socket errors and triggers the per-track crash counter.

    Narrower than swallowing raw ``FileNotFoundError`` / ``KeyError``: a
    future code change that introduces a bare dict lookup inside
    ``learn_step`` won't be miscategorized as "env crash" and silently
    absorbed by the retry loop.
    """


def _make_env(cfg: TrainConfig, env_id: int | None = None) -> MkwDolphinEnv:
    """Construct a single ``MkwDolphinEnv`` from a ``TrainConfig``.

    Uses whatever optional paths the config carries (``dolphin_app``, ``iso``,
    ``mkw_rl_src``) and falls through to ``MkwDolphinEnv`` defaults for the
    ones left unset. Centralized so the initial launch and the crash-restart
    path stay in sync — a silent divergence here produces the exact
    "restart-clobbers-custom-paths" bug. Also plumbs ``log_dir`` so the env
    writes Dolphin's stdout/stderr next to our CSV/ckpt outputs.

    ``env_id`` overrides ``cfg.env_id`` when constructing one env in a
    multi-env setup. With num_envs > 1, ``cfg.dolphin_app`` should point at
    the canonical ``dolphin0/`` install; this function picks the sibling
    ``dolphin{env_id}/`` from the same parent directory for env_ids > 0.
    That keeps single-env configs usable verbatim in multi-env runs.
    """
    effective_env_id = env_id if env_id is not None else cfg.env_id
    kwargs: dict[str, Any] = {
        "env_id": effective_env_id,
        "savestate_dir": cfg.savestate_dir,
        "track_metadata_path": cfg.track_metadata_path,
        "log_dir": cfg.log_dir,
    }
    if cfg.dolphin_app is not None:
        if cfg.num_envs > 1:
            # Sibling-dir mode: ``cfg.dolphin_app`` is expected to point at
            # ``dolphin0/``; we resolve to the sibling ``dolphin{env_id}/``
            # in the same parent dir. Fail loudly if the config points
            # somewhere else — silently constructing ``.../dolphin0/dolphin1``
            # from a macOS default like ``~/code/mkw/Wii-RL/dolphin0/DolphinQt.app``
            # would look fine at build time and blow up during the first
            # reset with a hard-to-diagnose path error.
            app_path = Path(cfg.dolphin_app)
            if app_path.name != "dolphin0":
                raise ValueError(
                    f"cfg.dolphin_app must point at a directory named 'dolphin0' "
                    f"when cfg.num_envs > 1 (got name={app_path.name!r} from "
                    f"{cfg.dolphin_app!r}). Multi-env resolves sibling dirs as "
                    f"{{parent}}/dolphin{{env_id}} and needs the base name to be "
                    f"'dolphin0' to compute them correctly. Run "
                    f"scripts/setup_dolphin_instances.py --parent "
                    f"{app_path.parent.parent} --num-envs {cfg.num_envs} to "
                    f"create the sibling tree."
                )
            parent = app_path.parent
            kwargs["dolphin_app"] = str(parent / f"dolphin{effective_env_id}")
        else:
            kwargs["dolphin_app"] = cfg.dolphin_app
    if cfg.iso is not None:
        kwargs["iso"] = cfg.iso
    if cfg.mkw_rl_src is not None:
        kwargs["mkw_rl_src"] = cfg.mkw_rl_src
    return MkwDolphinEnv(**kwargs)


def _make_envs(cfg: TrainConfig) -> list[MkwDolphinEnv]:
    """Construct num_envs parallel ``MkwDolphinEnv`` instances.

    num_envs=1 returns a single-element list — callers that want the legacy
    single-env path can unwrap with ``envs[0]``. num_envs>1 returns N envs
    with env_id=0..N-1, each pointing at its own dolphin{i}/ binary dir.
    """
    if cfg.num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {cfg.num_envs}")
    return [_make_env(cfg, env_id=i) for i in range(cfg.num_envs)]


def _infer_run_name_from_ckpt(path: Path | str) -> str:
    """Derive a stable ``run_name`` from a checkpoint filename.

    Strips any of the suffixes our checkpoint layout produces:
      ``{run_name}_grad{N}.pt`` → ``{run_name}``
      ``{run_name}_final.pt`` → ``{run_name}``
      ``{run_name}_diverged.pt`` → ``{run_name}``

    This keeps log files + subsequent checkpoints under the same ``run_name``
    across preemption/resume, so wandb charts stitch and CSVs stay
    append-compatible.
    """
    stem = Path(path).stem  # drop .pt
    for suffix in ("_final", "_diverged"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    m = re.match(r"(.+)_grad\d+$", stem)
    if m:
        return m.group(1)
    return stem


def _install_shutdown_handler() -> tuple[dict[str, bool], callable]:
    """Install SIGTERM handler + return ``(flag_dict, restore_fn)``.

    The returned mutable dict ``{"shutdown": bool, "second_signal": bool}`` is
    polled by ``run_one_episode`` between env steps (first-signal path) and
    at the outer loop boundary. A second SIGTERM bypasses graceful exit — the
    default handler is reinstated so a third kill takes the process down.

    The returned ``restore_fn()`` is for ``train()`` to call in its ``finally``
    block so the SIGTERM handler doesn't leak across sequential ``train()``
    calls (tests, orchestrators, etc.).
    """
    flag: dict[str, bool] = {"shutdown": False, "second_signal": False}
    original = signal.getsignal(signal.SIGTERM)

    def handler(signum: int, frame: object) -> None:
        if flag["shutdown"]:
            log.warning("received second signal %d; restoring default handler", signum)
            signal.signal(signal.SIGTERM, original if callable(original) else signal.SIG_DFL)
            flag["second_signal"] = True
            return
        log.warning(
            "received signal %d; will shut down gracefully after current episode",
            signum,
        )
        flag["shutdown"] = True

    signal.signal(signal.SIGTERM, handler)

    def restore() -> None:
        current = signal.getsignal(signal.SIGTERM)
        if current is handler:
            signal.signal(signal.SIGTERM, original if callable(original) else signal.SIG_DFL)

    return flag, restore


# ---------------------------------------------------------------------------
# Multi-env training path.
# ---------------------------------------------------------------------------


# Minimum age (seconds) for an X11/xvfb artifact to be considered "stale".
# Anything younger than this is assumed to belong to a live xvfb-run — we
# leave it alone. 60s is well past Dolphin boot (~2s) while tight enough
# to clean up just-crashed orphans before they accumulate. Originally
# 300s; tightened after a 4-env resume run hit track_streak=3/3 within
# 20s when env-3's orphans from the aborted run were still 'live' by
# the 300s threshold, preventing the fresh X11 cleanup from removing
# them, which triggered repeated SIGSEGVs on env-3's relaunches.
_X11_STALE_AGE_S = 60.0

# How often the main training thread re-runs _cleanup_stale_x11_state.
# Every Dolphin env crash leaves one orphan at /tmp/.X11-unix/XNN and
# one /tmp/xvfb-run.XXX; at ~20% crash rate across 4 envs, orphans can
# cluster in the first few minutes. 60s sweep catches orphans within
# a couple minutes of the crash, staying ahead of the ~10-orphan SIGSEGV
# threshold observed on Vast.
_X11_CLEANUP_INTERVAL_S = 60.0


def _cleanup_stale_x11_state() -> None:
    """Wipe leftover Xvfb sockets + xvfb-run work dirs older than
    ``_X11_STALE_AGE_S`` in /tmp.

    Each Dolphin launch picks a fresh X display via ``xvfb-run -a``. When a
    run crashes (common during dev, and inevitable over a multi-hour prod
    run that relaunches envs on EOFError), the X socket under
    ``/tmp/.X11-unix/XNN`` and the xvfb-run scratch dir ``/tmp/xvfb-run.XXX``
    can persist. After enough accumulation (~10 orphans observed on Vast),
    Dolphin's Qt init interacts badly with the leftover state and SIGSEGVs
    silently during the first ``reset()`` — crashing every subsequent
    relaunch until the orphans are cleaned.

    Liveness protection: we only touch artifacts whose mtime is older than
    ``_X11_STALE_AGE_S`` seconds. Anything newer probably belongs to a live
    process (a concurrent ``train()`` invocation, a live ``eval_btr.py``
    rollout, the user's own Xvfb session on a shared dev box). Without
    this guard, running eval alongside a live trainer would nuke the
    trainer's X socket and SIGSEGV all its envs.

    Linux-only; no-op elsewhere so macOS dev machines aren't affected.
    """
    if platform.system() != "Linux":
        return
    import glob  # noqa: PLC0415
    now = time.time()
    removed = 0
    skipped_live = 0
    for pattern in ("/tmp/.X11-unix/X*", "/tmp/.X*-lock", "/tmp/xvfb-run.*"):
        for p in glob.glob(pattern):
            try:
                age = now - os.path.getmtime(p)
            except OSError:
                # File disappeared between glob and stat — nothing to do.
                continue
            if age < _X11_STALE_AGE_S:
                skipped_live += 1
                continue
            try:
                if os.path.isdir(p):
                    import shutil  # noqa: PLC0415
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.unlink(p)
                removed += 1
            except OSError:
                # Permission denied / already gone.
                pass
    if removed or skipped_live:
        log.info(
            "X11/xvfb cleanup: removed %d stale, skipped %d live (age<%.0fs)",
            removed, skipped_live, _X11_STALE_AGE_S,
        )


def _train_vector(
    cfg: TrainConfig,
    agent: "BTRAgent",
    logger: Logger,
    run_name: str,
) -> "BTRAgent":
    """Multi-env training loop.

    Spawns ``cfg.num_envs`` rollout threads, each driving one ``MkwDolphinEnv``
    on its own episode loop. The main thread drives the learn-step cadence
    (1 per total env step at ``replay_ratio=1``, matching single-env
    semantics), periodic checkpointing, and warmup-progress logging. An
    ``agent_lock`` serializes every mutation of shared agent state
    (``act``, ``learn_step``, ``replay.append``, ``env_steps``).

    Crash handling is per-thread: each rollout thread catches its own env
    errors, rebuilds its env, and retries; persistent per-track failures
    remove the slug from the shared sampler. A global crash limit fires if
    ANY thread hits MAX_ENV_CRASHES in a row, aborting the whole run.

    Signals a clean exit by setting ``shutdown_flag["shutdown"]`` — rollout
    threads check this between steps via the shared flag passed through to
    ``run_one_episode``.
    """
    envs = _make_envs(cfg)
    agent_lock = threading.Lock()
    shutdown_flag, restore_sigterm = _install_shutdown_handler()
    # Guards crash-counter dict updates from multiple rollout threads.
    crash_lock = threading.Lock()
    # Crash-counter semantics: an entry in ``track_crash_counts`` increments
    # on each env crash and only clears after ``CRASH_RESET_AFTER_SUCCESSES``
    # consecutive CLEAN episodes on that track. A purely monotonic counter
    # (old behavior) made long runs unkillable even at healthy 2% crash
    # rate; resetting on every success (interim fix) meant a 50% flaky
    # track ping-pongs forever because every alternate success wipes the
    # counter back to 0. Requiring a streak of successes strikes a middle
    # ground: truly broken tracks still hit the threshold; tracks with
    # transient flakes recover.
    track_crash_counts: dict[str, int] = {}
    track_success_streaks: dict[str, int] = {}
    per_env_crash_streaks: list[int] = [0] * cfg.num_envs
    # MAX_ENV_CRASHES is a "N disasters in a row before we give up" bound.
    # 5 was too tight for multi-hour runs — a cluster of X11-orphan-induced
    # SIGSEGVs on one env can easily chain 5 deep before the periodic X11
    # cleanup catches up. 20 gives margin for such clusters while still
    # bailing on a truly persistently-broken env. The per-env streak
    # resets on any successful episode on that env, so 20 only fires
    # when env i cannot complete a single clean rollout 20 times straight.
    MAX_ENV_CRASHES = 20
    # MAX_TRACK_CRASHES: in single-track mode, removing the track aborts
    # the whole run. We need to tolerate the crash bursts that happen in
    # the first ~2 minutes before the periodic X11 cleanup clears the
    # orphans from the previous run. 20 aligns with MAX_ENV_CRASHES: a
    # track that has 20 crashes without stringing together 3 consecutive
    # clean episodes is genuinely broken.
    MAX_TRACK_CRASHES = 20
    CRASH_RESET_AFTER_SUCCESSES = 3
    # Mutable holder so rollout threads can write back their replacement env
    # after a crash without losing main-thread visibility.
    env_slots: list[MkwDolphinEnv] = list(envs)

    episode_idx_lock = threading.Lock()
    episode_idx = [0]  # boxed for mutation from threads

    aborted_due_to_divergence = False
    aborted_with_error: BaseException | None = None

    def _rollout_worker(i: int) -> None:
        nonlocal aborted_with_error
        while not shutdown_flag["shutdown"]:
            try:
                track_slug = agent.sampler.sample()
            except Exception as exc:  # noqa: BLE001 — propagate via main thread
                shutdown_flag["shutdown"] = True
                aborted_with_error = exc
                return

            try:
                ep_return, rb_sums, n_steps = run_one_episode(
                    agent, env_slots[i], track_slug,
                    logger=logger, shutdown_flag=shutdown_flag,
                    stream=i, agent_lock=agent_lock, skip_learn=True,
                )
                with crash_lock:
                    per_env_crash_streaks[i] = 0
                    # Bump success streak; clear crash counter only after
                    # CRASH_RESET_AFTER_SUCCESSES clean episodes in a row.
                    track_success_streaks[track_slug] = (
                        track_success_streaks.get(track_slug, 0) + 1
                    )
                    if track_success_streaks[track_slug] >= CRASH_RESET_AFTER_SUCCESSES:
                        track_crash_counts.pop(track_slug, None)
            except RuntimeError as exc:
                if "consecutive non-finite" in str(exc):
                    # NaN abort from learn_step (though learn_step shouldn't
                    # fire here with skip_learn=True; defensive).
                    shutdown_flag["shutdown"] = True
                    aborted_with_error = exc
                    return
                shutdown_flag["shutdown"] = True
                aborted_with_error = exc
                return
            except (
                EOFError,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                TimeoutError,
                EnvResetFailed,
            ) as exc:
                with crash_lock:
                    per_env_crash_streaks[i] += 1
                    track_crash_counts[track_slug] = track_crash_counts.get(track_slug, 0) + 1
                    # Crash breaks the success streak — no partial credit.
                    track_success_streaks[track_slug] = 0
                    env_streak = per_env_crash_streaks[i]
                    track_streak = track_crash_counts[track_slug]
                log.error(
                    "[env %d] crashed on %s (%s: %r); env_streak=%d/%d track_streak=%d/%d — relaunching",
                    i, track_slug, type(exc).__name__, str(exc),
                    env_streak, MAX_ENV_CRASHES,
                    track_streak, MAX_TRACK_CRASHES,
                )
                if env_streak >= MAX_ENV_CRASHES:
                    shutdown_flag["shutdown"] = True
                    aborted_with_error = RuntimeError(
                        f"[env {i}] {MAX_ENV_CRASHES} consecutive crashes; aborting"
                    )
                    return
                if track_streak >= MAX_TRACK_CRASHES:
                    try:
                        with agent_lock:
                            agent.sampler.remove_track(track_slug)
                        log.warning(
                            "track %s crashed %d times across envs; removed from sampler",
                            track_slug, track_streak,
                        )
                    except KeyError:
                        # Already removed by another thread — race between duplicate crashes.
                        pass
                    except RuntimeError as rm_exc:
                        # sampler.remove_track raises RuntimeError when the last
                        # track is removed. There's nothing left to sample, so
                        # abort the whole run with a clean message.
                        shutdown_flag["shutdown"] = True
                        aborted_with_error = RuntimeError(
                            f"last track ({track_slug}) removed after {track_streak} crashes — "
                            f"no tracks remain to sample from: {rm_exc}"
                        )
                        return
                try:
                    env_slots[i].close()
                except Exception:  # noqa: BLE001
                    pass
                env_slots[i] = _make_env(cfg, env_id=i)
                continue

            with agent_lock:
                agent.sampler.update(track_slug, ep_return)
            with episode_idx_lock:
                episode_idx[0] += 1
                ep_num = episode_idx[0]

            log.info(
                "ep=%d[env=%d] track=%s return=%.2f len=%d env_steps=%d grad_steps=%d",
                ep_num, i, track_slug, ep_return, n_steps,
                agent.env_steps, agent.grad_steps,
            )

            # Per-episode structured metrics.
            replay_fill = agent.replay.capacity / max(cfg.min_sampling_size, 1)
            metrics: dict[str, float] = {
                "episode/return": ep_return,
                "episode/length": n_steps,
                f"track/{track_slug}/episode_return": ep_return,
                f"track/{track_slug}/episode_length": n_steps,
                "env_steps": agent.env_steps,
                "grad_steps": agent.grad_steps,
                "replay/capacity": agent.replay.capacity,
                "replay/fill_ratio": replay_fill,
                "env/id": i,
            }
            for comp, val in rb_sums.items():
                metrics[f"reward/{comp}"] = val
            with agent_lock:
                for slug, weight in agent.sampler.distribution().items():
                    metrics[f"track_sampler/{slug}/weight"] = weight
            # Best-effort log — if the main thread already closed the logger
            # (its finally: ran after a 30s join timed out and the thread
            # kept going), ValueError bubbles up from the CSV writer. We
            # prefer a silent skip over a PytestUnhandledThreadExceptionWarning
            # or, worse, a crash in production. The operator has all
            # metrics up to the point of logger close.
            try:
                logger.log(metrics, step=agent.env_steps)
            except (ValueError, OSError):
                return

    threads = [
        threading.Thread(target=_rollout_worker, args=(i,), name=f"rollout-{i}", daemon=True)
        for i in range(cfg.num_envs)
    ]
    for t in threads:
        t.start()
    try:
        # Main-thread loop: drive learn_step + checkpointing + warmup logging.
        last_learn_env_steps = 0
        last_warmup_log_env_steps = 0
        last_ckpt_grad_steps = -1
        last_x11_cleanup_time = time.time()
        log_cadence = cfg.log_every_grad_steps
        try:
            while agent.env_steps < cfg.total_frames and not shutdown_flag["shutdown"]:
                # Learn-step cadence: one per total env step (matches single-env
                # replay_ratio=1 semantics scaled across envs). We batch them
                # together under the lock to avoid fine-grained contention.
                with agent_lock:
                    current_env_steps = agent.env_steps
                    replay_warm = agent.replay.capacity >= cfg.min_sampling_size

                if replay_warm:
                    delta = current_env_steps - last_learn_env_steps
                    if delta > 0:
                        with agent_lock:
                            for _ in range(delta * cfg.replay_ratio):
                                # Respect shutdown mid-batch. Without this, a
                                # NaN-triggered shutdown (or user SIGTERM) can
                                # grind through hundreds more learn_step()
                                # calls on poisoned weights before the outer
                                # loop's shutdown check fires, stomping the
                                # _diverged.pt save with even worse state.
                                if shutdown_flag["shutdown"]:
                                    break
                                learn_metrics = agent.learn_step()
                                if (
                                    learn_metrics
                                    and log_cadence > 0
                                    and agent.grad_steps > 0
                                    and agent.grad_steps % log_cadence == 0
                                ):
                                    learn_log = {f"learn/{k}": v for k, v in learn_metrics.items()}
                                    logger.log(learn_log, step=agent.env_steps)
                        last_learn_env_steps = current_env_steps

                # Warmup-progress log (throttled).
                if (
                    not replay_warm
                    and current_env_steps - last_warmup_log_env_steps >= max(cfg.min_sampling_size // 20, 100)
                ):
                    log.info(
                        "warmup: replay=%d/%d (%.1f%%) env_steps=%d",
                        agent.replay.capacity, cfg.min_sampling_size,
                        100.0 * agent.replay.capacity / max(cfg.min_sampling_size, 1),
                        current_env_steps,
                    )
                    last_warmup_log_env_steps = current_env_steps

                # Checkpoint cadence.
                with agent_lock:
                    grad_steps_now = agent.grad_steps
                if (
                    grad_steps_now > 0
                    and cfg.checkpoint_every_grad_steps > 0
                    and grad_steps_now % cfg.checkpoint_every_grad_steps == 0
                    and grad_steps_now != last_ckpt_grad_steps
                ):
                    ckpt_path = Path(cfg.log_dir) / f"{run_name}_grad{grad_steps_now}.pt"
                    with agent_lock:
                        _save_checkpoint(agent, cfg, ckpt_path)
                    _prune_old_checkpoints(
                        Path(cfg.log_dir), run_name, cfg.keep_last_n_checkpoints,
                    )
                    last_ckpt_grad_steps = grad_steps_now

                # Periodic X11 cleanup. Each env crash leaves an orphan at
                # /tmp/.X11-unix/XNN + /tmp/xvfb-run.XXX; over a multi-hour
                # run with a ~20% crash rate those accumulate past the
                # ~10-orphan threshold that triggers fresh-Dolphin SIGSEGV.
                # Run every _X11_CLEANUP_INTERVAL_S; the liveness guard
                # (_X11_STALE_AGE_S) keeps us from touching live envs'
                # sockets.
                if time.time() - last_x11_cleanup_time >= _X11_CLEANUP_INTERVAL_S:
                    _cleanup_stale_x11_state()
                    last_x11_cleanup_time = time.time()

                # Avoid spinning when nothing changed.
                time.sleep(0.01)
        except KeyboardInterrupt:
            log.warning("KeyboardInterrupt — signalling rollout threads")
            shutdown_flag["shutdown"] = True

        # Propagate any error captured from a rollout thread.
        if aborted_with_error is not None:
            if "consecutive non-finite" in str(aborted_with_error):
                aborted_due_to_divergence = True
                log.error("training diverged — will save as _diverged.pt")
    finally:
        # Wait for rollout threads to wind down BEFORE touching shared
        # resources (logger.close, env.close). Without this join-first
        # ordering a rollout thread mid-logger.log() races against the
        # finally's logger.close() and crashes with "I/O on closed file".
        shutdown_flag["shutdown"] = True
        for t in threads:
            t.join(timeout=30)
            if t.is_alive():
                log.warning("%s didn't exit within 30s", t.name)
        suffix = "_diverged" if aborted_due_to_divergence else "_final"
        try:
            final_path = Path(cfg.log_dir) / f"{run_name}{suffix}.pt"
            with agent_lock:
                _save_checkpoint(agent, cfg, final_path)
        except Exception:  # noqa: BLE001
            log.exception("failed to save final checkpoint")
        for i, e in enumerate(env_slots):
            try:
                e.close()
            except Exception:  # noqa: BLE001
                log.exception("error closing env %d", i)
        try:
            logger.close()
        except Exception:  # noqa: BLE001
            log.exception("error closing logger")
        restore_sigterm()

    if aborted_with_error is not None and not aborted_due_to_divergence:
        raise aborted_with_error
    return agent


def train(
    cfg: TrainConfig,
    env: MkwDolphinEnv | None = None,
    resume_from: Path | str | None = None,
    run_name: str | None = None,
) -> BTRAgent:
    """Main training driver. Constructs agent, runs episodes until
    ``total_frames`` env steps, logs + checkpoints per config cadence.

    ``resume_from``: path to a ``_save_checkpoint`` output; if provided, the
    agent's weights/optimizer/counters are restored before training resumes.
    Replay is re-warmed from scratch (see ``load_checkpoint`` docstring).

    ``run_name``: explicit run identifier (also used for wandb run ID and
    CSV log filename). If omitted and ``resume_from`` is given, the run_name
    is inferred from the checkpoint filename so resumed runs keep writing to
    the same log/ckpt namespace. If omitted and not resuming, a timestamp
    string is generated.
    """
    random.seed(cfg.seed)

    # Wipe stale Xvfb artifacts before spawning any Dolphin — see
    # _cleanup_stale_x11_state's docstring for the full story. Safe no-op
    # on non-Linux and on a fresh box with nothing to clean.
    _cleanup_stale_x11_state()

    agent = BTRAgent.build(cfg)
    if resume_from is not None:
        load_checkpoint(agent, resume_from)

    if run_name is None:
        if resume_from is not None:
            run_name = _infer_run_name_from_ckpt(resume_from)
            log.info("inferred run_name=%s from resume path", run_name)
        else:
            run_name = time.strftime("btr_%Y%m%d_%H%M%S")

    logger = make_logger(cfg, run_name)
    log.info(
        "starting training — run=%s device=%s num_envs=%d",
        run_name, cfg.device, cfg.num_envs,
    )

    if cfg.num_envs > 1:
        if env is not None:
            raise ValueError(
                "explicit env= kwarg not supported when cfg.num_envs > 1; "
                "pass cfg only — multi-env launches happen inside train()"
            )
        return _train_vector(cfg, agent, logger, run_name)

    if env is None:
        env = _make_env(cfg)

    shutdown_flag, restore_sigterm = _install_shutdown_handler()
    episode_idx = 0
    env_crash_streak = 0
    # Crash-counter semantics: see _train_vector for the rationale. Short
    # version: monotonic killed long runs over transient flakes; reset-on-
    # any-success let 50% flaky tracks ping-pong forever; reset on a STREAK
    # of clean episodes splits the difference.
    track_crash_counts: dict[str, int] = {}
    track_success_streaks: dict[str, int] = {}
    # See _train_vector for the MAX_ENV_CRASHES=20 and MAX_TRACK_CRASHES=20
    # rationales — both relaxed from earlier tight values after
    # X11-orphan-induced crash clusters killed production runs.
    MAX_ENV_CRASHES = 20
    MAX_TRACK_CRASHES = 20
    CRASH_RESET_AFTER_SUCCESSES = 3
    last_warmup_log_env_steps = 0
    last_x11_cleanup_time = time.time()
    # Flag set when learn_step's NaN-streak abort fires. Used in `finally:` to
    # redirect the save to `_diverged.pt` instead of overwriting the last clean
    # `_final.pt` with poisoned weights (which resume would silently re-load).
    aborted_due_to_divergence = False
    try:
        while agent.env_steps < cfg.total_frames and not shutdown_flag["shutdown"]:
            # Periodic X11 cleanup — same rationale as _train_vector.
            # Single-env loop sweeps at the episode boundary since there's
            # only one rollout and it respects the 300s liveness guard.
            if time.time() - last_x11_cleanup_time >= _X11_CLEANUP_INTERVAL_S:
                _cleanup_stale_x11_state()
                last_x11_cleanup_time = time.time()
            track_slug = agent.sampler.sample()

            try:
                ep_return, rb_sums, n_steps = run_one_episode(
                    agent, env, track_slug,
                    logger=logger, shutdown_flag=shutdown_flag,
                )
                env_crash_streak = 0
                # Bump success streak; clear crash counter only after
                # CRASH_RESET_AFTER_SUCCESSES clean episodes in a row.
                track_success_streaks[track_slug] = (
                    track_success_streaks.get(track_slug, 0) + 1
                )
                if track_success_streaks[track_slug] >= CRASH_RESET_AFTER_SUCCESSES:
                    track_crash_counts.pop(track_slug, None)
            except RuntimeError as exc:
                # NaN-abort path from learn_step — propagate but flag so the
                # finally: block saves to _diverged.pt, preserving a forensic
                # trail and sparing the last clean _final.pt.
                if "consecutive non-finite" in str(exc):
                    aborted_due_to_divergence = True
                    log.error("training diverged — will save as _diverged.pt")
                raise
            except (
                EOFError,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                TimeoutError,
                # run_one_episode wraps env.reset's per-track FileNotFoundError
                # / KeyError into this narrow exception class so a future dict
                # lookup in learn_step can't be miscategorized as "env crash"
                # and silently absorbed by the retry loop.
                EnvResetFailed,
            ) as exc:
                env_crash_streak += 1
                track_crash_counts[track_slug] = track_crash_counts.get(track_slug, 0) + 1
                # Crash breaks the success streak — no partial credit.
                track_success_streaks[track_slug] = 0
                log.error(
                    "env crashed mid-episode on %s (%s: %r); env_streak=%d/%d track_streak=%d/%d — relaunching",
                    track_slug, type(exc).__name__, str(exc),
                    env_crash_streak, MAX_ENV_CRASHES,
                    track_crash_counts[track_slug], MAX_TRACK_CRASHES,
                )
                if env_crash_streak >= MAX_ENV_CRASHES:
                    raise RuntimeError(
                        f"aborting: {MAX_ENV_CRASHES} consecutive env crashes. "
                        "Dolphin is likely unrecoverable; check its log + savestate integrity."
                    ) from exc
                # Per-track: if a specific savestate keeps crashing Dolphin,
                # remove it from the curriculum rather than looping forever.
                if track_crash_counts[track_slug] >= MAX_TRACK_CRASHES:
                    try:
                        agent.sampler.remove_track(track_slug)
                        log.warning(
                            "track %s crashed %d times; removed from sampler. "
                            "Inspect data/savestates/%s.sav for corruption.",
                            track_slug, track_crash_counts[track_slug], track_slug,
                        )
                    except KeyError:
                        pass  # already removed — race between duplicate crashes
                try:
                    env.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
                # Fresh env from the SAME cfg — custom dolphin_app/iso paths survive.
                env = _make_env(cfg)
                continue

            agent.sampler.update(track_slug, ep_return)
            episode_idx += 1

            # Per-episode stdout summary so tmux tail shows progress without
            # tail -f'ing the CSV. Emitted regardless of whether the structured
            # `logger` is wandb or CSV — useful for a live eyeball check.
            log.info(
                "ep=%d track=%s return=%.2f len=%d env_steps=%d grad_steps=%d",
                episode_idx, track_slug, ep_return, n_steps,
                agent.env_steps, agent.grad_steps,
            )

            replay_fill = agent.replay.capacity / max(cfg.min_sampling_size, 1)
            if (
                replay_fill < 1.0
                and agent.env_steps - last_warmup_log_env_steps >= max(cfg.min_sampling_size // 20, 100)
            ):
                log.info(
                    "warmup: replay=%d/%d (%.1f%%) env_steps=%d",
                    agent.replay.capacity, cfg.min_sampling_size,
                    100.0 * replay_fill, agent.env_steps,
                )
                last_warmup_log_env_steps = agent.env_steps

            metrics: dict[str, float] = {
                "episode/return": ep_return,
                "episode/length": n_steps,
                f"track/{track_slug}/episode_return": ep_return,
                f"track/{track_slug}/episode_length": n_steps,
                "env_steps": agent.env_steps,
                "grad_steps": agent.grad_steps,
                "replay/capacity": agent.replay.capacity,
                "replay/fill_ratio": replay_fill,
            }
            for comp, val in rb_sums.items():
                metrics[f"reward/{comp}"] = val
            for slug, weight in agent.sampler.distribution().items():
                metrics[f"track_sampler/{slug}/weight"] = weight
            logger.log(metrics, step=agent.env_steps)

            if (
                agent.grad_steps > 0
                and cfg.checkpoint_every_grad_steps > 0
                and agent.grad_steps % cfg.checkpoint_every_grad_steps == 0
            ):
                ckpt_path = Path(cfg.log_dir) / f"{run_name}_grad{agent.grad_steps}.pt"
                _save_checkpoint(agent, cfg, ckpt_path)
                _prune_old_checkpoints(
                    Path(cfg.log_dir), run_name, cfg.keep_last_n_checkpoints,
                )
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt — saving final checkpoint before exit")
        shutdown_flag["shutdown"] = True
    finally:
        # On NaN-abort, save to _diverged.pt instead of _final.pt — otherwise a
        # `--resume` from _final.pt silently re-loads the poisoned weights.
        # Ckpt layout: {run_name}_grad{N}.pt (periodic), {run_name}_final.pt
        # (clean exit), {run_name}_diverged.pt (NaN abort).
        suffix = "_diverged" if aborted_due_to_divergence else "_final"
        try:
            final_path = Path(cfg.log_dir) / f"{run_name}{suffix}.pt"
            _save_checkpoint(agent, cfg, final_path)
        except Exception:  # noqa: BLE001 — don't mask the original exception
            log.exception("failed to save final checkpoint")
        try:
            env.close()
        except Exception:  # noqa: BLE001
            log.exception("error closing env")
        try:
            logger.close()
        except Exception:  # noqa: BLE001
            log.exception("error closing logger")
        restore_sigterm()

    return agent
