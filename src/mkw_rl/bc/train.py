"""BC training loop with truncated BPTT.

Core contract (see MKW_RL_SPEC.md §2.3):

1. ``detach()`` hidden state after every backward call — the compute graph
   must NOT grow across TBPTT windows.
2. Reset hidden state to zeros at demo boundaries (not batch boundaries).
   Use ``compute_is_continuation`` to decide.
3. Epoch-level demo shuffling only; within-demo chunk order preserved.
4. Grad clipping at configurable max-norm (default 1.0).
5. Three diagnostic aggregates (reported per epoch):
   - per-bin steering CE (ensures training isn't collapsing to bin 10)
   - A-button training F1 (learnability floor)
   - LSTM gradient norm

The training loop is deliberately a function rather than a Trainer class —
one fewer layer of indirection, easier to step through in a debugger.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mkw_rl.bc.model import BCPolicy, BCPolicyConfig, LstmState, bc_loss
from mkw_rl.dtm.action_encoding import N_STEERING_BINS
from mkw_rl.dtm.dataset import (
    DemoAwareBatchSampler,
    MkwBCDataset,
    bc_collate_fn,
    compute_is_continuation,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + results.
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All tunable training knobs. Mirrors configs/bc.yaml."""

    # Data.
    demo_glob: str = "data/processed/user_demos/*.pkl"
    train_val_split: float = 0.9
    batch_size: int = 16
    num_workers: int = 0  # 0 is safer for initial bring-up; increase after profiling

    # Model shape (passed through to BCPolicyConfig).
    stack_size: int = 4
    seq_len: int = 32
    frame_skip: int = 4

    # Optim.
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    grad_clip: float = 1.0

    # Loss weights.
    steering_weight: float = 1.0
    button_weight: float = 1.0

    # Logging + checkpoints.
    wandb_project: str = "mkw-rl"
    log_dir: Path = field(default_factory=lambda: Path("runs/bc"))
    checkpoint_every: int = 5

    # Device.
    device: str = "cpu"  # overridden per-machine

    # Reproducibility.
    seed: int = 0


@dataclass
class EpochStats:
    """Per-epoch training diagnostics. Returned from ``train_epoch``."""

    n_batches: int
    loss_total: float
    loss_steering: float
    loss_buttons: float
    per_bin_steering_loss: np.ndarray  # shape (N_STEERING_BINS,)
    per_button_f1: dict[str, float]
    lstm_grad_norm: float


# ---------------------------------------------------------------------------
# Data loading from pickled payloads.
# ---------------------------------------------------------------------------


def load_pickled_samples(paths: Iterable[Path]) -> dict[str, list]:
    """Merge pickled parse_demo.py payloads into one samples_by_demo dict."""
    merged: dict[str, list] = {}
    for p in paths:
        with Path(p).open("rb") as f:
            payload = pickle.load(f)
        sbd = payload["samples_by_demo"]
        for k, v in sbd.items():
            if k in merged:
                log.warning("duplicate demo_id %s (from %s); keeping the first", k, p)
                continue
            merged[k] = v
    if not merged:
        raise ValueError(f"no demos loaded from paths: {list(paths)!r}")
    return merged


def split_train_val(
    samples_by_demo: dict[str, list],
    train_frac: float = 0.9,
    seed: int = 0,
) -> tuple[dict[str, list], dict[str, list]]:
    """Split demos (not timesteps) into train/val sets.

    We split at the demo level so the LSTM can see full demos during eval.
    """
    rng = np.random.default_rng(seed)
    ids = list(samples_by_demo.keys())
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * train_frac))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    train = {k: samples_by_demo[k] for k in train_ids}
    val = {k: samples_by_demo[k] for k in val_ids}
    return train, val


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------


def _maybe_detach(hidden: LstmState) -> LstmState:
    return hidden[0].detach(), hidden[1].detach()


def _hidden_zero_like(model: BCPolicy, batch_size: int, device: torch.device) -> LstmState:
    return model.initial_hidden(batch_size, device=device)


def _reset_at_boundaries(
    hidden: LstmState,
    is_continuation: list[bool],
    model: BCPolicy,
    device: torch.device,
) -> LstmState:
    """Zero out hidden state for batch positions that are NOT continuations."""
    h, c = hidden
    if not is_continuation:
        return h, c
    mask = torch.tensor(
        [1.0 if cont else 0.0 for cont in is_continuation],
        device=device,
        dtype=h.dtype,
    ).view(1, -1, 1)
    h = h * mask
    c = c * mask
    return h, c


def _truncate_or_rezero(
    model: BCPolicy,
    hidden: LstmState,
    new_batch_size: int,
    device: torch.device,
) -> LstmState:
    """Adjust hidden state to a new batch size.

    If the new batch size is smaller, slice the existing hidden state
    (preserves continuation for positions 0..new_batch_size-1). If the
    new batch size is larger, zero-pad the tail (no prior state for new
    positions). Both cases are rare in normal operation; the sampler
    keeps batches full-width.
    """
    h, c = hidden
    cur = h.shape[1]
    if cur == new_batch_size:
        return h, c
    if cur > new_batch_size:
        return h[:, :new_batch_size, :].contiguous(), c[:, :new_batch_size, :].contiguous()
    # cur < new_batch_size: zero-pad the tail.
    extra = new_batch_size - cur
    shape = (model.config.lstm_layers, extra, model.config.lstm_hidden)
    pad_h = torch.zeros(shape, device=device, dtype=h.dtype)
    pad_c = torch.zeros(shape, device=device, dtype=c.dtype)
    return torch.cat([h, pad_h], dim=1), torch.cat([c, pad_c], dim=1)


@dataclass
class ValStats:
    """Per-epoch validation diagnostics."""

    n_batches: int
    loss_total: float
    loss_steering: float
    loss_buttons: float


def val_epoch(
    model: BCPolicy,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> ValStats:
    """One pass over the validation loader with carried hidden state.

    Mirrors train_epoch's is_continuation / demo-boundary reset logic
    but runs under no_grad and does not compute gradients or step the
    optimizer. Used to pick the best-val-loss checkpoint.
    """
    model.eval()

    n_batches = 0
    sum_total = 0.0
    sum_steering = 0.0
    sum_buttons = 0.0

    prev_meta_by_pos: list[dict[str, Any] | None] = [None] * config.batch_size
    hidden: LstmState | None = None

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in batch["actions"].items()}
            meta = batch["meta"]
            b_actual = frames.shape[0]

            is_cont = []
            curr_meta_by_pos: list[dict[str, Any]] = []
            for p in range(b_actual):
                m = {"demo_id": meta["demo_id"][p], "seq_start": meta["seq_start"][p]}
                is_cont.append(compute_is_continuation(prev_meta_by_pos[p], m, config.seq_len))
                curr_meta_by_pos.append(m)

            if hidden is None:
                hidden = _hidden_zero_like(model, b_actual, device)
            else:
                if hidden[0].shape[1] != b_actual:
                    # Batch size changed; truncate or re-zero as appropriate.
                    hidden = _truncate_or_rezero(model, hidden, b_actual, device)
                hidden = _reset_at_boundaries(hidden, is_cont, model, device)

            logits, new_hidden = model(frames, hidden)
            losses = bc_loss(
                logits,
                targets,
                steering_weight=config.steering_weight,
                button_weight=config.button_weight,
            )

            hidden = _maybe_detach(new_hidden)
            prev_meta_by_pos[:b_actual] = curr_meta_by_pos

            n_batches += 1
            sum_total += float(losses["total"].item())
            sum_steering += float(losses["steering"].item())
            sum_buttons += float(losses["buttons"].item())

    return ValStats(
        n_batches=n_batches,
        loss_total=sum_total / max(n_batches, 1),
        loss_steering=sum_steering / max(n_batches, 1),
        loss_buttons=sum_buttons / max(n_batches, 1),
    )


def train_epoch(
    model: BCPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: TrainConfig,
    device: torch.device,
) -> EpochStats:
    """One epoch of TBPTT training. Returns diagnostic stats."""
    model.train()

    # Running totals.
    n_batches = 0
    sum_total = 0.0
    sum_steering = 0.0
    sum_buttons = 0.0
    per_bin_sum = np.zeros(N_STEERING_BINS, dtype=np.float64)
    per_bin_count = np.zeros(N_STEERING_BINS, dtype=np.int64)
    # For F1: accumulate TP/FP/FN per button.
    button_stats = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("accelerate", "brake", "drift", "item")}
    lstm_grad_norms: list[float] = []

    # Per-batch-position state tracking for is_continuation.
    prev_meta_by_pos: list[dict[str, Any] | None] = [None] * config.batch_size
    hidden: LstmState | None = None

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["actions"].items()}
        meta = batch["meta"]
        B_actual = frames.shape[0]

        # Compute is_continuation per batch position.
        is_cont = []
        curr_meta_by_pos: list[dict[str, Any]] = []
        for p in range(B_actual):
            m = {"demo_id": meta["demo_id"][p], "seq_start": meta["seq_start"][p]}
            is_cont.append(compute_is_continuation(prev_meta_by_pos[p], m, config.seq_len))
            curr_meta_by_pos.append(m)

        # Initialize hidden on first batch; otherwise reset non-continuations.
        if hidden is None:
            hidden = _hidden_zero_like(model, B_actual, device)
        else:
            # If batch_size changed, slice or pad (H-5 audit fix) rather than
            # zeroing everything — preserves continuation for surviving streams.
            if hidden[0].shape[1] != B_actual:
                hidden = _truncate_or_rezero(model, hidden, B_actual, device)
            hidden = _reset_at_boundaries(hidden, is_cont, model, device)

        logits, new_hidden = model(frames, hidden)
        losses = bc_loss(
            logits,
            targets,
            steering_weight=config.steering_weight,
            button_weight=config.button_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()

        # Grab LSTM grad norm before clipping.
        # True concatenated-L2 norm: sqrt(sum(||g||^2)). Per the audit M-1,
        # summing per-tensor norms over-reports by the triangle inequality.
        sqsum = 0.0
        for p in model.lstm.parameters():
            if p.grad is not None:
                sqsum += float(p.grad.pow(2).sum().item())
        lstm_grad_norm = sqsum**0.5
        lstm_grad_norms.append(lstm_grad_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Detach hidden for next iteration.
        hidden = _maybe_detach(new_hidden)
        prev_meta_by_pos[:B_actual] = curr_meta_by_pos

        # Accumulate stats.
        n_batches += 1
        sum_total += float(losses["total"].item())
        sum_steering += float(losses["steering"].item())
        sum_buttons += float(losses["buttons"].item())

        # Per-bin steering CE: compute per-sample CE and bucket by true bin.
        with torch.no_grad():
            steer_logits = logits["steering"]
            steer_tgt = targets["steering_bin"].long()
            # Per-sample CE (no reduction).
            bt = steer_logits.shape[0] * steer_logits.shape[1]
            flat_logits = steer_logits.reshape(bt, N_STEERING_BINS)
            flat_tgt = steer_tgt.reshape(bt)
            log_prob = torch.nn.functional.log_softmax(flat_logits, dim=-1)
            per_sample_ce = -log_prob.gather(1, flat_tgt.unsqueeze(1)).squeeze(1).cpu().numpy()
            tgt_cpu = flat_tgt.cpu().numpy()
            for bin_idx in range(N_STEERING_BINS):
                mask = tgt_cpu == bin_idx
                if mask.any():
                    per_bin_sum[bin_idx] += per_sample_ce[mask].sum()
                    per_bin_count[bin_idx] += int(mask.sum())

            # Button F1 stats.
            for name in button_stats:
                pred = (logits[name] > 0).long().cpu().numpy()
                tgt = targets[name].long().cpu().numpy()
                tp = int(((pred == 1) & (tgt == 1)).sum())
                fp = int(((pred == 1) & (tgt == 0)).sum())
                fn = int(((pred == 0) & (tgt == 1)).sum())
                button_stats[name]["tp"] += tp
                button_stats[name]["fp"] += fp
                button_stats[name]["fn"] += fn

    if scheduler is not None:
        scheduler.step()

    # Finalize per-bin loss (mean over seen samples, inf for unseen bins).
    per_bin_loss = np.where(
        per_bin_count > 0,
        per_bin_sum / np.maximum(per_bin_count, 1),
        np.nan,
    )

    # Per-button F1.
    f1s: dict[str, float] = {}
    for name, s in button_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        denom = 2 * tp + fp + fn
        f1s[name] = (2 * tp / denom) if denom > 0 else 0.0

    avg_lstm_grad = float(np.mean(lstm_grad_norms)) if lstm_grad_norms else 0.0

    return EpochStats(
        n_batches=n_batches,
        loss_total=sum_total / max(n_batches, 1),
        loss_steering=sum_steering / max(n_batches, 1),
        loss_buttons=sum_buttons / max(n_batches, 1),
        per_bin_steering_loss=per_bin_loss,
        per_button_f1=f1s,
        lstm_grad_norm=avg_lstm_grad,
    )


# ---------------------------------------------------------------------------
# Top-level training entry (used by scripts/train_bc.py).
# ---------------------------------------------------------------------------


def make_dataset_and_loader(
    samples_by_demo: dict[str, list],
    config: TrainConfig,
    shuffle: bool,
) -> tuple[MkwBCDataset, DataLoader]:
    ds = MkwBCDataset(
        samples_by_demo,
        stack_size=config.stack_size,
        frame_skip=config.frame_skip,
        seq_len=config.seq_len,
    )
    sampler = DemoAwareBatchSampler(ds, batch_size=config.batch_size, shuffle=shuffle, seed=config.seed)
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=bc_collate_fn,
        num_workers=config.num_workers,
        pin_memory=False,  # gated on CUDA availability in the scripts/ wrapper
    )
    return ds, loader


def build_model_and_optim(
    config: TrainConfig,
    device: torch.device,
) -> tuple[BCPolicy, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    model_cfg = BCPolicyConfig(stack_size=config.stack_size)
    model = BCPolicy(model_cfg).to(device)
    optim = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=max(1, config.epochs))
    return model, optim, sched
