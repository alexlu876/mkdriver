"""BTR policy: IMPALA CNN + stateful LSTM + IQN dueling heads with noisy linears.

Architecture (v2 methodology; see ``docs/TRAINING_METHODOLOGY.md`` §2):

    Input    (B, T, stack=4, H=75, W=140), uint8 or float in [0, 255]

    Encoder  per-timestep IMPALA CNN (reused verbatim from bc.model.ImpalaEncoder,
             keeping BC↔BTR weight compat for future warm-start)
               → (B, T, feature_dim=256)

    LSTM     stateful, hidden_dim=512, 1 layer, batch_first
               → (B, T, lstm_hidden=512)

    IQN head per-(timestep × quantile sample):
               τ ~ U(0, 1), shape (B*T, num_tau, 1)
               cos(τ · π · {0..n_cos-1}) → Linear(n_cos → lstm_hidden) → ReLU
               element-wise multiply with LSTM features
               → Dueling(value, advantage) with FactorizedNoisyLinear
               → (B, T, num_tau, n_actions) quantiles

The forward returns (quantiles, taus, new_hidden). For greedy action
selection use ``q_values(...)`` which returns the mean over quantiles.

v1 vs v2 comparison: VIPTankz's published ``ImpalaCNNLargeIQN`` has the
cos embedding multiply directly against conv features (shape 2304 by
default), bypassing any LSTM. v2 inserts the LSTM between and has the
cos embedding multiply against LSTM features (shape 512). The algorithm
is otherwise unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mkw_rl.bc.model import ImpalaEncoder
from mkw_rl.rl.networks import Dueling, FactorizedNoisyLinear

type LstmState = tuple[torch.Tensor, torch.Tensor]  # (h, c)


@dataclass(frozen=True)
class BTRConfig:
    """Architecture + IQN hyperparameters. Training hyperparameters (lr, eps
    schedule, replay capacity) live separately in configs/btr.yaml / the
    training loop, not here.
    """

    n_actions: int = 40  # matches VIPTankz's Discrete(40) action space
    stack_size: int = 4
    input_hw: tuple[int, int] = (75, 140)
    encoder_channels: tuple[int, int, int] = (16, 32, 32)
    feature_dim: int = 256  # encoder output / LSTM input
    lstm_hidden: int = 512  # LSTM output / IQN dueling input
    lstm_layers: int = 1
    linear_size: int = 512  # dueling branch intermediate
    num_tau: int = 8  # quantile samples per forward (IQN)
    n_cos: int = 64  # cosine embedding dim for IQN τ
    layer_norm: bool = True  # LayerNorm inside dueling branches (VIPTankz default True on CUDA)


class _DuelingBranch(nn.Module):
    """One branch (value or advantage) of the IQN dueling head.

    Structure: NoisyLinear → [LayerNorm] → ReLU → NoisyLinear. Matches
    VIPTankz's ``linear_layersV`` / ``linear_layersA`` composition
    (BTR.py:229-240) minus the ``Sequential``-wrapped named sub-modules
    since we don't need them for checkpoint inspection.
    """

    def __init__(
        self, in_dim: int, intermediate: int, out_dim: int, use_layer_norm: bool
    ) -> None:
        super().__init__()
        self.fc1 = FactorizedNoisyLinear(in_dim, intermediate)
        self.norm: nn.Module = nn.LayerNorm(intermediate) if use_layer_norm else nn.Identity()
        self.fc2 = FactorizedNoisyLinear(intermediate, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        return self.fc2(x)


class BTRPolicy(nn.Module):
    """Full BTR policy. See module docstring for architecture overview.

    Forward signature:
        quantiles, taus, new_hidden = model(frames, hidden=None, advantages_only=False)

    Args:
        frames: (B, T, stack, H, W), uint8 or float in [0, 255] — both are
            accepted; we normalize internally to match VIPTankz's convention
            (BTR.py:262).
        hidden: optional (h, c) each shaped (lstm_layers, B, lstm_hidden).
            ``None`` yields zero init.
        advantages_only: short-circuit the Dueling branch — return raw
            advantages without the value term, for argmax-based action
            selection.
    """

    def __init__(self, config: BTRConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or BTRConfig()

        self.encoder = ImpalaEncoder(
            in_channels=self.cfg.stack_size,
            channels=self.cfg.encoder_channels,
            feature_dim=self.cfg.feature_dim,
            input_hw=self.cfg.input_hw,
        )

        self.lstm = nn.LSTM(
            input_size=self.cfg.feature_dim,
            hidden_size=self.cfg.lstm_hidden,
            num_layers=self.cfg.lstm_layers,
            batch_first=True,
        )

        # IQN: cosine embedding of τ (BTR.py:199-202, 270-272).
        self.register_buffer(
            "_cos_pis",
            torch.tensor(
                [math.pi * i for i in range(self.cfg.n_cos)], dtype=torch.float32
            ).view(1, 1, self.cfg.n_cos),
        )
        self.cos_embedding = nn.Linear(self.cfg.n_cos, self.cfg.lstm_hidden)

        # Dueling heads.
        self._value_branch = _DuelingBranch(
            self.cfg.lstm_hidden, self.cfg.linear_size, 1, self.cfg.layer_norm
        )
        self._advantage_branch = _DuelingBranch(
            self.cfg.lstm_hidden,
            self.cfg.linear_size,
            self.cfg.n_actions,
            self.cfg.layer_norm,
        )
        self.dueling = Dueling(self._value_branch, self._advantage_branch)

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: torch.Tensor,
        hidden: LstmState | None = None,
        advantages_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, LstmState]:
        if frames.ndim != 5:
            raise ValueError(
                f"expected frames shape (B, T, stack, H, W), got {tuple(frames.shape)}"
            )
        B, T, C, H, W = frames.shape
        if C != self.cfg.stack_size:
            raise ValueError(
                f"stack size mismatch: got {C}, config expects {self.cfg.stack_size}"
            )

        # Normalize to [0, 1] and collapse time for the encoder.
        x = frames.float() / 255.0
        x = x.view(B * T, C, H, W)
        features = self.encoder(x)  # (B*T, feature_dim)
        features = features.view(B, T, -1)

        # Stateful LSTM.
        if hidden is None:
            hidden = self.initial_hidden(batch_size=B, device=frames.device)
        lstm_out, new_hidden = self.lstm(features, hidden)
        # lstm_out: (B, T, lstm_hidden)

        # IQN — sample taus and compute per-quantile Q-values.
        num_tau = self.cfg.num_tau
        lstm_flat = lstm_out.reshape(B * T, -1)  # (B*T, lstm_hidden)

        taus = torch.rand(B * T, num_tau, 1, device=frames.device)
        cos = torch.cos(taus * self._cos_pis)  # (B*T, num_tau, n_cos)
        cos_flat = cos.view(B * T * num_tau, self.cfg.n_cos)
        cos_emb = F.relu(self.cos_embedding(cos_flat)).view(B * T, num_tau, -1)
        # (B*T, num_tau, lstm_hidden)

        # Element-wise multiply features by cos embedding — the IQN mixing step.
        mixed = (lstm_flat.unsqueeze(1) * cos_emb).view(B * T * num_tau, -1)
        # (B*T*num_tau, lstm_hidden)

        # Dueling head (or just advantages if caller asks).
        q_quantiles = self.dueling(mixed, advantages_only=advantages_only)
        # (B*T*num_tau, n_actions)

        q_quantiles = q_quantiles.view(B, T, num_tau, self.cfg.n_actions)
        taus = taus.view(B, T, num_tau, 1)

        return q_quantiles, taus, new_hidden

    def q_values(
        self,
        frames: torch.Tensor,
        hidden: LstmState | None = None,
        advantages_only: bool = False,
    ) -> tuple[torch.Tensor, LstmState]:
        """Return mean-quantile Q-values: (B, T, n_actions). Used for action selection."""
        quantiles, _taus, new_hidden = self.forward(
            frames, hidden=hidden, advantages_only=advantages_only
        )
        return quantiles.mean(dim=2), new_hidden

    # ------------------------------------------------------------------
    # Hidden state + noise management.
    # ------------------------------------------------------------------

    def initial_hidden(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> LstmState:
        device = device if device is not None else next(self.parameters()).device
        shape = (self.cfg.lstm_layers, batch_size, self.cfg.lstm_hidden)
        return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))

    @torch.no_grad()
    def reset_noise(self) -> None:
        """Re-sample ε for every FactorizedNoisyLinear. Call before each rollout step
        and each gradient step per VIPTankz's Agent.learn (BTR.py:936, 990)."""
        for module in self.modules():
            if isinstance(module, FactorizedNoisyLinear):
                module.reset_noise()

    @torch.no_grad()
    def disable_noise(self) -> None:
        """Zero ε everywhere so forward passes become deterministic. Used for
        evaluation rollouts (VIPTankz.Agent.prep_evaluation, BTR.py:916)."""
        for module in self.modules():
            if isinstance(module, FactorizedNoisyLinear):
                module.disable_noise()

    # ------------------------------------------------------------------
    # Checkpoint helpers.
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
