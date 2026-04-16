"""BC policy: IMPALA CNN encoder + stateful LSTM + mixed action heads.

Architecture (see MKW_RL_SPEC.md §2.2):

Input:   (B, T, stack=4, H=114, W=140), grayscale, float in [0, 1]

Encoder: IMPALA-style CNN applied per-timestep
    Block 1: Conv(in=4, out=16) → MaxPool(3x3, s=2) → 2 residual blocks
    Block 2: Conv(out=32)        → MaxPool(3x3, s=2) → 2 residual blocks
    Block 3: Conv(out=32)        → MaxPool(3x3, s=2) → 2 residual blocks
    Flatten → Linear → ReLU → 256-dim per-timestep feature

Temporal: LSTM(hidden=512, layers=1), stateful across TBPTT windows.

Heads (per timestep, consuming LSTM output):
    steering_head:   Linear(512 → N_STEERING_BINS=21)
    accelerate/brake/drift/item: Linear(512 → 1) each

forward(frames, hidden) -> (logits_dict, new_hidden)

Why IMPALA over ResNet-18: IMPALA is ~1M params, standard for pixel-based
RL, and trains faster. ResNet-18 is 11M params and its ImageNet init is
half-destroyed by the 4-channel first-conv reinit anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mkw_rl.dtm.action_encoding import N_STEERING_BINS

type LstmState = tuple[torch.Tensor, torch.Tensor]  # (h, c)


# ---------------------------------------------------------------------------
# IMPALA residual block.
# ---------------------------------------------------------------------------


class _ImpalaResBlock(nn.Module):
    """Two 3x3 convs with ReLU pre-activations and a residual add."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(x)
        y = self.conv1(y)
        y = F.relu(y)
        y = self.conv2(y)
        return x + y


class _ImpalaBlock(nn.Module):
    """IMPALA block: 3x3 conv → maxpool(3x3, s=2) → 2× residual blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ImpalaResBlock(out_channels)
        self.res2 = _ImpalaResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


# ---------------------------------------------------------------------------
# Encoder.
# ---------------------------------------------------------------------------


class ImpalaEncoder(nn.Module):
    """IMPALA CNN → 256-dim feature vector per frame."""

    def __init__(
        self,
        in_channels: int = 4,
        channels: tuple[int, int, int] = (16, 32, 32),
        feature_dim: int = 256,
        input_hw: tuple[int, int] = (114, 140),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        c1, c2, c3 = channels
        self.block1 = _ImpalaBlock(in_channels, c1)
        self.block2 = _ImpalaBlock(c1, c2)
        self.block3 = _ImpalaBlock(c2, c3)

        # Probe flatten dim with a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            flat_dim = self._forward_conv(dummy).flatten(1).shape[1]
        self.linear = nn.Linear(flat_dim, feature_dim)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return F.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) → (N, feature_dim)."""
        x = self._forward_conv(x)
        x = x.flatten(1)
        x = self.linear(x)
        x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# Full policy.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BCPolicyConfig:
    stack_size: int = 4
    input_hw: tuple[int, int] = (114, 140)
    encoder_channels: tuple[int, int, int] = (16, 32, 32)
    feature_dim: int = 256
    lstm_hidden: int = 512
    lstm_layers: int = 1
    n_steering_bins: int = N_STEERING_BINS


class BCPolicy(nn.Module):
    """Full BC policy.

    Forward signature:

        logits, new_hidden = model(frames, hidden)

    Inputs:
        frames: (B, T, stack, H, W), float32 in [0, 1]. ``stack`` must
            equal ``cfg.stack_size``.
        hidden: (h, c) each shaped (lstm_layers, B, lstm_hidden). Pass
            None for zero init.

    Outputs:
        logits: dict
            "steering":   (B, T, n_steering_bins)
            "accelerate": (B, T)  # scalar logit per timestep
            "brake":      (B, T)
            "drift":      (B, T)
            "item":       (B, T)
        new_hidden: (h, c) same shape as input hidden; detach before
            next forward if you're doing TBPTT.
    """

    _BUTTON_NAMES = ("accelerate", "brake", "drift", "item")

    def __init__(self, config: BCPolicyConfig | None = None) -> None:
        super().__init__()
        cfg = config or BCPolicyConfig()
        self.config = cfg

        self.encoder = ImpalaEncoder(
            in_channels=cfg.stack_size,
            channels=cfg.encoder_channels,
            feature_dim=cfg.feature_dim,
            input_hw=cfg.input_hw,
        )
        self.lstm = nn.LSTM(
            input_size=cfg.feature_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )
        self.steering_head = nn.Linear(cfg.lstm_hidden, cfg.n_steering_bins)
        self.button_heads = nn.ModuleDict(
            {name: nn.Linear(cfg.lstm_hidden, 1) for name in self._BUTTON_NAMES}
        )

    def initial_hidden(self, batch_size: int, device: torch.device | str | None = None) -> LstmState:
        """Return a zero (h, c) pair for the given batch size."""
        dev = device or next(self.parameters()).device
        shape = (self.config.lstm_layers, batch_size, self.config.lstm_hidden)
        h = torch.zeros(shape, device=dev)
        c = torch.zeros(shape, device=dev)
        return h, c

    def forward(
        self,
        frames: torch.Tensor,
        hidden: LstmState | None = None,
    ) -> tuple[dict[str, torch.Tensor], LstmState]:
        if frames.ndim != 5:
            raise ValueError(f"frames must be (B, T, stack, H, W); got {tuple(frames.shape)}")
        B, T, stack, H, W = frames.shape
        if stack != self.config.stack_size:
            raise ValueError(f"stack dim {stack} != config.stack_size {self.config.stack_size}")

        # Fold time into batch for the CNN.
        x = frames.reshape(B * T, stack, H, W)
        features = self.encoder(x)  # (B*T, feature_dim)
        features = features.view(B, T, self.config.feature_dim)

        if hidden is None:
            hidden = self.initial_hidden(B, device=frames.device)

        lstm_out, new_hidden = self.lstm(features, hidden)  # (B, T, lstm_hidden)

        steering_logits = self.steering_head(lstm_out)  # (B, T, n_bins)
        button_logits = {
            name: head(lstm_out).squeeze(-1)  # (B, T)
            for name, head in self.button_heads.items()
        }
        logits: dict[str, torch.Tensor] = {"steering": steering_logits}
        logits.update(button_logits)

        return logits, new_hidden

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def bc_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    steering_weight: float = 1.0,
    button_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Mixed loss: CE over steering bins + mean BCE over buttons.

    Args:
        logits: output of BCPolicy.forward.
        targets: dict with
            'steering_bin': (B, T) long
            'accelerate', 'brake', 'drift', 'item': (B, T) float in {0, 1}
        steering_weight, button_weight: scalar weights.

    Returns a dict with:
        'total': scalar
        'steering': scalar CE
        'buttons': scalar mean BCE
        per-button losses for diagnostics.
    """
    # Steering CE: flatten (B*T, n_bins), targets (B*T,).
    steer_logits = logits["steering"]
    B, T, n_bins = steer_logits.shape
    ce = F.cross_entropy(
        steer_logits.reshape(B * T, n_bins),
        targets["steering_bin"].reshape(B * T).long(),
    )

    button_losses: dict[str, torch.Tensor] = {}
    button_names = ("accelerate", "brake", "drift", "item")
    for name in button_names:
        button_losses[name] = F.binary_cross_entropy_with_logits(logits[name], targets[name].float())

    buttons_mean = torch.stack([button_losses[n] for n in button_names]).mean()
    total = steering_weight * ce + button_weight * buttons_mean

    return {
        "total": total,
        "steering": ce,
        "buttons": buttons_mean,
        **{f"button_{n}": button_losses[n] for n in button_names},
    }
