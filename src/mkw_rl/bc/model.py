"""BC policy: IMPALA CNN encoder + stateful LSTM + mixed action heads.

Architecture (see MKW_RL_SPEC.md §2.2):

Input:   (B, T, stack=4, H=75, W=140), grayscale, float in [0, 1]

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
import torch.utils.checkpoint

from mkw_rl.dtm.action_encoding import N_STEERING_BINS

type LstmState = tuple[torch.Tensor, torch.Tensor]  # (h, c)


# ---------------------------------------------------------------------------
# IMPALA residual block.
# ---------------------------------------------------------------------------


def _maybe_spectral_norm(conv: nn.Conv2d, use: bool) -> nn.Module:
    """Wrap ``conv`` in ``nn.utils.parametrizations.spectral_norm`` when
    ``use`` is True. Matches VIPTankz's BTR.py default (``spectral=True``).

    Spectral norm bounds the Lipschitz constant of the layer's linear map by
    dividing the weight by its largest singular value (estimated via power
    iteration stored in buffers ``_u`` / ``_v``). Load-bearing for BTR
    training stability per the paper's ablations.
    """
    if use:
        return nn.utils.parametrizations.spectral_norm(conv)
    return conv


class _ImpalaResBlock(nn.Module):
    """Two 3x3 convs with ReLU pre-activations and a residual add. Convs may
    optionally be spectral-normed (matches VIPTankz default)."""

    def __init__(self, channels: int, use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.conv1 = _maybe_spectral_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), use_spectral_norm
        )
        self.conv2 = _maybe_spectral_norm(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), use_spectral_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(x)
        y = self.conv1(y)
        y = F.relu(y)
        y = self.conv2(y)
        return x + y


class _ImpalaBlock(nn.Module):
    """IMPALA block: 3x3 conv → maxpool(3x3, s=2) → [LayerNorm] → 2× residual
    blocks. The optional post-pool LayerNorm and the conv spectral-norm
    wrapping both match VIPTankz's defaults (BTR.py:148,153-154,162-163).
    LayerNorm's spatial shape is attached post-init by the owning encoder
    (which probes each block's output shape via a dummy forward)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = _maybe_spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            use_spectral_norm,
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Placeholder Identity; replaced with nn.LayerNorm(shape) post-init when
        # the encoder enables conv-block LN. Keeps shape invariant at probe time.
        self.norm: nn.Module = nn.Identity()
        self.res1 = _ImpalaResBlock(out_channels, use_spectral_norm=use_spectral_norm)
        self.res2 = _ImpalaResBlock(out_channels, use_spectral_norm=use_spectral_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.norm(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


# ---------------------------------------------------------------------------
# Encoder.
# ---------------------------------------------------------------------------


class ImpalaEncoder(nn.Module):
    """IMPALA CNN → ``feature_dim``-dim feature vector per frame.

    Optional knobs to match VIPTankz/BTR defaults (off by default for BC
    back-compat; BTR passes True for both):

    - ``use_spectral_norm``: wrap all conv layers in ``nn.utils.parametrizations.spectral_norm``.
      Bounds the Lipschitz constant of each conv; load-bearing for BTR
      training stability per the ICML paper's ablations.
    - ``layer_norm``: apply ``nn.LayerNorm`` on each block's post-pool feature
      map (matches VIPTankz BTR.py:148,153-154,162-163). Spatial shapes are
      probed via a dummy forward during init.

    The final ``Linear(flat_dim → feature_dim) + ReLU`` bottleneck is an
    mkw-rl v2 addition (VIPTankz's encoder feeds the flat conv features
    directly into the IQN cos-embedding mixing with no learned compression).
    We add this bottleneck so the downstream LSTM's input_size matches
    ``feature_dim``; see docs/TRAINING_METHODOLOGY.md §2.
    """

    def __init__(
        self,
        in_channels: int = 4,
        channels: tuple[int, int, int] = (16, 32, 32),
        feature_dim: int = 256,
        input_hw: tuple[int, int] = (75, 140),
        use_spectral_norm: bool = False,
        layer_norm: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.use_gradient_checkpointing = gradient_checkpointing

        c1, c2, c3 = channels
        self.block1 = _ImpalaBlock(in_channels, c1, use_spectral_norm=use_spectral_norm)
        self.block2 = _ImpalaBlock(c1, c2, use_spectral_norm=use_spectral_norm)
        self.block3 = _ImpalaBlock(c2, c3, use_spectral_norm=use_spectral_norm)

        # Two-phase probe: first get each block's post-pool shape (for LN
        # attachment), then compute flat_dim. Blocks currently have norm=Identity
        # so the probed shapes are correct regardless of whether LN is enabled.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            post_pool_shapes: list[torch.Size] = []
            y = dummy
            for blk in (self.block1, self.block2, self.block3):
                y = blk.conv(y)
                y = blk.pool(y)
                post_pool_shapes.append(y.shape[1:])  # (C, H, W)
                y = blk.res1(blk.norm(y))
                y = blk.res2(y)
            y = F.relu(y)
            flat_dim = y.flatten(1).shape[1]

        if layer_norm:
            # LayerNorm across (C, H, W) — matches VIPTankz's "normalized_shape"
            # arg passing the full post-pool feature-map shape.
            self.block1.norm = nn.LayerNorm(list(post_pool_shapes[0]))
            self.block2.norm = nn.LayerNorm(list(post_pool_shapes[1]))
            self.block3.norm = nn.LayerNorm(list(post_pool_shapes[2]))

        self.linear = nn.Linear(flat_dim, feature_dim)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        # Gradient checkpointing: only active when training. During inference
        # (target forward, eval, act()), we skip it — no backward graph needed.
        # Re-runs each block's forward during backward instead of storing
        # intermediate activations — for bs=128 × seq_len=60 BPTT, this drops
        # per-block activation storage from ~GB to near zero, trading ~30%
        # extra compute for ~80% less encoder memory. Without checkpointing
        # the BTR config at VIPTankz hyperparameters OOMs on 32 GB GPUs.
        if self.training and self.use_gradient_checkpointing:
            # use_reentrant=False is the modern-correct mode; retains
            # autocast state across the checkpoint boundary.
            x = torch.utils.checkpoint.checkpoint(
                self.block1, x, use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                self.block2, x, use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                self.block3, x, use_reentrant=False,
            )
        else:
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
    input_hw: tuple[int, int] = (75, 140)
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
