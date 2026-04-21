"""BTR network components — ported verbatim from VIPTankz's ``BTR.py``.

Two building blocks live here:

- ``FactorizedNoisyLinear``: Noisy-Nets parameterization of ``nn.Linear``
  (factorized Gaussian noise per Fortunato et al. 2017). Used inside the
  IQN dueling heads. Replaces explicit ε-greedy with learnable exploration.
- ``Dueling``: the standard dueling-DQN branch combining a scalar value
  head with a per-action advantage head into Q-values.

Ported from ``~/code/mkw/Wii-RL/BTR.py:27-117``. Line-by-line behavior
preserved; only formatting, type hints, and module-level docstrings have
been touched. The math, default hyperparameters, and parameter shapes
match exactly so future weight transfer from VIPTankz's pre-trained model
(if we ever want a sanity baseline) would require no architectural
adjustment.
"""

from __future__ import annotations

import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init


class FactorizedNoisyLinear(nn.Module):
    """Factorized Gaussian noise layer for noisy-nets DQN.

    Equations 10-11 of Fortunato et al. (2017). The weight is
    ``μ_w + σ_w ⊙ ε_w`` where ``ε_w = f(ε_out) ⊗ f(ε_in)`` and
    ``f(x) = sgn(x)√|x|``. Bias is the same with a single outer noise.

    ``disable_noise()`` zeroes ε so the layer becomes deterministic
    (used for greedy rollouts); ``reset_noise()`` resamples ε from
    ``N(0, 1)``. Both are no_grad.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_0: float = 0.5,
        self_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = μ^w + σ^w ⊙ ε^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        # bias: b = μ^b + σ^b ⊙ ε^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        if self_norm:
            self.reset_parameters_self_norm()
        else:
            self.reset_parameters()
        self.reset_noise()
        # Start with noise disabled — the Agent class will call reset_noise()
        # before each rollout / learn step.
        self.disable_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Similar to Kaiming uniform (He initialization) with fan_mode=fan_in.
        scale = 1 / sqrt(self.in_features)
        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)
        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def reset_parameters_self_norm(self) -> None:
        # Self-normalizing init variant. Kept for parity with VIPTankz's config
        # options even though we don't enable it by default.
        nn.init.normal_(self.weight_mu, std=1 / math.sqrt(self.out_features))
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x) · √|x|  — shapes the factorized noise per eq. 11.
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002 — match PyTorch convention
        return F.linear(
            input,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )


class Dueling(nn.Module):
    """Dueling-DQN branch: ``Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))``.

    Takes a feature tensor, routes through two sub-branches (value and
    advantage), and combines per the standard dueling formula. When
    ``advantages_only=True`` we short-circuit and return raw advantages —
    used for greedy action selection (argmax ignores the constant V(s)).
    """

    def __init__(self, value_branch: nn.Module, advantage_branch: nn.Module) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x: Tensor, advantages_only: bool = False) -> Tensor:
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages
        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))
