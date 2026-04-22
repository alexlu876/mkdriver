"""Multi-track BTR training — IMPALA+LSTM policy, IQN/Munchausen loss, R2D2 replay.

See ``docs/TRAINING_METHODOLOGY.md`` for the algorithmic spec and
``docs/PIVOT_2026-04-17.md`` for the strategic context (v2 on top of
VIPTankz's published v1 BTR).

Components:
- ``networks`` — FactorizedNoisyLinear (noisy-nets), Dueling branch. Ported
  verbatim from VIPTankz's ``BTR.py`` with light formatting cleanup.
- ``replay`` — PER + SumTree + R2D2 sequence sampling (``sample_sequences``).
  Pass 1 ported VIPTankz's transition-level replay; pass 3 added the
  sequence-level sampler without breaking the transition API.
- ``model`` — BTRPolicy composing ``mkw_rl.bc.model.ImpalaEncoder`` + LSTM +
  IQN heads, chosen for direct BC↔BTR weight compatibility at the encoder
  and LSTM.
- ``track_sampler`` — progress-weighted track picker. Maintains a
  per-track EMA of episode return; the training loop calls
  ``sampler.sample()`` before each ``env.reset()`` and
  ``sampler.update(slug, episode_return)`` after each rollout.
"""

from mkw_rl.rl.model import BTRConfig, BTRPolicy
from mkw_rl.rl.networks import Dueling, FactorizedNoisyLinear
from mkw_rl.rl.replay import PER, SumTree
from mkw_rl.rl.track_sampler import (
    ProgressWeightedTrackSampler,
    TrackSamplerConfig,
    construct_from_available,
)

__all__ = [
    "PER",
    "BTRConfig",
    "BTRPolicy",
    "Dueling",
    "FactorizedNoisyLinear",
    "ProgressWeightedTrackSampler",
    "SumTree",
    "TrackSamplerConfig",
    "construct_from_available",
]
