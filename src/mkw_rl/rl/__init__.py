"""Multi-track BTR training — IMPALA+LSTM policy, IQN/Munchausen loss, R2D2 replay.

See ``docs/TRAINING_METHODOLOGY.md`` for the algorithmic spec and
``docs/PIVOT_2026-04-17.md`` for the strategic context (v2 on top of
VIPTankz's published v1 BTR).

Components:
- ``networks`` — FactorizedNoisyLinear (noisy-nets), Dueling branch. Ported
  verbatim from VIPTankz's ``BTR.py`` with light formatting cleanup.
- ``replay`` — PER + SumTree. Pass 1 mirrors VIPTankz; pass 3 will extend
  for R2D2-style burn-in sequence sampling.
- ``model`` — BTRPolicy composing ``mkw_rl.bc.model.ImpalaEncoder`` + LSTM +
  IQN heads, chosen for direct BC↔BTR weight compatibility at the encoder
  and LSTM.
- ``track_sampler`` (pass 4) — progress-weighted track picker.
"""

from mkw_rl.rl.model import BTRConfig, BTRPolicy
from mkw_rl.rl.networks import Dueling, FactorizedNoisyLinear
from mkw_rl.rl.replay import PER, SumTree

__all__ = [
    "PER",
    "BTRConfig",
    "BTRPolicy",
    "Dueling",
    "FactorizedNoisyLinear",
    "SumTree",
]
