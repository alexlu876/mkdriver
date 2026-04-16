"""Action encoding for BC / RL training.

Steering in MKWii is effectively bimodal during racing: hard-left and
hard-right during drifts, near-zero during straights. MSE regression on
a bimodal target converges to the mean, producing a policy that drives
gently straight into walls.

We discretize steering into 21 equal-width bins over ``[-1, 1]``. The
number of bins is odd so exactly-zero steering maps to a single
canonical bin center. 21 is chosen to capture the drift-left / drift-
right / small-corrections resolution without making the output space
too coarse to hit a specific drift angle. Revisit if post-training
analysis shows the BC model struggling with fine steering.

Buttons (accelerate / brake / drift / item) remain binary — no
encoding needed here. Their prediction heads use BCE-with-logits.
"""

from __future__ import annotations

from typing import SupportsFloat

N_STEERING_BINS: int = 21
assert N_STEERING_BINS % 2 == 1, "N_STEERING_BINS must be odd so 0 is a bin center"

_BIN_WIDTH: float = 2.0 / N_STEERING_BINS  # bin width across [-1, 1]
_MAX_BIN: int = N_STEERING_BINS - 1
_CENTER_BIN: int = N_STEERING_BINS // 2


def encode_steering(x: SupportsFloat) -> int:
    """Map ``x`` in [-1, 1] to a bin index in [0, N_STEERING_BINS - 1].

    Values outside [-1, 1] are clipped. The mapping uses equal-width bins:
    bin ``i`` covers ``[-1 + i * width, -1 + (i+1) * width)`` for
    ``i < N_STEERING_BINS - 1``, and bin ``N_STEERING_BINS - 1`` covers
    the closed interval up to and including +1.
    """
    v = max(-1.0, min(1.0, float(x)))
    # Map [-1, 1] → [0, N). Inclusive handling of +1 below.
    idx = int((v + 1.0) / _BIN_WIDTH)
    if idx > _MAX_BIN:
        idx = _MAX_BIN
    return idx


def decode_steering(bin_idx: int) -> float:
    """Map a bin index back to the bin's center in ``[-1, 1]``.

    For bin ``i``, returns ``-1 + (i + 0.5) * bin_width``.
    """
    if not 0 <= bin_idx <= _MAX_BIN:
        raise ValueError(f"bin_idx must be in [0, {_MAX_BIN}], got {bin_idx}")
    return -1.0 + (bin_idx + 0.5) * _BIN_WIDTH


def bin_width() -> float:
    """The width of a single bin in the [-1, 1] domain."""
    return _BIN_WIDTH


def center_bin() -> int:
    """The index of the bin containing exact-zero steering."""
    return _CENTER_BIN
