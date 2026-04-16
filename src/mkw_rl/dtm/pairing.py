"""Pair .dtm controller states with rendered frames (hardened).

Implements the full MKW_RL_SPEC.md §1.4 alignment strategy:

* **Start-aligned**: frame 0 of the frame dump corresponds to input frame 0
  of the .dtm (both anchored to the same savestate load). Alignment from the
  end — v1's approach — is incorrect for savestate-anchored recordings
  because the end is the ragged edge.
* **skip_first_n**: drop this many initial input frames to skip menu /
  countdown / HUD-noisy frames. Supplied per savestate via the JSON sidecar
  (see docs/SAVESTATE_PROTOCOL.md). Applied symmetrically to both streams.
* **tail_margin**: drop this many trailing frames to account for the
  ragged stop-recording end. Default 10.
* **Divergence warning**: if ``abs(len(frames) - len(inputs))`` exceeds
  ``max(30, 0.02 * len(inputs))``, we log a warning with the actual counts.
  This is loose-on-purpose: the replay pipeline has legitimate buffering
  that can produce small discrepancies.
* **lag_count warning**: if the .dtm's ``lag_count`` is nonzero, we warn —
  the demo is still usable but alignment may drift. Treat as second-class
  training data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .frames import FrameDump, load_frame_dump
from .parser import ControllerState, parse_dtm

log = logging.getLogger(__name__)


@dataclass
class PairedSample:
    """One frame paired with its corresponding controller input."""

    frame_idx: int  # 0-indexed, after skip/trim
    input_frame_idx: int  # index into the original .dtm input list
    frame_path: Path
    controller: ControllerState


class PairingError(ValueError):
    """Raised when pairing cannot produce any samples at all."""


def _divergence_threshold(n_inputs: int) -> int:
    """Return the max allowed |len(frames) - len(inputs)| before warning."""
    return max(30, int(0.02 * n_inputs))


def pair_dtm_and_frames(
    dtm_path: Path | str,
    frame_dir: Path | str,
    skip_first_n: int = 0,
    tail_margin: int = 10,
) -> list[PairedSample]:
    """Align a .dtm and its frame dump into paired samples.

    Args:
        dtm_path: Path to the .dtm.
        frame_dir: Path to the frame dump directory (PNG sequence).
        skip_first_n: Number of leading input frames to drop from both
            streams. Typically read from the savestate's JSON sidecar.
        tail_margin: Number of trailing frames to drop. Covers the ragged
            edge of stop-recording timing.

    Returns:
        List of PairedSample in emission order.

    Raises:
        PairingError: if the alignment produces zero samples (empty
            inputs, empty frames, or skip/trim exceeding length).
    """
    dtm_path = Path(dtm_path)
    frame_dir = Path(frame_dir)

    header, controller_states = parse_dtm(dtm_path)
    frame_dump: FrameDump = load_frame_dump(frame_dir)

    n_inputs = len(controller_states)
    n_frames = len(frame_dump.frame_paths)

    # Divergence check — loose threshold, just warn.
    divergence = abs(n_frames - n_inputs)
    threshold = _divergence_threshold(n_inputs)
    if divergence > threshold:
        log.warning(
            "pair_dtm_and_frames: |len(frames) - len(inputs)| = %d > threshold %d "
            "(n_frames=%d, n_inputs=%d). This demo may have alignment issues.",
            divergence,
            threshold,
            n_frames,
            n_inputs,
        )

    # lag_count warning — treat nonzero lag as a second-class demo.
    if header.lag_count > 0:
        log.warning(
            "pair_dtm_and_frames: .dtm header reports lag_count=%d. Alignment may "
            "drift at lag points. Consider this demo second-class training data.",
            header.lag_count,
        )

    # skip_first_n / tail_margin arithmetic. All applied symmetrically
    # so frame_idx and input_frame_idx stay in lockstep after trimming.
    n = min(n_inputs, n_frames)
    if skip_first_n < 0:
        raise ValueError(f"skip_first_n must be >= 0, got {skip_first_n}")
    if tail_margin < 0:
        raise ValueError(f"tail_margin must be >= 0, got {tail_margin}")

    start = skip_first_n
    end = n - tail_margin
    if end <= start:
        raise PairingError(
            f"pairing produced zero samples: n_inputs={n_inputs}, n_frames={n_frames}, "
            f"skip_first_n={skip_first_n}, tail_margin={tail_margin}"
        )

    pairs: list[PairedSample] = []
    for rel_idx, i in enumerate(range(start, end)):
        pairs.append(
            PairedSample(
                frame_idx=rel_idx,
                input_frame_idx=i,
                frame_path=frame_dump.frame_paths[i],
                controller=controller_states[i],
            )
        )
    return pairs
