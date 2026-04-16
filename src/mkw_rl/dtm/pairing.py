"""Pair .dtm controller states with rendered frames.

This module ships in two stages:

* **Phase 1b** (this file, initial): minimal alignment — take the leading
  ``min(len(inputs), len(frames))`` from both sequences, matching frame 0
  to input frame 0. This is what Prompt 1b needs so the visualizer can
  run against a real recording. Tail is untrimmed; no menu skip; no
  length-divergence warnings.

* **Phase 1c**: harden per MKW_RL_SPEC.md §1.4 — add ``skip_first_n``
  (from the savestate JSON sidecar), ``tail_margin``, and a divergence
  warning at ``max(30, 0.02 * len(inputs))``.

The alignment direction is **from the start** (not from the end, per v1
spec). Rationale: recordings are anchored to a known savestate — the
start is canonical, the end is ragged (stop-recording timing).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .frames import FrameDump, load_frame_dump
from .parser import ControllerState, parse_dtm


@dataclass
class PairedSample:
    """One frame paired with its corresponding controller input."""

    frame_idx: int  # 0-indexed, after any skip/truncation
    input_frame_idx: int  # index into the original .dtm input list
    frame_path: Path
    controller: ControllerState


def pair_dtm_and_frames(
    dtm_path: Path | str,
    frame_dir: Path | str,
    skip_first_n: int = 0,  # hardened in Phase 1c; currently accepted but unused
    tail_margin: int = 0,  # hardened in Phase 1c
) -> list[PairedSample]:
    """Align a .dtm and its frame dump into paired samples.

    Phase 1b implementation: naive start-aligned ``min(len)`` truncation.
    ``skip_first_n`` and ``tail_margin`` are accepted so the call signature
    is stable between 1b and 1c.

    TODO(phase-1c): implement full §1.4 alignment: skip_first_n applied
    to both streams, tail_margin trim, divergence warning with threshold
    ``max(30, 0.02 * len(inputs))``, lag_count > 0 warning.
    """
    dtm_path = Path(dtm_path)
    frame_dir = Path(frame_dir)

    _, controller_states = parse_dtm(dtm_path)
    frame_dump: FrameDump = load_frame_dump(frame_dir)

    n = min(len(controller_states), len(frame_dump.frame_paths))

    # Apply skip_first_n symmetrically (even in the minimal version — the
    # common case where the user set skip_first_n=300 in the savestate
    # sidecar is worth supporting from day one, it's cheap).
    start = max(0, skip_first_n)
    end = max(start, n - max(0, tail_margin))

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
