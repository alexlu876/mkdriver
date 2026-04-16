"""Tests for the hardened pairing (MKW_RL_SPEC.md §1.4)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkw_rl.dtm.pairing import (
    PairingError,
    _divergence_threshold,
    pair_dtm_and_frames,
)
from mkw_rl.dtm.parser import build_dtm_blob, build_frame


def _write_synthetic(
    tmp_path: Path, n_inputs: int, n_frames: int, *, lag_count: int = 0, from_savestate: int = 1
) -> tuple[Path, Path]:
    dtm_path = tmp_path / "d.dtm"
    frame_dir = tmp_path / "f"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = [build_frame(accelerate=True, analog_x=100 + (i % 100)) for i in range(n_inputs)]
    blob = build_dtm_blob(
        vi_count=n_inputs + lag_count,
        input_count=n_inputs,
        lag_count=lag_count,
        from_savestate=from_savestate,
        frames=frames,
    )
    dtm_path.write_bytes(blob)
    for i in range(n_frames):
        arr = np.full((60, 80, 3), (i % 256, 0, 0), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(frame_dir / f"framedump_{i}.png")
    return dtm_path, frame_dir


# ---------------------------------------------------------------------------
# Threshold.
# ---------------------------------------------------------------------------


class TestDivergenceThreshold:
    def test_floor_is_30(self) -> None:
        assert _divergence_threshold(0) == 30
        assert _divergence_threshold(100) == 30  # 2% of 100 = 2, floor wins

    def test_scales_with_inputs(self) -> None:
        assert _divergence_threshold(10_000) == 200  # 2% of 10k
        assert _divergence_threshold(100_000) == 2_000


# ---------------------------------------------------------------------------
# Core pairing behavior.
# ---------------------------------------------------------------------------


class TestPairingCore:
    def test_equal_lengths_default_tail_margin(self, tmp_path: Path) -> None:
        dtm, fr = _write_synthetic(tmp_path, 100, 100)
        pairs = pair_dtm_and_frames(dtm, fr)  # default tail_margin=10
        assert len(pairs) == 90
        assert pairs[0].input_frame_idx == 0
        assert pairs[-1].input_frame_idx == 89

    def test_equal_lengths_no_tail_margin(self, tmp_path: Path) -> None:
        dtm, fr = _write_synthetic(tmp_path, 50, 50)
        pairs = pair_dtm_and_frames(dtm, fr, tail_margin=0)
        assert len(pairs) == 50

    def test_skip_first_n_applied_symmetrically(self, tmp_path: Path) -> None:
        dtm, fr = _write_synthetic(tmp_path, 200, 200)
        pairs = pair_dtm_and_frames(dtm, fr, skip_first_n=30, tail_margin=10)
        assert len(pairs) == 200 - 30 - 10
        assert pairs[0].input_frame_idx == 30
        assert pairs[0].frame_idx == 0  # frame_idx is relative to the trimmed start
        assert pairs[-1].input_frame_idx == 200 - 10 - 1

    def test_frames_shorter_than_inputs(self, tmp_path: Path) -> None:
        dtm, fr = _write_synthetic(tmp_path, 100, 80)
        pairs = pair_dtm_and_frames(dtm, fr, tail_margin=0)
        # min(100, 80) = 80
        assert len(pairs) == 80

    def test_inputs_shorter_than_frames(self, tmp_path: Path) -> None:
        dtm, fr = _write_synthetic(tmp_path, 60, 100)
        pairs = pair_dtm_and_frames(dtm, fr, tail_margin=0)
        assert len(pairs) == 60

    def test_skip_too_aggressive_raises(self, tmp_path: Path) -> None:
        dtm, fr = _write_synthetic(tmp_path, 40, 40)
        with pytest.raises(PairingError, match="zero samples"):
            pair_dtm_and_frames(dtm, fr, skip_first_n=35, tail_margin=10)

    def test_negative_skip_raises(self, tmp_path: Path) -> None:
        dtm, fr = _write_synthetic(tmp_path, 20, 20)
        with pytest.raises(ValueError):
            pair_dtm_and_frames(dtm, fr, skip_first_n=-1)

    def test_negative_tail_margin_raises(self, tmp_path: Path) -> None:
        dtm, fr = _write_synthetic(tmp_path, 20, 20)
        with pytest.raises(ValueError):
            pair_dtm_and_frames(dtm, fr, tail_margin=-1)


# ---------------------------------------------------------------------------
# Warnings.
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_divergence_below_threshold_is_silent(self, tmp_path: Path, caplog) -> None:
        dtm, fr = _write_synthetic(tmp_path, 1000, 995)  # diff=5, threshold=30
        with caplog.at_level(logging.WARNING, logger="mkw_rl.dtm.pairing"):
            pair_dtm_and_frames(dtm, fr, tail_margin=0)
        assert not any("alignment issues" in r.message for r in caplog.records)

    def test_divergence_above_threshold_warns(self, tmp_path: Path, caplog) -> None:
        # diff=50, threshold=max(30, 0.02*100)=30
        dtm, fr = _write_synthetic(tmp_path, 100, 50)
        with caplog.at_level(logging.WARNING, logger="mkw_rl.dtm.pairing"):
            pair_dtm_and_frames(dtm, fr, tail_margin=0)
        assert any("alignment issues" in r.message for r in caplog.records)

    def test_lag_count_nonzero_warns(self, tmp_path: Path, caplog) -> None:
        dtm, fr = _write_synthetic(tmp_path, 100, 100, lag_count=5)
        with caplog.at_level(logging.WARNING, logger="mkw_rl.dtm.pairing"):
            pair_dtm_and_frames(dtm, fr, tail_margin=0)
        assert any("lag_count=5" in r.message for r in caplog.records)

    def test_lag_count_zero_silent(self, tmp_path: Path, caplog) -> None:
        dtm, fr = _write_synthetic(tmp_path, 100, 100, lag_count=0)
        with caplog.at_level(logging.WARNING, logger="mkw_rl.dtm.pairing"):
            pair_dtm_and_frames(dtm, fr, tail_margin=0)
        assert not any("lag_count" in r.message for r in caplog.records)

    def test_from_savestate_false_warns(self, tmp_path: Path, caplog) -> None:
        dtm, fr = _write_synthetic(tmp_path, 100, 100, from_savestate=0)
        with caplog.at_level(logging.WARNING, logger="mkw_rl.dtm.pairing"):
            pair_dtm_and_frames(dtm, fr, tail_margin=0)
        assert any("from_savestate=False" in r.message for r in caplog.records)

    def test_from_savestate_true_silent_on_flag(self, tmp_path: Path, caplog) -> None:
        dtm, fr = _write_synthetic(tmp_path, 100, 100, from_savestate=1)
        with caplog.at_level(logging.WARNING, logger="mkw_rl.dtm.pairing"):
            pair_dtm_and_frames(dtm, fr, tail_margin=0)
        assert not any("from_savestate" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Empty edge cases.
# ---------------------------------------------------------------------------


class TestEmpty:
    def test_empty_dtm_frames_raises(self, tmp_path: Path) -> None:
        # Zero input frames, zero PNGs.
        dtm_path = tmp_path / "d.dtm"
        frame_dir = tmp_path / "f"
        frame_dir.mkdir()
        dtm_path.write_bytes(build_dtm_blob(input_count=0, vi_count=0, frames=[]))
        # load_frame_dump raises on no PNGs — before pairing even runs.
        with pytest.raises(FileNotFoundError):
            pair_dtm_and_frames(dtm_path, frame_dir)

    def test_short_demo_under_tail_margin_raises(self, tmp_path: Path) -> None:
        dtm, fr = _write_synthetic(tmp_path, 5, 5)
        with pytest.raises(PairingError):
            pair_dtm_and_frames(dtm, fr, tail_margin=10)


# ---------------------------------------------------------------------------
# Integrity: paired frame/input indices stay in lockstep.
# ---------------------------------------------------------------------------


def test_frame_and_input_indices_stay_in_lockstep(tmp_path: Path) -> None:
    dtm, fr = _write_synthetic(tmp_path, 200, 200)
    pairs = pair_dtm_and_frames(dtm, fr, skip_first_n=50, tail_margin=25)
    for p in pairs:
        # After skip, input_frame_idx should equal 50 + frame_idx.
        assert p.input_frame_idx == 50 + p.frame_idx
