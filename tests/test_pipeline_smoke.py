"""End-to-end smoke tests for the Phase 1b pipeline.

Builds a synthetic .dtm + matching PNG frame directory, runs them through
parser → pairing → visualizer, and verifies an MP4 is produced. This is
CI-runnable — no real data required.

The point is to catch integration regressions (API drift, import breaks,
format mismatches) without waiting for a human to record real footage.
Visual correctness of the overlay is still gated on the human watching
`scripts/sanity_check.py` output against real footage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mkw_rl.dtm.frames import load_frame, load_frame_dump
from mkw_rl.dtm.pairing import pair_dtm_and_frames
from mkw_rl.dtm.parser import build_dtm_blob, build_frame
from mkw_rl.dtm.viz import render_overlay, write_overlay_video


def _write_synthetic_frames(frame_dir: Path, n: int) -> None:
    """Write ``n`` synthetic 320x240 RGB PNGs named framedump_0.png, ..."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # Each frame is a solid-color field that varies with index, so the
        # visualizer output isn't just black.
        arr = np.full((240, 320, 3), (i * 3 % 256, i * 5 % 256, 100), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(frame_dir / f"framedump_{i}.png")


def _write_synthetic_dtm(dtm_path: Path, n: int) -> None:
    """Write an n-frame .dtm file with varied inputs."""
    frames = []
    for i in range(n):
        # Cycle through a few input patterns so the overlay has something to render.
        frames.append(
            build_frame(
                accelerate=(i % 3 != 0),
                drift=(i % 10 < 3),
                analog_x=(128 + int(100 * np.sin(i * 0.1))) & 0xFF,
            )
        )
    blob = build_dtm_blob(vi_count=n, input_count=n, frames=frames)
    dtm_path.parent.mkdir(parents=True, exist_ok=True)
    dtm_path.write_bytes(blob)


# ---------------------------------------------------------------------------
# Frame dump loading.
# ---------------------------------------------------------------------------


class TestFrameDumpLoader:
    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_frame_dump(tmp_path / "does_not_exist")

    def test_no_pngs_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="no PNGs"):
            load_frame_dump(empty)

    def test_sorts_numerically(self, tmp_path: Path) -> None:
        # Write out-of-order to verify numeric sort (not lexicographic).
        for i in [10, 2, 1, 20, 3]:
            Image.new("RGB", (4, 4)).save(tmp_path / f"framedump_{i}.png")
        dump = load_frame_dump(tmp_path)
        indexed = [int(p.stem.split("_")[-1]) for p in dump.frame_paths]
        assert indexed == [1, 2, 3, 10, 20]

    def test_mixed_naming_patterns(self, tmp_path: Path) -> None:
        # Mix of naming patterns Dolphin has used historically.
        names = ["frame_001.png", "frame_002.png", "frame_003.png"]
        for n in names:
            Image.new("RGB", (4, 4)).save(tmp_path / n)
        dump = load_frame_dump(tmp_path)
        assert [p.name for p in dump.frame_paths] == names


class TestLoadFrame:
    def test_grayscale_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "f.png"
        Image.new("RGB", (320, 240), color=(100, 150, 200)).save(p)
        arr = load_frame(p, size=(140, 114), grayscale=True)
        assert arr.shape == (114, 140)
        assert arr.dtype == np.uint8

    def test_rgb_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "f.png"
        Image.new("RGB", (320, 240), color=(100, 150, 200)).save(p)
        arr = load_frame(p, size=(140, 114), grayscale=False)
        assert arr.shape == (114, 140, 3)


# ---------------------------------------------------------------------------
# Pairing (minimal 1b impl).
# ---------------------------------------------------------------------------


class TestMinimalPairing:
    # Smoke tests exercise the end-to-end pipeline on tiny synthetic demos.
    # We pass tail_margin=0 explicitly because the hardened default of 10
    # would trim our 20-frame synthetic demos to nothing useful.

    def test_pairs_equal_length(self, tmp_path: Path) -> None:
        dtm = tmp_path / "d.dtm"
        frames = tmp_path / "f"
        _write_synthetic_dtm(dtm, 20)
        _write_synthetic_frames(frames, 20)
        pairs = pair_dtm_and_frames(dtm, frames, tail_margin=0)
        assert len(pairs) == 20
        assert [p.frame_idx for p in pairs] == list(range(20))
        assert [p.input_frame_idx for p in pairs] == list(range(20))

    def test_pairs_truncate_to_min(self, tmp_path: Path) -> None:
        dtm = tmp_path / "d.dtm"
        frames = tmp_path / "f"
        _write_synthetic_dtm(dtm, 25)
        _write_synthetic_frames(frames, 20)
        pairs = pair_dtm_and_frames(dtm, frames, tail_margin=0)
        assert len(pairs) == 20

    def test_skip_first_n_applied(self, tmp_path: Path) -> None:
        dtm = tmp_path / "d.dtm"
        frames = tmp_path / "f"
        _write_synthetic_dtm(dtm, 30)
        _write_synthetic_frames(frames, 30)
        pairs = pair_dtm_and_frames(dtm, frames, skip_first_n=5, tail_margin=0)
        assert len(pairs) == 25
        assert pairs[0].input_frame_idx == 5
        assert pairs[0].frame_idx == 0


# ---------------------------------------------------------------------------
# Visualizer.
# ---------------------------------------------------------------------------


class TestVisualizer:
    def test_render_single_overlay(self, tmp_path: Path) -> None:
        dtm = tmp_path / "d.dtm"
        frames = tmp_path / "f"
        _write_synthetic_dtm(dtm, 5)
        _write_synthetic_frames(frames, 5)
        pairs = pair_dtm_and_frames(dtm, frames, tail_margin=0)
        out = render_overlay(pairs[0])
        assert out.mode == "RGB"
        assert out.size == (320, 240)

    def test_write_overlay_video_produces_mp4(self, tmp_path: Path) -> None:
        dtm = tmp_path / "d.dtm"
        frames = tmp_path / "f"
        _write_synthetic_dtm(dtm, 60)  # 1 second at 60 fps
        _write_synthetic_frames(frames, 60)
        pairs = pair_dtm_and_frames(dtm, frames, tail_margin=0)
        out = tmp_path / "overlay.mp4"
        # Small fps keeps encoding fast in CI.
        result = write_overlay_video(pairs, out, fps=30, n_seconds=None)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 1024  # non-trivial file

    def test_write_overlay_video_zero_samples_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no samples"):
            write_overlay_video([], tmp_path / "x.mp4")
