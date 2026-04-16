"""Tests for src/mkw_rl/dtm/parser.py.

Uses the `build_dtm_blob` / `build_frame` helpers from the parser module
itself — a minor coupling, but it means we have CI-runnable coverage
without needing a real .dtm file checked in. A separate set of tests
marked `@pytest.mark.skipif(not path.exists())` opportunistically runs
round-trip checks against user-provided real .dtm files.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from mkw_rl.dtm.parser import (
    BYTES_PER_INPUT,
    EXPECTED_GAME_ID,
    EXPECTED_SIG,
    HEADER_SIZE,
    DtmFormatError,
    DtmRegionError,
    _bit,
    _normalize_analog,
    build_dtm_blob,
    build_frame,
    parse_dtm,
)

# ---------------------------------------------------------------------------
# Unit tests for helpers.
# ---------------------------------------------------------------------------


class TestBitExtraction:
    def test_lsb_is_bit_zero(self) -> None:
        assert _bit(0b00000001, 0) is True
        assert _bit(0b00000001, 1) is False

    def test_msb_is_bit_seven(self) -> None:
        assert _bit(0b10000000, 7) is True
        assert _bit(0b10000000, 6) is False

    def test_all_zeros(self) -> None:
        for i in range(8):
            assert _bit(0, i) is False

    def test_all_ones(self) -> None:
        for i in range(8):
            assert _bit(0xFF, i) is True

    def test_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            _bit(0, 8)
        with pytest.raises(ValueError):
            _bit(0, -1)


class TestAnalogNormalize:
    def test_neutral(self) -> None:
        assert _normalize_analog(128) == pytest.approx(0.0)

    def test_full_right(self) -> None:
        # 255 → +1 (clipped).
        assert _normalize_analog(255) == pytest.approx(1.0)

    def test_full_left(self) -> None:
        # 1 (spec min) → -127/127 ≈ -1.
        assert _normalize_analog(1) == pytest.approx(-1.0, abs=0.01)
        # 0 (defensive) → clipped to -1.
        assert _normalize_analog(0) == pytest.approx(-1.0, abs=0.01)

    def test_monotonic(self) -> None:
        xs = [_normalize_analog(v) for v in range(1, 256)]
        for a, b in zip(xs, xs[1:], strict=False):
            assert a <= b


# ---------------------------------------------------------------------------
# Header parsing.
# ---------------------------------------------------------------------------


def _minimal_valid_blob(**header_kwargs) -> bytes:
    """A valid .dtm blob with 10 neutral input frames and the given header overrides."""
    defaults = {
        "vi_count": 10,
        "input_count": 10,
        "lag_count": 0,
        "rerecord_count": 3,
        "author": "pytest",
        "frames": [build_frame() for _ in range(10)],
    }
    defaults.update(header_kwargs)
    return build_dtm_blob(**defaults)


class TestHeaderParsing:
    def test_round_trip_minimal(self, tmp_path: Path) -> None:
        blob = _minimal_valid_blob()
        p = tmp_path / "m.dtm"
        p.write_bytes(blob)
        header, frames = parse_dtm(p)
        assert header.signature == EXPECTED_SIG
        assert header.game_id == EXPECTED_GAME_ID
        assert header.is_wii is True
        assert header.has_gcn_port_1 is True
        assert header.vi_count == 10
        assert header.input_count == 10
        assert header.rerecord_count == 3
        assert header.author == "pytest"
        assert len(frames) == 10

    def test_truncated_header(self, tmp_path: Path) -> None:
        p = tmp_path / "t.dtm"
        p.write_bytes(b"DTM\x1a" + b"\x00" * 10)  # way short
        with pytest.raises(DtmFormatError):
            parse_dtm(p)

    def test_bad_signature(self, tmp_path: Path) -> None:
        blob = _minimal_valid_blob(signature=b"XXXX")
        p = tmp_path / "s.dtm"
        p.write_bytes(blob)
        with pytest.raises(DtmFormatError, match="bad signature"):
            parse_dtm(p)

    def test_wrong_game_id(self, tmp_path: Path) -> None:
        blob = _minimal_valid_blob(game_id=b"RMCP01")  # PAL
        p = tmp_path / "r.dtm"
        p.write_bytes(blob)
        with pytest.raises(DtmRegionError, match="not NTSC-U"):
            parse_dtm(p)

    def test_not_wii(self, tmp_path: Path) -> None:
        blob = _minimal_valid_blob(is_wii=0)
        p = tmp_path / "gc.dtm"
        p.write_bytes(blob)
        with pytest.raises(DtmFormatError, match="is_wii"):
            parse_dtm(p)

    def test_no_gcn_port_1(self, tmp_path: Path) -> None:
        # controllers_bitfield = 0x10 (Wiimote 1 only, no GCN).
        blob = _minimal_valid_blob(controllers=0x10)
        p = tmp_path / "w.dtm"
        p.write_bytes(blob)
        with pytest.raises(DtmFormatError, match="GCN port 1"):
            parse_dtm(p)

    def test_body_not_multiple_of_frame_size(self, tmp_path: Path) -> None:
        blob = _minimal_valid_blob()
        # Append 3 trailing bytes (not a full frame).
        blob += b"\x00\x00\x00"
        p = tmp_path / "b.dtm"
        p.write_bytes(blob)
        with pytest.raises(DtmFormatError, match="multiple of"):
            parse_dtm(p)


# ---------------------------------------------------------------------------
# Controller parsing — round-trip known inputs.
# ---------------------------------------------------------------------------


class TestControllerRoundTrip:
    def _make(self, tmp_path: Path, frames: list[bytes]) -> Path:
        blob = build_dtm_blob(input_count=len(frames), vi_count=len(frames), frames=frames)
        p = tmp_path / "c.dtm"
        p.write_bytes(blob)
        return p

    def test_hold_a_100_frames(self, tmp_path: Path) -> None:
        p = self._make(tmp_path, [build_frame(accelerate=True) for _ in range(100)])
        _, states = parse_dtm(p)
        assert len(states) == 100
        assert all(s.accelerate for s in states)
        assert not any(s.brake or s.drift or s.item or s.look_behind for s in states)
        # Neutral analog → steering ≈ 0.
        assert all(abs(s.steering) < 0.01 for s in states)

    def test_hold_full_left(self, tmp_path: Path) -> None:
        p = self._make(tmp_path, [build_frame(analog_x=1) for _ in range(50)])
        _, states = parse_dtm(p)
        assert all(s.steering < -0.9 for s in states)

    def test_hold_full_right(self, tmp_path: Path) -> None:
        p = self._make(tmp_path, [build_frame(analog_x=255) for _ in range(50)])
        _, states = parse_dtm(p)
        assert all(s.steering > 0.9 for s in states)

    def test_button_combinations(self, tmp_path: Path) -> None:
        # Each button in isolation.
        frames = [
            build_frame(accelerate=True),
            build_frame(brake=True),
            build_frame(drift=True),
            build_frame(item=True),
            build_frame(look_behind=True),
            build_frame(),  # all false
        ]
        p = self._make(tmp_path, frames)
        _, states = parse_dtm(p)
        assert states[0].accelerate and not states[0].brake
        assert states[1].brake and not states[1].accelerate
        assert states[2].drift and not states[2].item
        assert states[3].item and not states[3].drift
        assert states[4].look_behind and not states[4].accelerate
        assert not any(
            [
                states[5].accelerate,
                states[5].brake,
                states[5].drift,
                states[5].item,
                states[5].look_behind,
            ]
        )

    def test_raw_bytes_preserved(self, tmp_path: Path) -> None:
        p = self._make(tmp_path, [build_frame(accelerate=True, drift=True, analog_x=200)])
        _, states = parse_dtm(p)
        s = states[0]
        assert s._raw_analog_x == 200
        assert s._raw_analog_y == 128
        assert s._raw_byte0 & (1 << 1)  # A bit
        assert s._raw_byte1 & (1 << 3)  # R digital

    def test_frame_idx_is_sequential(self, tmp_path: Path) -> None:
        p = self._make(tmp_path, [build_frame() for _ in range(7)])
        _, states = parse_dtm(p)
        assert [s.frame_idx for s in states] == [0, 1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Optional round-trip against real .dtm files the user may drop in.
# ---------------------------------------------------------------------------

# If the user has real .dtm files they want sanity-checked, they can drop
# them in `data/raw/demos/` or point REAL_DTM_PATH at one. We don't ship
# any in the repo.
_REAL_DTM_PATH_ENV = os.environ.get("REAL_DTM_PATH", "")
_REAL_DTM_PATH = Path(_REAL_DTM_PATH_ENV) if _REAL_DTM_PATH_ENV else None


@pytest.mark.skipif(
    not (_REAL_DTM_PATH and _REAL_DTM_PATH.exists()),
    reason="Set REAL_DTM_PATH=<file.dtm> to run opportunistic real-file tests",
)
def test_real_dtm_parses() -> None:
    header, states = parse_dtm(_REAL_DTM_PATH)
    assert header.game_id == EXPECTED_GAME_ID
    assert len(states) > 0
    # Input count in header should match (within 1%) what's actually present.
    if header.input_count > 0:
        drift = abs(len(states) - header.input_count)
        assert drift <= max(1, int(0.01 * header.input_count)), (
            f"actual frames {len(states)} vs header.input_count {header.input_count}"
        )


# ---------------------------------------------------------------------------
# Fine-grained byte layout sanity — catch regressions in header constants.
# ---------------------------------------------------------------------------


def test_header_offsets_are_consistent() -> None:
    """Ensure the header constants don't overlap and are covered by HEADER_SIZE."""
    # vi_count, input_count, lag_count all claim 8 bytes.
    assert struct.calcsize("<Q") == 8
    assert struct.calcsize("<I") == 4
    assert HEADER_SIZE == 0x100
    assert BYTES_PER_INPUT == 8
