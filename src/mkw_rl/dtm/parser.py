"""Dolphin .dtm (movie) file parser.

References:
  - TASVideos DTM spec: https://tasvideos.org/EmulatorResources/Dolphin/DTM
  - Dolphin Source/Core/Core/Movie.cpp (ground truth when the wiki is ambiguous).

We only support NTSC-U MKWii (game_id == b"RMCE01") with GCN port 1 (GameCube
controller on first port). PAL / NTSC-J / Korean and Wiimote-only inputs are
rejected.

The parser returns a (DtmHeader, list[ControllerState]) pair. The header surfaces
the fields downstream code actually needs — vi_count, input_count, lag_count,
from_savestate. Raw bytes are preserved on each controller state for debugging.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Header layout constants (offsets and sizes in bytes).
HEADER_SIZE = 0x100
SIG_OFFSET = 0x000
SIG_SIZE = 4
GAME_ID_OFFSET = 0x004
GAME_ID_SIZE = 6
IS_WII_OFFSET = 0x00A
CONTROLLERS_OFFSET = 0x00B
FROM_SAVESTATE_OFFSET = 0x00C
VI_COUNT_OFFSET = 0x00D
INPUT_COUNT_OFFSET = 0x015
LAG_COUNT_OFFSET = 0x01D
RESERVED_OFFSET = 0x025
RERECORD_OFFSET = 0x02D
AUTHOR_OFFSET = 0x031
AUTHOR_SIZE = 32

EXPECTED_SIG = b"DTM\x1a"
EXPECTED_GAME_ID = b"RMCE01"
BYTES_PER_INPUT = 8


class DtmFormatError(ValueError):
    """Raised on malformed or truncated .dtm data."""


class DtmRegionError(ValueError):
    """Raised when the .dtm is for a game ID other than NTSC-U MKWii."""


@dataclass(frozen=True)
class DtmHeader:
    """Parsed .dtm header. Fields follow TASVideos / Movie.cpp naming."""

    signature: bytes
    game_id: bytes
    is_wii: bool
    controllers_bitfield: int
    from_savestate: bool
    vi_count: int
    input_count: int
    lag_count: int
    rerecord_count: int
    author: str

    @property
    def has_gcn_port_1(self) -> bool:
        return bool(self.controllers_bitfield & 0x01)


@dataclass(frozen=True)
class ControllerState:
    """Normalized controller state for a single input frame."""

    frame_idx: int  # 0-indexed input frame (not VI)

    # Semantic MKWii mapping.
    steering: float  # analog X, normalized to [-1, 1]
    accelerate: bool  # A
    brake: bool  # B
    drift: bool  # R digital
    item: bool  # L digital
    look_behind: bool  # X

    # Raw fields preserved for debugging / round-trip tests.
    _raw_analog_x: int = field(repr=False)
    _raw_analog_y: int = field(repr=False)
    _raw_byte0: int = field(repr=False)
    _raw_byte1: int = field(repr=False)


def _bit(byte: int, bit_idx: int) -> bool:
    """Extract bit `bit_idx` from `byte`. Bit 0 is LSB.

    Example: `_bit(0b00000001, 0) == True`, `_bit(0b00000001, 1) == False`.
    """
    if not 0 <= bit_idx <= 7:
        raise ValueError(f"bit_idx must be 0-7, got {bit_idx}")
    return bool((byte >> bit_idx) & 1)


def _normalize_analog(raw: int) -> float:
    """Map raw analog byte (1-255 with 128 neutral) to [-1, 1].

    The GCN controller's nominal analog range is 1..255 (not 0..255); 128 is
    mechanical neutral. Dolphin clamps output to the full 0..255 range, so we
    accept 0..255 inputs defensively.
    """
    centered = raw - 128
    # 127 is the max magnitude in either direction (128 → 0, 255 → +127, 0 → -128).
    # We divide by 127 and clip to [-1, 1].
    value = centered / 127.0
    return max(-1.0, min(1.0, value))


def _parse_header(data: bytes) -> DtmHeader:
    if len(data) < HEADER_SIZE:
        raise DtmFormatError(f"truncated header: got {len(data)} bytes, need {HEADER_SIZE}")

    signature = data[SIG_OFFSET : SIG_OFFSET + SIG_SIZE]
    if signature != EXPECTED_SIG:
        raise DtmFormatError(f"bad signature: {signature!r} (expected {EXPECTED_SIG!r})")

    game_id = data[GAME_ID_OFFSET : GAME_ID_OFFSET + GAME_ID_SIZE]
    if game_id != EXPECTED_GAME_ID:
        raise DtmRegionError(f"game_id {game_id!r} is not NTSC-U MKWii; this project is NTSC-U only")

    is_wii = data[IS_WII_OFFSET] == 1
    if not is_wii:
        raise DtmFormatError("is_wii byte != 1; .dtm is not a Wii title")

    controllers = data[CONTROLLERS_OFFSET]
    from_savestate = data[FROM_SAVESTATE_OFFSET] == 1

    # All multi-byte header ints are little-endian unsigned.
    (vi_count,) = struct.unpack_from("<Q", data, VI_COUNT_OFFSET)
    (input_count,) = struct.unpack_from("<Q", data, INPUT_COUNT_OFFSET)
    (lag_count,) = struct.unpack_from("<Q", data, LAG_COUNT_OFFSET)
    (rerecord_count,) = struct.unpack_from("<I", data, RERECORD_OFFSET)

    raw_author = data[AUTHOR_OFFSET : AUTHOR_OFFSET + AUTHOR_SIZE]
    author = raw_author.split(b"\x00", 1)[0].decode("utf-8", errors="replace")

    header = DtmHeader(
        signature=signature,
        game_id=game_id,
        is_wii=is_wii,
        controllers_bitfield=controllers,
        from_savestate=from_savestate,
        vi_count=vi_count,
        input_count=input_count,
        lag_count=lag_count,
        rerecord_count=rerecord_count,
        author=author,
    )

    if not header.has_gcn_port_1:
        raise DtmFormatError(f"controllers_bitfield 0b{controllers:08b} does not have GCN port 1 enabled")

    # Reject multi-controller recordings. This pipeline only supports a single
    # GCN controller on port 1 — bits 1-3 (other GCN ports) and bits 4-7
    # (Wiimotes) would produce variable-width input frames in the body and
    # silently misalign our 8-bytes-per-frame reader.
    if controllers != 0x01:
        raise DtmFormatError(
            f"controllers_bitfield=0x{controllers:02x} has controllers other than GCN port 1 "
            f"enabled (other GCN ports: bits 1-3; Wiimotes: bits 4-7). This pipeline "
            f"supports only single-GCN-port-1 .dtm."
        )

    return header


def _parse_controller_frame(frame_bytes: bytes, frame_idx: int) -> ControllerState:
    """Parse one 8-byte GCN input frame."""
    if len(frame_bytes) != BYTES_PER_INPUT:
        raise DtmFormatError(
            f"controller frame {frame_idx}: got {len(frame_bytes)} bytes, need {BYTES_PER_INPUT}"
        )

    byte0, byte1, _l_pressure, _r_pressure, analog_x, analog_y, _c_x, _c_y = frame_bytes

    # Byte 0 bits: Start(0), A(1), B(2), X(3), Y(4), Z(5), D-Up(6), D-Down(7)
    # Byte 1 bits: D-Left(0), D-Right(1), L digital(2), R digital(3), disc_change(4),
    #              reset(5), controller_connected(6), origin_reset(7)
    accelerate = _bit(byte0, 1)  # A
    brake = _bit(byte0, 2)  # B
    look_behind = _bit(byte0, 3)  # X
    item = _bit(byte1, 2)  # L digital
    drift = _bit(byte1, 3)  # R digital

    steering = _normalize_analog(analog_x)

    return ControllerState(
        frame_idx=frame_idx,
        steering=steering,
        accelerate=accelerate,
        brake=brake,
        drift=drift,
        item=item,
        look_behind=look_behind,
        _raw_analog_x=analog_x,
        _raw_analog_y=analog_y,
        _raw_byte0=byte0,
        _raw_byte1=byte1,
    )


def parse_dtm(path: Path | str) -> tuple[DtmHeader, list[ControllerState]]:
    """Parse a .dtm file into (header, per-frame controller states).

    Validates NTSC-U (RMCE01) and GCN port 1. Raises DtmFormatError on
    malformed input, DtmRegionError on wrong game ID.
    """
    path = Path(path)
    data = path.read_bytes()

    header = _parse_header(data)

    body = data[HEADER_SIZE:]
    full_frames = len(body) // BYTES_PER_INPUT
    trailing = len(body) - full_frames * BYTES_PER_INPUT
    if trailing:
        raise DtmFormatError(
            f"controller data is not a multiple of {BYTES_PER_INPUT} bytes (trailing {trailing})"
        )

    # If the header's input_count disagrees with what the file actually contains,
    # we trust the file. Some TAS recordings have been observed with slightly
    # inflated input_count values; downstream code can cross-check via
    # `len(controller_states) vs header.input_count`.
    states = [
        _parse_controller_frame(body[i * BYTES_PER_INPUT : (i + 1) * BYTES_PER_INPUT], i)
        for i in range(full_frames)
    ]

    if header.input_count != full_frames:
        log.warning(
            "parse_dtm: header.input_count=%d but body contains %d frames. "
            "Trusting the body. Small discrepancies (<1%%) are normal; large ones suggest "
            "a truncated or corrupted .dtm.",
            header.input_count,
            full_frames,
        )

    return header, states


def build_dtm_blob(
    controllers: int = 0x01,
    is_wii: int = 1,
    from_savestate: int = 0,
    vi_count: int = 0,
    input_count: int = 0,
    lag_count: int = 0,
    rerecord_count: int = 0,
    author: str = "",
    game_id: bytes = EXPECTED_GAME_ID,
    signature: bytes = EXPECTED_SIG,
    frames: list[bytes] | None = None,
) -> bytes:
    """Build a valid .dtm byte blob. Test helper; not for production use.

    Keep this in the parser module (not tests/) so it's available for
    smoke tests of downstream modules (pairing, dataset, viz) too.
    """
    header = bytearray(HEADER_SIZE)
    header[SIG_OFFSET : SIG_OFFSET + SIG_SIZE] = signature.ljust(SIG_SIZE, b"\x00")[:SIG_SIZE]
    header[GAME_ID_OFFSET : GAME_ID_OFFSET + GAME_ID_SIZE] = game_id.ljust(GAME_ID_SIZE, b"\x00")[
        :GAME_ID_SIZE
    ]
    header[IS_WII_OFFSET] = is_wii
    header[CONTROLLERS_OFFSET] = controllers
    header[FROM_SAVESTATE_OFFSET] = from_savestate
    struct.pack_into("<Q", header, VI_COUNT_OFFSET, vi_count)
    struct.pack_into("<Q", header, INPUT_COUNT_OFFSET, input_count)
    struct.pack_into("<Q", header, LAG_COUNT_OFFSET, lag_count)
    struct.pack_into("<I", header, RERECORD_OFFSET, rerecord_count)
    author_bytes = author.encode("utf-8")[:AUTHOR_SIZE]
    header[AUTHOR_OFFSET : AUTHOR_OFFSET + len(author_bytes)] = author_bytes

    body = b"".join(frames or [])
    return bytes(header) + body


def build_frame(
    *,
    accelerate: bool = False,
    brake: bool = False,
    drift: bool = False,
    item: bool = False,
    look_behind: bool = False,
    analog_x: int = 128,
    analog_y: int = 128,
) -> bytes:
    """Build one 8-byte GCN controller frame. Test helper."""
    byte0 = 0
    if accelerate:
        byte0 |= 1 << 1  # A
    if brake:
        byte0 |= 1 << 2  # B
    if look_behind:
        byte0 |= 1 << 3  # X

    byte1 = 0
    if item:
        byte1 |= 1 << 2  # L digital
    if drift:
        byte1 |= 1 << 3  # R digital
    byte1 |= 1 << 6  # controller_connected

    return bytes([byte0, byte1, 0, 0, analog_x & 0xFF, analog_y & 0xFF, 128, 128])
