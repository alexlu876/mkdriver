#!/usr/bin/env python3
"""Replay a .dtm with frame dumping enabled.

This script is the second half of the record-then-replay pipeline
(MKW_RL_SPEC.md §1.2). Recording and dumping must not happen
simultaneously — dumping slows the emulator below realtime, and live
recordings at reduced speed produce inputs that don't match
clean-speed replay.

## How it works

Two-process setup:

1. **Launcher** (this script, ran under uv): validates args, clears
   Dolphin's dump directory, writes out a Dolphin-side driver script,
   and launches Dolphin via ``dolphin-emu --script <driver> <iso>``.

2. **Driver** (generated ``_replay_driver.py``): runs inside Dolphin's
   embedded Python. Uses the VIPTankz scripting API
   (``from dolphin import event, savestate, controller``) to load the
   savestate and inject controller inputs parsed directly from the .dtm
   bytes on each frame. Inline-parses the .dtm — does not import our
   mkw_rl package because Dolphin's Python may not have it installed.

## Status

This is verified against VIPTankz/Wii-RL @ d8358cb... via reading their
DolphinScript.py. Not end-to-end tested until P-1 confirms the
scripting API works. If Dolphin rejects any of the calls, see
scripts/REPLAY_PROTOCOL.md for the manual fallback (using Dolphin's
GUI `Movie → Play Input`).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Driver template: written next to replay_demo.py, passed to Dolphin
# via --script. Runs inside Dolphin's Python. Inline-parses the .dtm
# (stdlib only) so Dolphin's Python doesn't need our package on PYTHONPATH.
_DRIVER_TEMPLATE = '''\
"""Dolphin-side replay driver — runs inside dolphin-emu --script.

Parses a .dtm file (inline, stdlib only), loads the savestate,
and injects one input frame per rendered frame via controller.set_gc_buttons.

Button mapping per MKW_RL_SPEC.md §1.1:
    byte 0 bit 1 = A          (accelerate)
    byte 0 bit 2 = B          (brake)
    byte 0 bit 3 = X          (look behind)
    byte 1 bit 2 = L digital  (item)
    byte 1 bit 3 = R digital  (drift)
    byte 4       = analog X   (steering, 0..255 w/ 128 neutral → [-1, 1] float)
"""

from dolphin import event, savestate, controller

# Injected by the launcher.
DTM_PATH = r"{dtm_path}"
SAVESTATE_PATH = r"{savestate_path}"


_HEADER_SIZE = 0x100
_BYTES_PER_INPUT = 8
_EXPECTED_SIG = b"DTM\\x1a"
_EXPECTED_GAME_ID = b"RMCE01"


def _parse_frames(dtm_path):
    """Inline .dtm parser (subset). Returns list of button dicts ready for set_gc_buttons.

    Deliberately stdlib-only so Dolphin's embedded Python needn't have our
    mkw_rl package on its path. Mirrors src/mkw_rl/dtm/parser.py but trimmed.
    """
    with open(dtm_path, "rb") as f:
        data = f.read()

    if len(data) < _HEADER_SIZE:
        raise RuntimeError("dtm header truncated")
    if data[:4] != _EXPECTED_SIG:
        raise RuntimeError("bad dtm signature")
    if data[4:10] != _EXPECTED_GAME_ID:
        raise RuntimeError("dtm is not NTSC-U RMCE01")
    # Reject multi-controller (spec §1.1; parser.py B-2 fix).
    controllers_bitfield = data[0x0B]
    if controllers_bitfield != 0x01:
        raise RuntimeError(
            "multi-controller .dtm not supported by this pipeline; "
            "controllers_bitfield=0x%02x" % controllers_bitfield
        )

    body = data[_HEADER_SIZE:]
    n = len(body) // _BYTES_PER_INPUT
    frames = []
    for i in range(n):
        off = i * _BYTES_PER_INPUT
        byte0 = body[off]
        byte1 = body[off + 1]
        analog_x = body[off + 4]
        # Raw [0..255] → float [-1..1], 128 = neutral, 127 is max magnitude.
        stick_x = max(-1.0, min(1.0, (analog_x - 128) / 127.0))
        frames.append({
            "Left": False, "Right": False, "Down": False, "Up": False,
            "Z": False,
            "R": bool(byte1 & (1 << 3)),    # R digital = drift
            "L": bool(byte1 & (1 << 2)),    # L digital = item
            "A": bool(byte0 & (1 << 1)),    # A = accelerate
            "B": bool(byte0 & (1 << 2)),    # B = brake
            "X": bool(byte0 & (1 << 3)),    # X = look behind
            "Y": False,
            "Start": False,                 # Start intentionally never pressed
            "StickX": stick_x,
            "StickY": 0,
            "CStickX": 0,
            "CStickY": 0,
            "TriggerLeft": 0,
            "TriggerRight": 0,
            "AnalogA": 0,
            "AnalogB": 0,
            "Connected": True,
        })
    return frames


def main():
    print("[replay_driver] parsing " + DTM_PATH)
    frames = _parse_frames(DTM_PATH)
    print("[replay_driver] parsed %d input frames" % len(frames))

    print("[replay_driver] loading savestate " + SAVESTATE_PATH)
    savestate.load_from_file(SAVESTATE_PATH)

    # Wait one frame so savestate is fully applied before we start injecting.
    event.on_frameadvance()

    for i, dic in enumerate(frames):
        controller.set_gc_buttons(0, dic)
        event.on_frameadvance()
        if i % 600 == 0:
            print("[replay_driver] frame %d / %d" % (i, len(frames)))

    print("[replay_driver] replay complete, %d frames injected" % len(frames))


main()
'''


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dolphin",
        required=True,
        type=Path,
        help="Path to the VIPTankz dolphin-emu binary.",
    )
    p.add_argument("--dtm", required=True, type=Path, help="Path to the .dtm to replay.")
    p.add_argument(
        "--savestate",
        required=True,
        type=Path,
        help="Path to the matching savestate (.sav).",
    )
    p.add_argument(
        "--iso",
        required=True,
        type=Path,
        help="Path to the NTSC-U MKWii ISO (RMCE01).",
    )
    p.add_argument(
        "--frames-out",
        required=True,
        type=Path,
        help="Where to move/copy the resulting frame dump after Dolphin exits.",
    )
    p.add_argument(
        "--dolphin-dump-dir",
        default=Path.home() / "Library" / "Application Support" / "Dolphin" / "Dump" / "Frames",
        type=Path,
        help="Dolphin's frame dump source directory (default: macOS standard).",
    )
    args = p.parse_args()

    # Validate inputs up front so Dolphin isn't spawned for nothing.
    for label, path in (
        ("dolphin binary", args.dolphin),
        (".dtm", args.dtm),
        ("savestate", args.savestate),
        ("ISO", args.iso),
    ):
        if not path.exists():
            print(f"error: {label} not found at {path}", file=sys.stderr)
            return 1

    # Clear any previous dump so we know what we produced this run.
    # Recursive — Dolphin writes to a game-ID subdirectory on some versions.
    if args.dolphin_dump_dir.exists():
        for p_png in args.dolphin_dump_dir.rglob("*.png"):
            try:
                p_png.unlink()
            except OSError:
                pass

    # Write the driver out next to this script.
    driver_path = Path(__file__).parent / "_replay_driver.py"
    driver_path.write_text(
        _DRIVER_TEMPLATE.format(
            dtm_path=str(args.dtm.resolve()).replace("\\", "\\\\"),
            savestate_path=str(args.savestate.resolve()).replace("\\", "\\\\"),
        )
    )

    # Invoke Dolphin. VIPTankz's pattern is `--script <path> <iso>` — see
    # third_party/Wii-RL/DolphinEnv.py:202.
    cmd = [
        str(args.dolphin),
        "--script",
        str(driver_path),
        str(args.iso.resolve()),
    ]
    print(f"[replay_demo] invoking: {' '.join(cmd)}")
    print(
        "[replay_demo] ensure Dolphin's graphics config has 'Dump Frames' + 'Dump Frames as Images' enabled."
    )
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[replay_demo] Dolphin exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    # Move the dumped PNGs to the output directory.
    args.frames_out.mkdir(parents=True, exist_ok=True)
    # Recursive because Dolphin may write to Dump/Frames/RMCE01/.
    pngs = sorted(args.dolphin_dump_dir.rglob("*.png"))
    if not pngs:
        print(
            "[replay_demo] warning: no PNGs produced — "
            "check Dolphin's Dump Frames config and the driver log above",
            file=sys.stderr,
        )
        return 2
    for p_png in pngs:
        shutil.move(str(p_png), args.frames_out / p_png.name)
    print(f"[replay_demo] moved {len(pngs)} PNGs to {args.frames_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
