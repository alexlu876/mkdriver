#!/usr/bin/env python3
"""Replay a .dtm with frame dumping enabled.

This script is the second half of the record-then-replay pipeline
(MKW_RL_SPEC.md §1.2). Recording and dumping must not happen
simultaneously — dumping slows the emulator below realtime, and live
recordings at reduced speed produce inputs that don't match
clean-speed replay.

## Status

This is a **skeleton / launcher** that invokes the VIPTankz Dolphin
scripting fork with an in-process replay driver. The driver script
(``_replay_driver.py``, produced alongside this launcher) runs inside
Dolphin and uses ``from dolphin import event, savestate, controller``.

**Unverified in CI**: this script can't be tested from the main Python
environment because it requires a running Dolphin binary and a real
savestate. Expect to iterate on it against real hardware.

See ``scripts/REPLAY_PROTOCOL.md`` for the manual fallback if automation
isn't working yet.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Driver template: this file is written out next to replay_demo.py and
# passed to Dolphin's --script flag. It runs inside the Dolphin process.
_DRIVER_TEMPLATE = '''\
"""Dolphin-side replay driver. Invoked via `dolphin-emu --script`.

Reads a .dtm, feeds frames to the emulator via the controller module,
and exits when playback completes. Frame dumping must be enabled in
the Dolphin config that loads this script.
"""

from dolphin import event, savestate, controller
import sys

# Injected by the launcher.
DTM_PATH = "{dtm_path}"
SAVESTATE_PATH = "{savestate_path}"


def main():
    # Load the anchor savestate.
    savestate.load_from_file(SAVESTATE_PATH)
    event.on_frameadvance()

    # TODO(verify): the exact API for playing back a .dtm within the
    # scripting fork is not fully documented at the pinned SHA. The
    # two known options are:
    #   1. dolphin.movie.play(DTM_PATH) — if the fork exposes a movie
    #      module in its Python API.
    #   2. Manually parse the .dtm and inject each frame's inputs via
    #      controller.set_gcpad_status(...).
    # Option 2 is more work but doesn't depend on a module that may not
    # exist. Start with option 1 and fall back to option 2.

    try:
        from dolphin import movie  # type: ignore
        movie.play(DTM_PATH)
        while movie.is_playing():
            event.on_frameadvance()
    except ImportError:
        print("[replay_driver] dolphin.movie not available — manual injection path TODO")
        sys.exit(2)

    print("[replay_driver] replay complete")


main()
'''


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dolphin", required=True, type=Path, help="Path to the VIPTankz dolphin-emu binary.")
    p.add_argument("--dtm", required=True, type=Path, help="Path to the .dtm to replay.")
    p.add_argument(
        "--savestate",
        required=True,
        type=Path,
        help="Path to the matching savestate (.sav).",
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

    if not args.dolphin.exists():
        print(f"error: dolphin binary not found at {args.dolphin}", file=sys.stderr)
        return 1
    if not args.dtm.exists():
        print(f"error: .dtm not found at {args.dtm}", file=sys.stderr)
        return 1
    if not args.savestate.exists():
        print(f"error: savestate not found at {args.savestate}", file=sys.stderr)
        return 1

    # Clear any previous dump so we know what we produced this run.
    if args.dolphin_dump_dir.exists():
        for p_png in args.dolphin_dump_dir.glob("*.png"):
            p_png.unlink()

    # Write the driver out next to this script.
    driver_path = Path(__file__).parent / "_replay_driver.py"
    driver_path.write_text(
        _DRIVER_TEMPLATE.format(
            dtm_path=str(args.dtm.resolve()).replace("\\", "\\\\"),
            savestate_path=str(args.savestate.resolve()).replace("\\", "\\\\"),
        )
    )

    # Launch Dolphin with the driver.
    # TODO(verify): the VIPTankz fork's CLI for loading a script may be
    # --script <path> or --exec <path> — check their README. Adjust
    # here if different.
    cmd = [str(args.dolphin), "--script", str(driver_path)]
    print(f"[replay_demo] invoking: {' '.join(cmd)}")
    print("[replay_demo] ensure Dolphin config has 'Dump Frames' enabled before running this.")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[replay_demo] Dolphin exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    # Move the dumped PNGs to the output directory.
    args.frames_out.mkdir(parents=True, exist_ok=True)
    pngs = sorted(args.dolphin_dump_dir.glob("*.png"))
    if not pngs:
        print("[replay_demo] warning: no PNGs produced — check Dolphin's Dump Frames config")
        return 2
    for p_png in pngs:
        shutil.move(str(p_png), args.frames_out / p_png.name)
    print(f"[replay_demo] moved {len(pngs)} PNGs to {args.frames_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
