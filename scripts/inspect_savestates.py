"""Savestate inspector — load each savestate and print game-state readouts.

Runs inside Dolphin via the scripting API. Iterates through every
``RMCP01.s*`` file under ``~/code/mkw/Wii-RL/MarioKartSaveStates/``,
loads each, waits for the emulator state to stabilize, and prints the
race-manager values (RaceCompletion, current lap, race stage) that
VIPTankz's ``DolphinScript.py`` reads.

Why it's useful
---------------
VIPTankz's shipped savestate zip has files named ``RMCP01.s01`` …
``RMCP01.s08`` with no hint in the filename about which track each
one is. This script lets you audit the contents without clicking
through Dolphin's GUI. Typical uses:

- Confirm all 8 shipped savestates are the same track (per
  2026-04-21 inspection: yes, all Luigi Circuit) and note the
  race_completion spread.
- After recording new savestates, verify they landed at sensible
  anchor points (race_completion ≈ 1.0, lap == 1, stage == 2 or 3).
- Debug when a savestate mysteriously refuses to load or loads
  into a weird state.

Limitation
----------
Does NOT identify the track by name — we'd need an MKWii course-ID
RAM address for that, which isn't in VIPTankz's DolphinScript.py.
For now, use the race_completion + lap diagnostics to spot obvious
outliers; fall back to loading the savestate in Dolphin's GUI and
reading the HUD when you need the exact track.

Running
-------
With the uv venv active (or any env where VIPTankz's requirements
are installed; this script itself uses only the ``dolphin`` scripting
module which Dolphin provides)::

    ~/code/mkw/Wii-RL/dolphin0/DolphinQt.app/Contents/MacOS/DolphinQt \\
        --no-python-subinterpreters \\
        --script scripts/inspect_savestates.py \\
        --exec=$HOME/code/mkw/Wii-RL/game/mkw.iso

Expected output (one line per savestate, once the game has fully
booted and started iterating)::

    [inspector] found 8 savestate files
    [inspector] loading RMCP01.s01
    [inspector]   RMCP01.s01: race_completion=1.0124, lap=1, stage=3
    [inspector] loading RMCP01.s02
    [inspector]   RMCP01.s02: race_completion=1.7832, lap=1, stage=3
    ...
    [inspector] all savestates inspected; close Dolphin to exit.

Close Dolphin manually when the "all savestates inspected" line
appears — the script intentionally doesn't call exit/quit because
the scripting API's lifecycle handling is brittle.

The race-manager pointer chains are copied verbatim from
``~/code/mkw/Wii-RL/DolphinScript.py`` (PAL RMCP01 addresses).
"""

from pathlib import Path

from dolphin import event, memory, savestate

SAVESTATE_DIR = Path.home() / "code" / "mkw" / "Wii-RL" / "MarioKartSaveStates"
SETTLE_FRAMES = 30  # frames to wait after a load before reading state


def _resolve(base_addr: int, offsets: list[int]) -> int:
    """Follow a pointer chain anchored at ``base_addr`` with the given offsets.

    Matches the layout VIPTankz uses: dereference once at base, then add
    each offset, dereferencing again except for the final one. Returns the
    final effective address that the caller can ``read_*`` from.
    """
    addr = memory.read_u32(base_addr)
    for off in offsets[:-1]:
        addr = memory.read_u32(addr + off)
    return addr + offsets[-1]


def _read_race_state() -> tuple[float, int, int]:
    """Return (race_completion, lap, stage) from VIPTankz's RAM layout."""
    race_com_ptr = _resolve(0x809BD730, [0xC, 0x0, 0xC])
    lap_ptr = _resolve(0x809BD730, [0xC, 0x0, 0x24])
    stage_ptr = _resolve(0x809BD730, [0x28])
    return (
        memory.read_f32(race_com_ptr),
        memory.read_u16(lap_ptr),
        memory.read_u8(stage_ptr),
    )


_files = sorted(SAVESTATE_DIR.glob("RMCP01.s*"))
print(f"[inspector] found {len(_files)} savestate files", flush=True)

# Mutable state carried across frames by closure.
_state = {"idx": 0, "wait_frames": 0, "done_reported": False}


def on_frame() -> None:
    s = _state

    if s["idx"] >= len(_files):
        if not s["done_reported"]:
            print("[inspector] all savestates inspected; close Dolphin to exit.", flush=True)
            s["done_reported"] = True
        return

    if s["wait_frames"] == 0:
        # Start inspecting the next savestate.
        path = _files[s["idx"]]
        print(f"[inspector] loading {path.name}", flush=True)
        savestate.load_from_file(str(path))
        s["wait_frames"] = SETTLE_FRAMES
        return

    s["wait_frames"] -= 1
    if s["wait_frames"] == 0:
        try:
            race_com, lap, stage = _read_race_state()
            print(
                f"[inspector]   {_files[s['idx']].name}: "
                f"race_completion={race_com:.4f}, lap={lap}, stage={stage}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — scripting API failure modes are ad-hoc
            print(
                f"[inspector]   {_files[s['idx']].name}: RAM read failed ({type(e).__name__}: {e})",
                flush=True,
            )
        s["idx"] += 1


event.on_frameadvance(on_frame)
