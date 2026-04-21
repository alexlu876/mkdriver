"""Semi-automated savestate recorder.

Assists the user in capturing one savestate per track at the exact
first-input-applies frame (race_completion crosses 1.0). The user
drives Dolphin's menus manually; this script just listens for the
"race started" edge and saves the state with a name taken from a
queue file.

Why semi-auto instead of fully automatic
----------------------------------------
Full menu automation (cup → track → character → vehicle → ghost
skip → start) is fragile — MKWii's TT flow has non-deterministic
dialogs (first-time-on-track messages, ghost prompts, etc.) that
break brittle input sequences. Detecting race start from memory is
rock-solid. Total human overhead stays small: ~30 sec of menu
clicks per track (you'd be doing that anyway) + no frame-stepping
or save-state-hotkey timing.

Workflow per track
------------------
1. Ensure ``data/savestates/tracks_to_record.txt`` has the upcoming
   track's slug on the first non-blank line. Prepare the whole
   queue up front (one slug per line, in the order you plan to
   record) to avoid editing between tracks::

        luigi_circuit_tt
        moo_moo_meadows_tt
        mushroom_gorge_tt
        ...

2. Launch Dolphin with this script attached (see "Running" below).
   The script logs "[recorder] ready …" and then waits.
3. Manually navigate: ``Single Player → Time Trial → (cup) → (track)
   → character → vehicle → start``. Choose whatever character/
   vehicle you want — the script doesn't care, but use consistent
   choices across tracks so training sees a uniform agent kart.
4. When the countdown finishes and the kart starts accepting
   inputs (``race_completion`` crosses 1.0), the script pops the
   top slug from the queue and writes ``data/savestates/{slug}.sav``.
5. Exit the race via Dolphin's pause menu (Start button or Esc →
   "Exit to menu"). You'll be back at the TT track-select screen.
6. Repeat from step 3 for the next track. The script is still
   running; it'll fire again on the next race start.

Running
-------
::

    ~/code/mkw/Wii-RL/dolphin0/DolphinQt.app/Contents/MacOS/DolphinQt \\
        --no-python-subinterpreters \\
        --script /Users/alex/lu/git/mkwii/scripts/record_savestates.py \\
        --exec=$HOME/code/mkw/Wii-RL/game/mkw.iso

Safety notes
------------
- The script **overwrites** ``data/savestates/{slug}.sav`` without
  asking. If you want to preserve a previous recording for a track,
  rename it first.
- If the queue file is empty when a race starts, the script logs a
  warning and does NOT save anything.
- The queue file is rewritten each time a slug is consumed. If
  Dolphin crashes mid-session you can open the queue file, see
  what's left, and resume without re-doing completed tracks.
- Race-start detection uses ``race_completion`` crossing 1.0. If a
  savestate somehow loads you mid-race, that detection can't
  distinguish "fresh race just started" from "loaded a mid-race
  savestate"; plan sessions to start from fresh races each time.

The PAL RMCP01 race-manager pointer chain is copied verbatim from
``~/code/mkw/Wii-RL/DolphinScript.py``.
"""

from pathlib import Path

from dolphin import event, memory, savestate

# --- configurable paths ---------------------------------------------------
OUTPUT_DIR = Path("/Users/alex/lu/git/mkwii/data/savestates")
QUEUE_PATH = OUTPUT_DIR / "tracks_to_record.txt"

# --- race manager RAM layout (PAL RMCP01) ---------------------------------
RACE_MGR_BASE = 0x809BD730
RACE_COMPLETION_OFFSETS = [0xC, 0x0, 0xC]


def _resolve(base_addr: int, offsets: list[int]) -> int:
    addr = memory.read_u32(base_addr)
    for off in offsets[:-1]:
        addr = memory.read_u32(addr + off)
    return addr + offsets[-1]


def _read_race_completion() -> float:
    return memory.read_f32(_resolve(RACE_MGR_BASE, RACE_COMPLETION_OFFSETS))


def _pop_next_slug() -> str | None:
    """Consume and return the first non-blank line from the queue file.

    Rewrites the queue file with the remaining lines. Returns None if
    the queue is missing or empty.
    """
    if not QUEUE_PATH.exists():
        return None
    lines = [line.strip() for line in QUEUE_PATH.read_text().splitlines()]
    non_empty = [line for line in lines if line and not line.startswith("#")]
    if not non_empty:
        return None
    consumed = non_empty[0]
    # Remove only the first occurrence from the original file, preserving
    # comments and blank lines in case the user has annotated the queue.
    out_lines: list[str] = []
    already_popped = False
    for line in lines:
        if not already_popped and line.strip() == consumed:
            already_popped = True
            continue
        out_lines.append(line)
    QUEUE_PATH.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return consumed


# State carried across frames.
_state = {
    "prev_race_com": 0.0,
    "captured_current_race": False,
}

print("[recorder] ready — drive Dolphin's menus; script will auto-save on race start", flush=True)
print(f"[recorder] queue: {QUEUE_PATH}", flush=True)
print(f"[recorder] output: {OUTPUT_DIR}", flush=True)
if not QUEUE_PATH.exists():
    print(
        "[recorder] WARNING: queue file does not exist yet — create it before starting a race",
        flush=True,
    )


def on_frame() -> None:
    s = _state
    try:
        race_com = _read_race_completion()
    except Exception:
        # Race manager pointer invalid (in menus). Skip this frame.
        return

    prev = s["prev_race_com"]
    s["prev_race_com"] = race_com

    # Edge: race_completion crossed 1.0 upward → race just started.
    if prev < 1.0 <= race_com and not s["captured_current_race"]:
        slug = _pop_next_slug()
        if slug is None:
            print(
                "[recorder] race started but queue is empty — nothing to save. "
                "Add a slug to the queue file and restart the race.",
                flush=True,
            )
        else:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / f"{slug}.sav"
            try:
                savestate.save_to_file(str(out_path))
                print(
                    f"[recorder] SAVED {slug} → {out_path.name} "
                    f"(race_completion={race_com:.4f})",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[recorder] save failed for {slug} ({type(e).__name__}: {e}) — "
                    "slug consumed from queue; add it back manually if you want to retry",
                    flush=True,
                )
        s["captured_current_race"] = True

    # Edge: race_completion dropped back below 1.0 → user exited. Arm for next.
    if prev >= 1.0 > race_com:
        s["captured_current_race"] = False


event.on_frameadvance(on_frame)
