"""Semi-automated savestate recorder — multi-capture per track.

Captures ``N_CAPTURES_PER_TRACK`` savestates per track at evenly-spaced
``race_completion`` thresholds, giving the agent variety in starting
positions during training. The user drives Dolphin's menus manually
and races each track once (~2 laps); this script listens for each
threshold crossing and saves with auto-numbered filenames.

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
4. When the countdown finishes (``race_completion`` crosses 1.0),
   the script pops the top slug from the queue, captures the first
   savestate as ``{slug}_001.sav``, and arms the remaining 11
   thresholds.
5. Continue driving. As ``race_completion`` crosses each subsequent
   threshold (every ~1/6 of a lap, covering ~1.83 laps total), the
   script saves ``{slug}_002.sav`` through ``{slug}_012.sav``.
6. Once all 12 are captured (or you finish the race), exit via the
   pause menu. The slug is fully consumed.
7. If you crash / exit early, you'll have fewer than 12 savestates
   for that track. The slug is still consumed; if you want more,
   add it back to the queue and re-run.
8. Repeat from step 3 for the next track.

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

# --- multi-capture parameters ---------------------------------------------
# 12 captures per track, spaced every 1/6 of a lap (race_completion 1.0 to
# 2.833). Covers ~1.83 laps with the agent at a different position each time
# — gives training data variety so the policy isn't only seeing "post-
# countdown" starts.
N_CAPTURES_PER_TRACK = 12
CAPTURE_THRESHOLDS = [1.0 + i * (1.0 / 6) for i in range(N_CAPTURES_PER_TRACK)]
# = [1.0000, 1.1667, 1.3333, 1.5000, 1.6667, 1.8333, 2.0000, 2.1667,
#    2.3333, 2.5000, 2.6667, 2.8333]

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
# ``prev_race_com`` starts at None so the first frame seeds it without spuriously
# firing the race-start edge detector — if a race is already running when the
# script attaches (rare but possible), the prior ``0.0`` sentinel would make
# ``prev < 1.0 <= race_com`` true on the very first tick and capture something
# we didn't intend. The None sentinel forces a one-frame warmup.
#
# Per-track lifecycle:
#   active_slug=None → looking for race-start edge to pop next slug
#   active_slug=set, next_threshold_idx<12 → in race, capturing on threshold crosses
#   reaches threshold_idx==12 → all 12 captured for this track, reset to active_slug=None
#   race exits early (race_com < 1.0) → log partial-capture warning, reset
_state: dict = {
    "prev_race_com": None,
    "active_slug": None,
    "next_threshold_idx": 0,
}

print(f"[recorder] ready — {N_CAPTURES_PER_TRACK} savestates per track at thresholds:", flush=True)
print(f"[recorder]   {[f'{t:.3f}' for t in CAPTURE_THRESHOLDS]}", flush=True)
print(f"[recorder] queue: {QUEUE_PATH}", flush=True)
print(f"[recorder] output: {OUTPUT_DIR}", flush=True)
if not QUEUE_PATH.exists():
    print(
        "[recorder] WARNING: queue file does not exist yet — create it before starting a race",
        flush=True,
    )


def _push_front(slug: str) -> None:
    """Put ``slug`` back at the top of the queue file (for save-failure rollback)."""
    existing = ""
    if QUEUE_PATH.exists():
        existing = QUEUE_PATH.read_text()
    QUEUE_PATH.write_text(f"{slug}\n{existing}" if existing else f"{slug}\n")


def _save_capture(slug: str, idx: int, race_com: float) -> bool:
    """Save savestate as ``{slug}_{idx+1:03d}.sav``. Returns True on success."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{slug}_{idx + 1:03d}.sav"
    try:
        savestate.save_to_file(str(out_path))
        print(
            f"[recorder] SAVED {slug} [{idx + 1}/{N_CAPTURES_PER_TRACK}] → "
            f"{out_path.name} (race_completion={race_com:.4f})",
            flush=True,
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(
            f"[recorder] save failed for {slug} idx={idx + 1} "
            f"({type(e).__name__}: {e})",
            flush=True,
        )
        return False


def on_frame() -> None:
    s = _state
    try:
        race_com = _read_race_completion()
    except Exception:
        # Race manager pointer invalid (in menus). Skip this frame.
        return

    prev = s["prev_race_com"]
    s["prev_race_com"] = race_com

    # First-frame warmup: seed prev_race_com and skip edge detection so we
    # don't spuriously fire if the script attaches during a running race.
    if prev is None:
        return

    # Edge: race_completion crossed 1.0 upward → race just started.
    # Pop next slug only when we're not already in a track.
    if prev < 1.0 <= race_com and s["active_slug"] is None:
        slug = _pop_next_slug()
        if slug is None:
            print(
                "[recorder] race started but queue is empty — nothing to save. "
                "Add a slug to the queue file and restart the race.",
                flush=True,
            )
            return
        s["active_slug"] = slug
        s["next_threshold_idx"] = 0
        print(f"[recorder] starting capture for {slug}", flush=True)

    # In an active race: check if we've crossed any pending capture threshold.
    if s["active_slug"] is not None and s["next_threshold_idx"] < N_CAPTURES_PER_TRACK:
        idx = s["next_threshold_idx"]
        threshold = CAPTURE_THRESHOLDS[idx]
        if prev < threshold <= race_com:
            ok = _save_capture(s["active_slug"], idx, race_com)
            if ok:
                s["next_threshold_idx"] += 1
                # All N captured? Wrap up this track.
                if s["next_threshold_idx"] >= N_CAPTURES_PER_TRACK:
                    print(
                        f"[recorder] DONE with {s['active_slug']} — all "
                        f"{N_CAPTURES_PER_TRACK} captures saved. Exit race "
                        "and select next track.",
                        flush=True,
                    )
                    s["active_slug"] = None
                    s["next_threshold_idx"] = 0
            # If save failed, leave next_threshold_idx unchanged so we can retry
            # on next frame if race_com keeps advancing past the threshold.

    # Edge: race_completion dropped back below 1.0 → user exited.
    if prev >= 1.0 > race_com:
        if s["active_slug"] is not None and s["next_threshold_idx"] < N_CAPTURES_PER_TRACK:
            print(
                f"[recorder] race exited early for {s['active_slug']} "
                f"({s['next_threshold_idx']}/{N_CAPTURES_PER_TRACK} captured). "
                "Re-add to queue if you want the rest.",
                flush=True,
            )
        # Reset for next race; slug is fully consumed regardless of completion.
        s["active_slug"] = None
        s["next_threshold_idx"] = 0


event.on_frameadvance(on_frame)
