"""Slave-side Dolphin scripting entry point.

Runs inside a Dolphin process via ``DolphinQt --script /path/to/dolphin_script.py``.
Receives actions from the master (``dolphin_env.py``) over a TCP socket,
applies them to the emulated controller, reads RAM into a ``RaceState``
each frame, computes our v2 reward via ``TrackRewardTracker``, and sends
``(observation, reward_breakdown, done, info)`` back.

Architecture (matches VIPTankz's DolphinScript / DolphinEnv pair):

- Shared memory for the observation frames (fastest).
- TCP socket for the action-reward control path (lowest round-trip time).
- The master owns the track curriculum; the slave loads whatever savestate
  path the master sends at reset time and uses the track_meta payload
  included in that message for reward computation.

This script is deliberately kept small — the heavy logic (reward shaping,
metadata schema, checkpoint layout) lives in ``mkw_rl.env.reward`` and
``mkw_rl.env.track_meta`` which we import from the project's ``src/`` tree
via ``sys.path.insert`` at the top. PyYAML is NOT needed on the slave side
(see ``track_meta.py`` — yaml import is lazy).

The RAM pointer chains and image preprocessing are ported verbatim from
VIPTankz's ``~/code/mkw/Wii-RL/DolphinScript.py`` (PAL RMCP01). Line
references are in comments for ease of cross-reference.

.. important::
    This file imports from ``mkw_rl.env.*`` using a hard-coded ``sys.path``
    entry to ``/Users/alex/lu/git/mkwii/src``. On Vast.ai or any machine
    where the repo lives elsewhere, edit ``PROJECT_SRC`` below (or set it
    via environment variable — TODO).
"""

# --- sys.path forwarding --------------------------------------------------
import os
import sys
from pathlib import Path

# Resolve project src/ dir. Default = the dev machine's known path. Allow
# override via MKW_RL_SRC env var for portability.
PROJECT_SRC = os.environ.get("MKW_RL_SRC", "/Users/alex/lu/git/mkwii/src")
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# --- stdlib imports -------------------------------------------------------
import time
from multiprocessing.connection import Client

import numpy as np
from PIL import Image  # noqa: F401  TODO(phase 2.1) observation capture uses this

# --- Dolphin scripting-API imports (only valid inside Dolphin) -----------
from dolphin import controller, event, memory, savestate  # type: ignore[import-not-found]

# --- our modules (via PROJECT_SRC path injection above) ------------------
from mkw_rl.env.reward import RaceState, RewardConfig, TrackRewardTracker
from mkw_rl.env.track_meta import TrackMetadata

# --- config --------------------------------------------------------------
# Matches VIPTankz's DolphinEnv defaults. Single-env for now; when we add
# multi-env we'll take env_id from CLI args or an instance-info file.
ENV_ID = int(os.environ.get("MKW_RL_ENV_ID", "0"))
MASTER_HOST = "localhost"
MASTER_PORT = 26330 + ENV_ID
SOCKET_AUTHKEY = b"mkw-rl-env-authkey"

# Observation shape per PAL BTR: 4×75×140 grayscale.
FRAME_HEIGHT = 75
FRAME_WIDTH = 140
FRAME_STACK = 4
FRAMESKIP = 4  # act every 4 emulation frames (BTR convention)

# Action-space decomposition, matching VIPTankz's DolphinScript.py:476-487.
# 40 actions = 5 (StickX) × 2 (R/drift) × 2 (Up) × 2 (L/item). A always held.
STICK_X_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
R_VALUES = [False, True]
UP_VALUES = [False, True]
L_VALUES = [False, True]
NUM_ACTIONS = len(STICK_X_VALUES) * len(R_VALUES) * len(UP_VALUES) * len(L_VALUES)
assert NUM_ACTIONS == 40


# --- RAM pointer chains (PAL RMCP01, from VIPTankz's DolphinScript.py) ---
def _resolve(base_addr: int, offsets: list[int]) -> int:
    """Follow a pointer chain. See DolphinScript.py:154-165 for the original."""
    addr = memory.read_u32(base_addr)
    for off in offsets[:-1]:
        addr = memory.read_u32(addr + off)
    return addr + offsets[-1]


class _Addresses:
    """Resolved pointers, computed once per reset (they're stable across frames).

    Line references in comments point at ``DolphinScript.py`` in VIPTankz's
    repo — don't re-derive these; treat the pointer chains as canonical.
    """

    def __init__(self) -> None:
        # DolphinScript.py:97
        self.race_completion = _resolve(0x809BD730, [0xC, 0x0, 0xC])
        # DolphinScript.py:100-102
        self.current_lap = _resolve(0x809BD730, [0xC, 0x0, 0x24])
        self.stage = _resolve(0x809BD730, [0x28])
        # DolphinScript.py:129 (surfaceFlags bit 6 = isTouchingOffroad, see :259)
        self.surface_flags = _resolve(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x18, 0x2C])
        # DolphinScript.py:119
        self.offroad_invincibility = _resolve(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x148])
        # DolphinScript.py:141
        self.race_position = _resolve(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x3C])
        # DolphinScript.py:145
        self.wall_collide = _resolve(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x8, 0x8])


def _read_race_state(addrs: _Addresses) -> RaceState:
    """Map the RAM reads into our ``RaceState`` dataclass."""
    surface = memory.read_u8(addrs.surface_flags)
    touching_offroad = bool(surface & (1 << 6))  # DolphinScript.py:259
    return RaceState(
        race_completion=memory.read_f32(addrs.race_completion),
        current_lap=memory.read_u16(addrs.current_lap),
        race_stage=memory.read_u8(addrs.stage),
        race_position=memory.read_u8(addrs.race_position),
        touching_offroad=touching_offroad,
        wall_collide=memory.read_u32(addrs.wall_collide),
        offroad_invincibility=memory.read_u16(addrs.offroad_invincibility),
    )


# --- action decoding -----------------------------------------------------
def _decode_action(action: int) -> dict:
    """Decode a Discrete(40) action index into a GCN controller-state dict.

    Layout matches VIPTankz's ``DolphinScript.py:475-490``. A is always held.
    """
    assert 0 <= action < NUM_ACTIONS, f"action {action} out of range [0,{NUM_ACTIONS})"
    stick_idx = action // 8
    rem = action % 8
    r_idx = rem // 4
    rem = rem % 4
    up_idx = rem // 2
    l_idx = rem % 2
    return {
        "Left": False,
        "Right": False,
        "Down": False,
        "Up": UP_VALUES[up_idx],
        "Z": False,
        "R": R_VALUES[r_idx],
        "L": L_VALUES[l_idx],
        "A": True,
        "B": False,
        "X": False,
        "Y": False,
        "Start": False,
        "StickX": STICK_X_VALUES[stick_idx],
        "StickY": 0,
        "CStickX": 0,
        "CStickY": 0,
        "TriggerLeft": 0,
        "TriggerRight": 0,
        "AnalogA": 0,
        "AnalogB": 0,
        "Connected": True,
    }


# --- frame preprocessing -------------------------------------------------
def _process_frame(rgb_bytes: bytes, width: int, height: int) -> np.ndarray:
    """RGB bytes → (75, 140) uint8 grayscale. Ported from DolphinScript.py."""
    img = Image.frombytes("RGB", (width, height), rgb_bytes, "raw")
    img = img.convert("L")  # grayscale
    img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# --- IPC helpers ---------------------------------------------------------
def _connect_to_master() -> "Client":
    """Wait until the master's Listener is up, then connect."""
    delay = 0.1
    for _ in range(50):  # ~5s total
        try:
            return Client((MASTER_HOST, MASTER_PORT), authkey=SOCKET_AUTHKEY)
        except ConnectionRefusedError:
            time.sleep(delay)
    raise RuntimeError(f"[slave {ENV_ID}] could not connect to master on :{MASTER_PORT}")


def _send_init(conn: "Client", obs: np.ndarray) -> None:
    """One-time handshake: slave sends the initial observation to master."""
    conn.send(("init", obs.tobytes()))
    reply = conn.recv()
    if reply != "ack":
        raise RuntimeError(f"[slave {ENV_ID}] bad init handshake: {reply!r}")


def _recv_reset(conn: "Client") -> tuple[str, dict]:
    """Wait for a reset command from master. Returns (savestate_path, track_meta_dict)."""
    msg = conn.recv()
    kind = msg[0]
    if kind != "reset":
        raise RuntimeError(f"[slave {ENV_ID}] expected 'reset', got {kind!r}")
    savestate_path, track_meta_dict = msg[1], msg[2]
    return savestate_path, track_meta_dict


def _recv_action(conn: "Client") -> int | str:
    """Wait for either an int action or a control string ('reset', 'close')."""
    return conn.recv()


# --- main loop -----------------------------------------------------------
def main() -> None:
    print(f"[slave {ENV_ID}] starting", flush=True)
    conn = _connect_to_master()
    print(f"[slave {ENV_ID}] connected to master", flush=True)

    # TODO(phase 2.1): proper boot-wait — we need to let the game fully load
    # before RAM reads are valid. The previous naive `event.on_frameadvance`
    # sentinel loop was a no-op (referenced the attribute without calling).
    # Correct mechanism is a frame-counter inside on_frame, gated until
    # boot_frames_seen >= N before allowing reset to proceed.

    addrs: _Addresses | None = None
    tracker: TrackRewardTracker | None = None
    # TODO(phase 2.1): frame_stack will hold the rolling 4-frame grayscale
    # observation once the sync frame-advance + event.framedrawn pattern is
    # wired. Unused today — see observation-capture TODO in step loop below.
    frame_stack: list[np.ndarray] = []
    last_action: int = 0
    pending_reset: tuple[str, dict] | None = None

    def on_frame() -> None:
        nonlocal addrs, tracker, frame_stack, last_action, pending_reset

        if pending_reset is not None:
            savestate_path, track_meta_dict = pending_reset
            savestate.load_from_file(savestate_path)
            # Rebuild addresses and tracker for the new race.
            addrs = _Addresses()
            meta = TrackMetadata(**track_meta_dict)
            tracker = TrackRewardTracker(track_meta=meta, config=RewardConfig())
            state = _read_race_state(addrs)
            tracker.align_to_state(state)
            frame_stack = []
            pending_reset = None

        if addrs is None or tracker is None:
            return

        # Apply the last-received action every frame (VIPTankz:545-548 uses
        # a keep-pressing pattern to avoid stutter).
        controller.set_gc_buttons(0, _decode_action(last_action))

    event.on_frameadvance(on_frame)

    # Step loop: alternate waiting for master commands with frame advances.
    # The on_frameadvance callback applies the action; the main loop handles
    # observation/reward/message passing.
    while True:
        msg = _recv_action(conn)
        if msg == "close":
            print(f"[slave {ENV_ID}] received close, exiting main loop", flush=True)
            break
        if isinstance(msg, tuple) and msg[0] == "reset":
            pending_reset = (msg[1], msg[2])
            # TODO(phase 2.1): synchronize via frame count to guarantee the
            # load lands before we send the first observation. Currently
            # relies on the next step() to observe post-load state.
            continue

        # Must be an int action.
        assert isinstance(msg, int), f"expected int action, got {msg!r}"
        last_action = msg

        # Advance FRAMESKIP frames. The on_frame callback keeps re-applying
        # last_action each frame; we compute reward once at the end.
        # (VIPTankz does frame pooling over the last 2 frames; we do the
        # same for now — see DolphinScript.py:556-559.)
        # TODO(phase 2.1): implement the frameskip loop here once we
        # confirm the scripting API's frame-advance primitive for sync
        # waits inside the main loop. For now, this is a skeleton that
        # needs a live Dolphin integration test to finalize.

        # Read state and compute reward.
        if addrs is None or tracker is None:
            # Not yet reset — shouldn't happen in normal flow.
            conn.send(("step", None, 0.0, False, {}))
            continue

        state = _read_race_state(addrs)
        breakdown, done = tracker.step(state)

        # TODO(phase 2.1): gather a real observation here. VIPTankz uses
        # ``event.framedrawn()`` in async context; we need the sync
        # equivalent or we switch this whole script to async. For now,
        # emit a zero observation so the wire format is stable and the
        # master can process.
        obs = np.zeros((FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

        info = {
            "reward_breakdown": breakdown.as_dict(),
            "race_completion": state.race_completion,
            "lap": state.current_lap,
            "stage": state.race_stage,
            "position": state.race_position,
        }
        conn.send(("step", obs.tobytes(), float(breakdown.total), bool(done), info))

    conn.close()


if __name__ == "__main__":
    main()
