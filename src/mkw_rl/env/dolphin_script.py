"""Slave-side Dolphin scripting entry point (async top-level).

Runs inside a Dolphin process via ``DolphinQt --script /path/to/dolphin_script.py``.
Uses the async top-level pattern Dolphin's scripting engine supports
(``await event.frameadvance()``, ``await event.framedrawn()``) — matches
VIPTankz's ``DolphinScript.py`` architecture, which is the only proven
pattern for the script + socket IO combination.

Flow (per VIPTankz's DolphinScript.py:555-613):

1. Top-level boot: wait ~4 frames for Dolphin to settle, connect socket to
   master, wait ~8 more frames for game boot.
2. Send initial frame-stack to master (handshake).
3. Enter main loop:
   - ``env.receive_action()`` — sync ``conn.recv()`` for next action OR
     control message (reset/close). Blocks the Python main thread until
     the master sends something.
   - For each of ``FRAMESKIP`` emulation frames: if it's one of the last
     ``FRAMES_POOLED``, ``await event.framedrawn()`` to capture image;
     otherwise ``await event.frameadvance()``. Call ``tracker.step()``
     every frame, accumulate reward, break early on terminated/truncated.
   - Pool the last two captured frames via per-pixel max (Atari trick).
   - Roll the 4-frame stack and insert the pooled frame.
   - Send ``(step, obs_bytes, total_reward, terminated, truncated, info)``.
4. On reset: reload the savestate, wait a few frames, rebuild the tracker,
   align to race_completion, refill the frame stack, reply to master.

The ``on_frameadvance(callback)`` pattern is used ONLY to re-apply the
current action every frame, which avoids stutter on frames when the
main loop is blocked waiting for the master's next command
(VIPTankz:545-550).

.. important::
    Imports from our ``mkw_rl.env.*`` package rely on a hard-coded or
    env-var-overridden ``sys.path`` entry. Default path assumes the
    dev machine layout; set ``MKW_RL_SRC`` for Vast.ai.
"""

# --- sys.path forwarding --------------------------------------------------
import os
import sys

# Our project's src/ dir so ``from mkw_rl.env.* import ...`` works.
PROJECT_SRC = os.environ.get("MKW_RL_SRC", "/Users/alex/lu/git/mkwii/src")
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# Our training venv's site-packages so numpy/PIL etc. are importable from
# Dolphin's embedded Python. The master process writes its current
# ``site.getsitepackages()`` into this env var before launching Dolphin.
# Pure-Python deps (PyYAML, etc.) and C-extension deps (numpy, Pillow)
# both live here. C-extension ABI compatibility across CPython patch
# versions (3.13.x training venv vs. Dolphin's compiled-in 3.13.5) is
# not strictly guaranteed but works for numpy 2.x and Pillow 10+ in
# practice because their wheels target the CPython stable ABI.
VENV_SITE_PACKAGES = os.environ.get("MKW_RL_VENV_SITE_PACKAGES")
if VENV_SITE_PACKAGES and VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, VENV_SITE_PACKAGES)

# --- stdlib + numeric imports --------------------------------------------
from multiprocessing.connection import Client

import numpy as np
from PIL import Image

# --- Dolphin scripting-API imports (only valid inside Dolphin) ----------
from dolphin import controller, event, memory, savestate  # type: ignore[import-not-found]

# --- our modules ---------------------------------------------------------
from mkw_rl.env.reward import RaceState, RewardBreakdown, RewardConfig, TrackRewardTracker
from mkw_rl.env.track_meta import TrackMetadata

# --- config --------------------------------------------------------------
ENV_ID = int(os.environ.get("MKW_RL_ENV_ID", "0"))
MASTER_HOST = "localhost"
MASTER_PORT = 26330 + ENV_ID
SOCKET_AUTHKEY = b"mkw-rl-env-authkey"

FRAME_HEIGHT = 75
FRAME_WIDTH = 140
FRAME_STACK = 4
FRAMESKIP = 4
FRAMES_POOLED = 2  # pool last 2 frames of each frameskip window (Atari convention)

STICK_X_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
R_VALUES = [False, True]
UP_VALUES = [False, True]
L_VALUES = [False, True]
NUM_ACTIONS = len(STICK_X_VALUES) * len(R_VALUES) * len(UP_VALUES) * len(L_VALUES)
assert NUM_ACTIONS == 40


# --- RAM pointer chains (PAL RMCP01, from VIPTankz's DolphinScript.py) ---
def _resolve(base_addr: int, offsets: list[int]) -> int:
    addr = memory.read_u32(base_addr)
    for off in offsets[:-1]:
        addr = memory.read_u32(addr + off)
    return addr + offsets[-1]


class _Addresses:
    """Resolved pointers, computed once per reset. Line refs: VIPTankz/DolphinScript.py."""

    def __init__(self) -> None:
        self.race_completion = _resolve(0x809BD730, [0xC, 0x0, 0xC])  # :97
        self.current_lap = _resolve(0x809BD730, [0xC, 0x0, 0x24])  # :100
        self.stage = _resolve(0x809BD730, [0x28])  # :102
        self.surface_flags = _resolve(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x18, 0x2C])  # :129
        self.offroad_invincibility = _resolve(0x809C18F8, [0x20, 0x0, 0x0, 0x28, 0x148])  # :119
        self.race_position = _resolve(0x809C18F8, [0x20, 0x0, 0x0, 0x18, 0x3C])  # :141
        self.wall_collide = _resolve(0x809C18F8, [0x20, 0x0, 0x0, 0x8, 0x90, 0x8, 0x8])  # :145


def _read_race_state(addrs: _Addresses) -> RaceState:
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
    """Decode Discrete(40) → GCN controller dict. VIPTankz/DolphinScript.py:475-490."""
    assert 0 <= action < NUM_ACTIONS, f"action {action} out of range [0,{NUM_ACTIONS})"
    stick_idx = action // 8
    rem = action % 8
    r_idx = rem // 4
    rem = rem % 4
    up_idx = rem // 2
    l_idx = rem % 2
    return {
        "Left": False, "Right": False, "Down": False,
        "Up": UP_VALUES[up_idx], "Z": False,
        "R": R_VALUES[r_idx], "L": L_VALUES[l_idx],
        "A": True, "B": False, "X": False, "Y": False, "Start": False,
        "StickX": STICK_X_VALUES[stick_idx], "StickY": 0,
        "CStickX": 0, "CStickY": 0,
        "TriggerLeft": 0, "TriggerRight": 0,
        "AnalogA": 0, "AnalogB": 0,
        "Connected": True,
    }


def _process_frame(rgb_bytes: bytes, width: int, height: int) -> np.ndarray:
    """RGB bytes → (FRAME_HEIGHT, FRAME_WIDTH) uint8 grayscale."""
    img = Image.frombytes("RGB", (width, height), rgb_bytes, "raw")
    img = img.convert("L")
    img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


# --- top-level script body (async) --------------------------------------
print(f"[slave {ENV_ID}] starting", flush=True)

# Wait a few frames for Dolphin to initialize before doing anything.
for _ in range(4):
    await event.frameadvance()

conn = Client((MASTER_HOST, MASTER_PORT), authkey=SOCKET_AUTHKEY)
print(f"[slave {ENV_ID}] connected to master :{MASTER_PORT}", flush=True)

# Wait for the game to be well past boot.
for _ in range(8):
    await event.frameadvance()

# Stateful across reset/step cycles.
addrs: _Addresses | None = None
tracker: TrackRewardTracker | None = None
last_action: int = 0
frame_stack: np.ndarray = np.zeros((FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

# Re-apply the current action every rendered frame so the game state stays
# consistent even while the main thread is blocked in conn.recv().
def _apply_current_action() -> None:
    controller.set_gc_buttons(0, _decode_action(last_action))


event.on_frameadvance(_apply_current_action)

# Send init handshake to master.
conn.send(("init", frame_stack.tobytes()))
reply = conn.recv()
if reply != "ack":
    raise RuntimeError(f"[slave {ENV_ID}] bad init handshake: {reply!r}")

print(f"[slave {ENV_ID}] init handshake complete; entering main loop", flush=True)

while True:
    msg = conn.recv()

    # --- Control messages ---
    if msg == "close":
        print(f"[slave {ENV_ID}] close received, exiting", flush=True)
        break

    if isinstance(msg, tuple) and msg[0] == "reset":
        savestate_path, track_meta_dict = msg[1], msg[2]
        savestate.load_from_file(savestate_path)
        # Wait a few frames for the load to settle (VIPTankz:594-600 does the same).
        for _ in range(3):
            await event.frameadvance()

        addrs = _Addresses()
        meta = TrackMetadata(**track_meta_dict)
        tracker = TrackRewardTracker(track_meta=meta, config=RewardConfig())
        state = _read_race_state(addrs)
        tracker.align_to_state(state)

        # Grab an initial frame and fill the stack with copies of it.
        (w, h, data) = await event.framedrawn()
        first_img = _process_frame(data, w, h)
        for i in range(FRAME_STACK):
            frame_stack[i] = first_img

        last_action = 0
        conn.send(("reset_ok", frame_stack.tobytes()))
        continue

    # --- Step message: an int action ---
    if not isinstance(msg, int):
        raise RuntimeError(f"[slave {ENV_ID}] unexpected message: {msg!r}")

    if addrs is None or tracker is None:
        # Shouldn't happen — master should always reset before stepping.
        conn.send(("step", frame_stack.tobytes(), 0.0, False, False, {"warning": "not reset"}))
        continue

    last_action = msg
    pooled_frames = np.zeros((FRAMES_POOLED, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    accumulated = RewardBreakdown()
    terminated = False
    truncated = False

    for i in range(FRAMESKIP):
        if i >= FRAMESKIP - FRAMES_POOLED:
            # Capture the frame for the pooled observation.
            (w, h, data) = await event.framedrawn()
            pooled_frames[i - (FRAMESKIP - FRAMES_POOLED)] = _process_frame(data, w, h)
        else:
            await event.frameadvance()

        state = _read_race_state(addrs)
        step_rb, step_term, step_trunc = tracker.step(state)

        # Accumulate reward components across the frameskip window.
        accumulated.checkpoint += step_rb.checkpoint
        accumulated.offroad += step_rb.offroad
        accumulated.wall += step_rb.wall
        accumulated.finish += step_rb.finish
        accumulated.position += step_rb.position

        terminated = terminated or step_term
        truncated = truncated or step_trunc

        if terminated or truncated:
            # Pad pooled frames with the last captured frame so the obs is
            # well-defined even if we broke out before the normal capture window.
            if i < FRAMESKIP - FRAMES_POOLED:
                # We haven't captured anything this window — grab one now.
                (w, h, data) = await event.framedrawn()
                last_img = _process_frame(data, w, h)
                for j in range(FRAMES_POOLED):
                    pooled_frames[j] = last_img
            else:
                # Fill any still-zero pooled slots with the most recent capture.
                last_captured = i - (FRAMESKIP - FRAMES_POOLED)
                for j in range(last_captured + 1, FRAMES_POOLED):
                    pooled_frames[j] = pooled_frames[last_captured]
            break

    # Atari-style pooling: per-pixel max over the last FRAMES_POOLED frames.
    new_obs = np.max(pooled_frames, axis=0)
    # Roll the 4-frame stack and insert new frame at the end.
    frame_stack = np.roll(frame_stack, -1, axis=0)
    frame_stack[-1] = new_obs

    # Build the info payload for master-side logging.
    final_state = _read_race_state(addrs)
    info = {
        "reward_breakdown": accumulated.as_dict(),
        "race_completion": final_state.race_completion,
        "lap": final_state.current_lap,
        "stage": final_state.race_stage,
        "position": final_state.race_position,
    }

    conn.send(
        (
            "step",
            frame_stack.tobytes(),
            float(accumulated.total),
            bool(terminated),
            bool(truncated),
            info,
        )
    )

conn.close()
