"""Master-side gymnasium env: launches Dolphin as a subprocess, talks to
``dolphin_script.py`` over a TCP socket, and exposes the standard gym
``reset(track_slug=...) → obs, info`` / ``step(action) → obs, reward,
terminated, truncated, info`` interface.

Single-env only for now. Multi-env (the 4-instance training VIPTankz uses)
is a future extension — the socket port and instance-info file scheme
already support it via ``env_id`` offsets, but this file only spawns one.

.. note::
    This file is a **skeleton ready for integration testing**. The
    IPC wire format + subprocess lifecycle are implemented, but
    several pieces (observation framestack assembly, action handshake
    timing, subprocess cleanup on exceptions) will need adjustment
    once we run against live Dolphin and see what actually works.
    Marked with ``TODO(phase 2.1)`` comments inline.

See ``docs/TRAINING_METHODOLOGY.md`` and ``docs/PIVOT_2026-04-17.md`` for
the v2 changes this env implements on top of VIPTankz's published v1.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import asdict
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mkw_rl.env.track_meta import TrackMetadata, load_track_metadata

log = logging.getLogger(__name__)

# Match the slave's constants.
FRAME_HEIGHT = 75
FRAME_WIDTH = 140
FRAME_STACK = 4
NUM_ACTIONS = 40
SOCKET_AUTHKEY = b"mkw-rl-env-authkey"
BASE_PORT = 26330

# Default project paths. Override via constructor args on non-dev machines.
_DEFAULT_DOLPHIN_APP = (
    Path.home() / "code" / "mkw" / "Wii-RL" / "dolphin0" / "DolphinQt.app"
)
_DEFAULT_ISO = Path.home() / "code" / "mkw" / "Wii-RL" / "game" / "mkw.iso"
_DEFAULT_SAVESTATE_DIR = Path(__file__).resolve().parents[3] / "data" / "savestates"
_SLAVE_SCRIPT = Path(__file__).resolve().parent / "dolphin_script.py"


class MkwDolphinEnv(gym.Env):
    """Gym wrapper around a single Dolphin instance running our slave script."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env_id: int = 0,
        savestate_dir: Path | str | None = None,
        dolphin_app: Path | str | None = None,
        iso: Path | str | None = None,
        track_metadata_path: Path | str | None = None,
        mkw_rl_src: Path | str | None = None,
    ) -> None:
        super().__init__()
        self.env_id = env_id
        self.savestate_dir = Path(savestate_dir) if savestate_dir else _DEFAULT_SAVESTATE_DIR
        self.dolphin_app = Path(dolphin_app) if dolphin_app else _DEFAULT_DOLPHIN_APP
        self.iso = Path(iso) if iso else _DEFAULT_ISO
        self._mkw_rl_src = (
            Path(mkw_rl_src)
            if mkw_rl_src
            else Path(__file__).resolve().parents[2]  # src/
        )

        if not self.dolphin_app.exists():
            raise FileNotFoundError(f"Dolphin app not found: {self.dolphin_app}")
        if not self.iso.exists():
            raise FileNotFoundError(f"MKW ISO not found: {self.iso}")

        self.track_metadata = load_track_metadata(track_metadata_path)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._process: subprocess.Popen | None = None
        self._conn = None
        self._current_track_slug: str | None = None

    # ------------------------------------------------------------------
    # Subprocess lifecycle.
    # ------------------------------------------------------------------

    def _port(self) -> int:
        return BASE_PORT + self.env_id

    def _launch_dolphin(self) -> None:
        """Launch Dolphin with our slave script attached.

        Mirrors VIPTankz's DolphinEnv.create_dolphin Darwin branch (205-214)
        — macOS requires ``open --args`` because .app bundles aren't directly
        executable, OR you invoke the inner ``Contents/MacOS/<name>`` binary
        directly. We use the latter so subprocess output stays attached.
        """
        if platform.system() != "Darwin":
            raise NotImplementedError(
                "non-Darwin support deferred; Vast.ai runbook will parameterize."
            )
        inner_binary = self.dolphin_app / "Contents" / "MacOS" / self.dolphin_app.stem
        if not inner_binary.exists():
            raise FileNotFoundError(f"Dolphin inner binary not found: {inner_binary}")

        env = dict(os.environ)
        env["MKW_RL_ENV_ID"] = str(self.env_id)
        env["MKW_RL_SRC"] = str(self._mkw_rl_src)

        cmd = [
            str(inner_binary),
            "--no-python-subinterpreters",
            "--script",
            str(_SLAVE_SCRIPT),
            f"--exec={self.iso}",
        ]
        log.info("[env %d] launching: %s", self.env_id, " ".join(cmd))
        self._process = subprocess.Popen(cmd, env=env)

    def _wait_for_slave(self) -> None:
        """Bind the Listener and accept the slave's connection."""
        port = self._port()
        log.info("[env %d] listening on :%d", self.env_id, port)
        listener = Listener(("localhost", port), authkey=SOCKET_AUTHKEY)
        # TODO(phase 2.1): add a timeout / heartbeat so we don't block forever
        # if Dolphin failed to launch.
        self._conn = listener.accept()
        log.info("[env %d] slave connected", self.env_id)
        # Expect an "init" message with the first observation.
        msg = self._conn.recv()
        if not (isinstance(msg, tuple) and msg[0] == "init"):
            raise RuntimeError(f"[env {self.env_id}] bad slave handshake: {msg!r}")
        self._conn.send("ack")

    def _ensure_running(self) -> None:
        if self._process is None:
            self._launch_dolphin()
            self._wait_for_slave()

    # ------------------------------------------------------------------
    # Gym API.
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._ensure_running()

        track_slug = (options or {}).get("track_slug")
        if track_slug is None:
            raise ValueError(
                "reset() requires options={'track_slug': ...}. The env is "
                "deliberately dumb about curriculum — caller (e.g., a "
                "progress-weighted sampler) picks the track."
            )
        if track_slug not in self.track_metadata:
            raise KeyError(f"unknown track slug: {track_slug!r}")

        savestate_path = self.savestate_dir / f"{track_slug}.sav"
        if not savestate_path.exists():
            raise FileNotFoundError(
                f"no savestate for track {track_slug!r} at {savestate_path}. "
                "Record it via scripts/record_savestates.py first."
            )

        self._current_track_slug = track_slug
        meta_dict = asdict(self.track_metadata[track_slug])
        self._conn.send(("reset", str(savestate_path), meta_dict))

        # TODO(phase 2.1): the slave's reset logic runs in its on_frame callback
        # a frame later; the first step() will include the first real observation.
        # For now, return a zero obs so the API contract is satisfied.
        obs = np.zeros((FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        info = {"track_slug": track_slug}
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError("env not reset yet")
        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"action {action} out of range [0,{NUM_ACTIONS})")

        self._conn.send(int(action))
        msg = self._conn.recv()
        assert isinstance(msg, tuple) and msg[0] == "step", f"unexpected msg: {msg!r}"
        _, obs_bytes, reward, done, info = msg

        if obs_bytes is None:
            obs = np.zeros((FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        else:
            obs = np.frombuffer(obs_bytes, dtype=np.uint8).reshape(
                FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH
            )
        info["track_slug"] = self._current_track_slug
        # Gym distinguishes terminated (MDP end) from truncated (time limit).
        # Our reset-threshold path is effectively a truncation; the finish
        # condition is termination. For phase 1 we surface both as `done` via
        # `terminated` since we don't yet track which triggered it.
        # TODO(phase 2.1): slave should flag which condition fired so we
        # can populate truncated correctly.
        return obs, reward, bool(done), False, info

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send("close")
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    # Gym idioms.
    def __enter__(self) -> "MkwDolphinEnv":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def available_tracks(savestate_dir: Path | str | None = None) -> list[str]:
    """Return the slugs of all tracks that have a savestate on disk.

    The progress-weighted sampler (Prompt 4) will consume this.
    """
    d = Path(savestate_dir) if savestate_dir else _DEFAULT_SAVESTATE_DIR
    return sorted(p.stem for p in d.glob("*.sav"))
