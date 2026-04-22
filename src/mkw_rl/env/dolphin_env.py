"""Master-side gymnasium env: launches Dolphin as a subprocess, talks to
``dolphin_script.py`` over a TCP socket, and exposes the standard gym
``reset(track_slug=...) → obs, info`` / ``step(action) → obs, reward,
terminated, truncated, info`` interface.

Single-env only for now. Multi-env (the 4-instance training VIPTankz uses)
is a future extension — the socket port and instance-info file scheme
already support it via ``env_id`` offsets, but this file only spawns one.

Cross-platform:
- **Darwin**: ``dolphin_app`` is a ``.app`` bundle; we invoke the inner
  ``Contents/MacOS/<name>`` binary directly so subprocess output stays
  attached.
- **Linux**: ``dolphin_app`` is either the binary path directly, or a
  directory containing ``DolphinQt`` / ``dolphin-emu`` / ``dolphin-emu-nogui``.
  We set ``QT_QPA_PLATFORM=offscreen`` so the emulator runs headless
  (required for Vast.ai / remote training hosts with no display server).

All subprocess stdout/stderr is captured to ``log_dir/dolphin_env_{env_id}.log``
for post-mortem diagnosis. Socket accept + recv calls are bounded with
timeouts so a stuck/crashed Dolphin surfaces as ``TimeoutError`` instead of
blocking indefinitely.

See ``docs/TRAINING_METHODOLOGY.md`` and ``docs/PIVOT_2026-04-17.md`` for
the v2 changes this env implements on top of VIPTankz's published v1.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from multiprocessing.connection import Listener
from pathlib import Path
from typing import IO, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mkw_rl.env.track_meta import load_track_metadata

log = logging.getLogger(__name__)

# Match the slave's constants.
FRAME_HEIGHT = 75
FRAME_WIDTH = 140
FRAME_STACK = 4
NUM_ACTIONS = 40
SOCKET_AUTHKEY = b"mkw-rl-env-authkey"
BASE_PORT = 26330

# Timeouts for the master ↔ slave socket. Accept + reset are generous because
# Dolphin boot + savestate load can take >60s on a cold box; step is tight
# because a healthy 20ms frame × FRAMESKIP=4 = ~80ms, so anything over ~30s
# is definitely a hang not a slow frame.
ACCEPT_TIMEOUT_S: float = 120.0
RESET_TIMEOUT_S: float = 120.0
STEP_TIMEOUT_S: float = 30.0

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
        log_dir: Path | str | None = None,
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
        # Subprocess log destination. Defaults to a tmpdir so the env works
        # standalone (e.g., in tests), but production use should pass the
        # training ``log_dir`` so Dolphin logs live next to CSV/ckpt outputs.
        self._log_dir = Path(log_dir) if log_dir is not None else Path(tempfile.gettempdir())

        if not self.dolphin_app.exists():
            raise FileNotFoundError(f"Dolphin app not found: {self.dolphin_app}")
        if not self.iso.exists():
            raise FileNotFoundError(f"MKW ISO not found: {self.iso}")

        self.track_metadata = load_track_metadata(track_metadata_path)

        # Preflight: diff the on-disk savestates against the metadata YAML.
        # Slugs in one but not the other are loud warnings — they will either
        # fail per-track at reset() or be silently unsampled by the curriculum.
        self._preflight_check_tracks()

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._process: subprocess.Popen | None = None
        self._conn = None
        self._current_track_slug: str | None = None
        self._dolphin_log_fh: IO[bytes] | None = None

    # ------------------------------------------------------------------
    # Subprocess lifecycle.
    # ------------------------------------------------------------------

    def _port(self) -> int:
        return BASE_PORT + self.env_id

    def _dolphin_log_path(self) -> Path:
        return self._log_dir / f"dolphin_env_{self.env_id}.log"

    def _preflight_check_tracks(self) -> None:
        """Warn about savestate/YAML mismatches.

        Masked failures are painful in training: a slug present on disk but
        missing from YAML produces a KeyError at ``reset()``, which our
        train-loop crash-catch (``train.py``'s per-track counter) will absorb
        after three failed launches — wasting ~90s of Dolphin-boot compute
        per missing slug. Logging loudly at construction time lets the user
        fix the YAML/savestate discrepancy BEFORE training burns cycles.
        """
        on_disk = {p.stem for p in self.savestate_dir.glob("*.sav")}
        in_yaml = set(self.track_metadata.keys())
        missing_yaml = on_disk - in_yaml
        missing_savestate = in_yaml - on_disk
        if missing_yaml:
            log.warning(
                "[env %d] %d savestates on disk are missing from track_metadata.yaml "
                "(reset() will KeyError on these): %s",
                self.env_id, len(missing_yaml), sorted(missing_yaml),
            )
        if missing_savestate:
            log.info(
                "[env %d] %d track_metadata entries have no savestate "
                "(curriculum will not sample these until recorded): %s",
                self.env_id, len(missing_savestate), sorted(missing_savestate),
            )

    def _resolve_inner_binary(self) -> Path:
        """Find the Dolphin executable on the current platform.

        Darwin: invokes the inner ``Contents/MacOS/<stem>`` of a ``.app``
        bundle, OR accepts ``dolphin_app`` pointing at the inner binary.
        Linux: accepts either a direct binary path, or a directory
        containing a known Dolphin binary name.
        """
        system = platform.system()
        if system == "Darwin":
            if self.dolphin_app.suffix == ".app" or self.dolphin_app.is_dir():
                inner = self.dolphin_app / "Contents" / "MacOS" / self.dolphin_app.stem
            else:
                inner = self.dolphin_app
        elif system == "Linux":
            if self.dolphin_app.is_dir():
                # Headless-first: on Linux (Vast.ai / servers) we prefer
                # ``dolphin-emu-nogui`` because it skips the Qt event loop +
                # rendering subsystem — ~1.5-2× faster per frame, and avoids
                # needing a display server. Fall back to the Qt variants if
                # the user only has those installed. Felk's scripting fork
                # ships scripting support (`--script`) in both binaries.
                for candidate in (
                    "dolphin-emu-nogui",
                    "DolphinQt",
                    "dolphin-emu",
                ):
                    p = self.dolphin_app / candidate
                    if p.exists() and os.access(p, os.X_OK):
                        inner = p
                        break
                else:
                    raise FileNotFoundError(
                        f"no Dolphin binary in {self.dolphin_app}; expected one of "
                        "dolphin-emu-nogui / DolphinQt / dolphin-emu (executable)"
                    )
            else:
                inner = self.dolphin_app  # direct binary
        else:
            raise NotImplementedError(f"platform {system!r} not supported")

        if not inner.exists():
            raise FileNotFoundError(f"Dolphin binary not found: {inner}")
        return inner

    @staticmethod
    def _find_xvfb_run() -> str | None:
        """Return the path to ``xvfb-run`` if present on PATH, else None."""
        import shutil  # noqa: PLC0415 — local import
        return shutil.which("xvfb-run")

    def _launch_dolphin(self) -> None:
        """Launch Dolphin with our slave script attached.

        On Linux, wraps the binary with ``xvfb-run`` to provide a virtual X
        display — ``dolphin-emu-nogui`` still initializes a Qt platform
        plugin and a video backend that both need an X server, even in
        headless mode (``QT_QPA_PLATFORM=offscreen`` alone is not
        sufficient). Vast.ai images include Xvfb via apt; we require it.
        On macOS a visible window is expected (dev-machine only).
        """
        inner_binary = self._resolve_inner_binary()

        env = dict(os.environ)
        env["MKW_RL_ENV_ID"] = str(self.env_id)
        env["MKW_RL_SRC"] = str(self._mkw_rl_src)

        xvfb_prefix: list[str] = []
        if platform.system() == "Linux":
            # Still set offscreen QPA as belt-and-braces; our real headless
            # strategy is the xvfb-run wrapper below.
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
            xvfb_run = self._find_xvfb_run()
            if xvfb_run is None:
                raise FileNotFoundError(
                    "xvfb-run not found on PATH. Install with `apt-get install "
                    "xvfb` — it's required for headless Dolphin on Linux since "
                    "the binary still requests an X display for Qt + video init."
                )
            # -a auto-selects a free display number; `-s` sets the virtual screen
            # size (doesn't affect our observations — we read framebuffer via
            # event.framedrawn, not the Qt window). Each env_id gets its own
            # Xvfb instance, so multi-env launches don't collide on :99.
            xvfb_prefix = [xvfb_run, "-a", "-s", "-screen 0 1024x768x24"]

        # Forward our training venv's site-packages so Dolphin's embedded
        # Python can import numpy, Pillow, etc. (pure-Python deps + C
        # extensions). Match VIPTankz's shared_site.txt pattern but via
        # env var rather than a file on disk.
        import site  # noqa: PLC0415 — local import keeps top-level imports tidy
        for p in site.getsitepackages() + [site.getusersitepackages()]:
            if p and "site-packages" in p and Path(p).exists():
                env["MKW_RL_VENV_SITE_PACKAGES"] = p
                log.info("[env %d] forwarding site-packages: %s", self.env_id, p)
                break
        else:
            log.warning(
                "[env %d] no site-packages path found via site.getsitepackages(); "
                "Dolphin's embedded Python may fail to import numpy",
                self.env_id,
            )

        cmd = [
            *xvfb_prefix,
            str(inner_binary),
            "--no-python-subinterpreters",
            "--script",
            str(_SLAVE_SCRIPT),
            f"--exec={self.iso}",
        ]

        # Capture subprocess stdout+stderr to a log file. Without this,
        # Dolphin tracebacks vanish on Vast.ai (no attached tty to nohup from),
        # and debugging a crash requires re-running under strace.
        log_path = self._dolphin_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._dolphin_log_fh = open(log_path, "wb")  # noqa: SIM115 — held for lifetime of subprocess
        log.info("[env %d] launching: %s (log → %s)", self.env_id, " ".join(cmd), log_path)
        self._process = subprocess.Popen(
            cmd, env=env,
            stdout=self._dolphin_log_fh,
            stderr=subprocess.STDOUT,
        )

    def _wait_for_slave(self) -> None:
        """Bind the Listener and accept the slave's connection.

        Uses the underlying socket's ``settimeout`` so a Dolphin that fails
        to launch / crashes / hangs surfaces as ``TimeoutError`` within
        ACCEPT_TIMEOUT_S, not a deadlock of the master loop.
        """
        port = self._port()
        log.info("[env %d] listening on :%d (accept timeout=%.0fs)",
                 self.env_id, port, ACCEPT_TIMEOUT_S)
        listener = Listener(("localhost", port), authkey=SOCKET_AUTHKEY)
        # multiprocessing.connection.Listener wraps a socket we can access
        # via `_listener._socket` on every supported Python; setting its
        # timeout causes `accept()` to raise `socket.timeout` on expiry.
        listener._listener._socket.settimeout(ACCEPT_TIMEOUT_S)  # type: ignore[attr-defined]
        try:
            try:
                self._conn = listener.accept()
            except TimeoutError as exc:  # socket.timeout aliases to TimeoutError in py≥3.10
                raise TimeoutError(
                    f"Dolphin slave didn't connect within {ACCEPT_TIMEOUT_S:.0f}s on "
                    f":{port} — check {self._dolphin_log_path()}"
                ) from exc
        finally:
            # Reset the listener socket's timeout so subsequent users of the
            # Listener object (none today) aren't affected.
            listener._listener._socket.settimeout(None)  # type: ignore[attr-defined]
        log.info("[env %d] slave connected", self.env_id)
        # Expect an "init" message with the first observation.
        msg = self._recv_with_timeout(RESET_TIMEOUT_S, "init handshake")
        if not (isinstance(msg, tuple) and msg[0] == "init"):
            raise RuntimeError(f"[env {self.env_id}] bad slave handshake: {msg!r}")
        self._conn.send("ack")

    def _recv_with_timeout(self, timeout_s: float, context: str) -> Any:
        """Poll-then-recv so a silent slave surfaces as a bounded TimeoutError.

        ``Connection.poll(timeout)`` returns True iff data is available within
        the deadline; recv without prior poll would block indefinitely.
        """
        if self._conn is None:
            raise RuntimeError(f"env {self.env_id}: no connection during {context}")
        if not self._conn.poll(timeout_s):
            raise TimeoutError(
                f"env {self.env_id}: slave silent for {timeout_s:.0f}s during {context}; "
                f"check {self._dolphin_log_path()}"
            )
        return self._conn.recv()

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

        meta_dict = asdict(self.track_metadata[track_slug])
        self._conn.send(("reset", str(savestate_path), meta_dict))

        # Slave replies with ("reset_ok", initial_obs_bytes) once the savestate
        # is loaded and the frame stack is primed, or ("reset_err", msg, tb)
        # if anything went wrong on the slave side. A hung slave (Dolphin froze
        # mid-load) surfaces as TimeoutError via _recv_with_timeout; EOFError
        # means the slave process exited.
        try:
            msg = self._recv_with_timeout(RESET_TIMEOUT_S, f"reset(track_slug={track_slug!r})")
        except EOFError as exc:
            raise RuntimeError(
                f"slave closed connection during reset (track_slug={track_slug!r}) — "
                "Dolphin likely crashed; check its log"
            ) from exc
        if isinstance(msg, tuple) and msg[0] == "reset_err":
            raise RuntimeError(
                f"slave failed to reset (track_slug={track_slug!r}): {msg[1]}\n"
                f"slave traceback:\n{msg[2]}"
            )
        if not (isinstance(msg, tuple) and msg[0] == "reset_ok"):
            raise RuntimeError(f"unexpected reset reply from slave: {msg!r}")

        # Reset succeeded — commit the new slug.
        self._current_track_slug = track_slug
        obs = np.frombuffer(msg[1], dtype=np.uint8).reshape(
            FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH
        ).copy()  # copy because frombuffer returns a view into the transient bytes
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
        msg = self._recv_with_timeout(STEP_TIMEOUT_S, f"step(action={action})")
        assert isinstance(msg, tuple) and msg[0] == "step", f"unexpected msg: {msg!r}"
        # Wire format: ("step", obs_bytes, reward, terminated, truncated, info).
        _, obs_bytes, reward, terminated, truncated, info = msg

        obs = np.frombuffer(obs_bytes, dtype=np.uint8).reshape(
            FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH
        ).copy()
        info["track_slug"] = self._current_track_slug
        return obs, reward, bool(terminated), bool(truncated), info

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
        if self._dolphin_log_fh is not None:
            try:
                self._dolphin_log_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._dolphin_log_fh = None
        # Clear episode-scoped state so a reuse doesn't leak stale slug into
        # info dicts (reuse currently unused in practice; guarding for safety).
        self._current_track_slug = None

    # Gym idioms.
    def __enter__(self) -> MkwDolphinEnv:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def available_tracks(
    savestate_dir: Path | str | None = None,
    track_metadata_path: Path | str | None = None,
) -> list[str]:
    """Return the slugs of all tracks that are BOTH on disk AND in the YAML.

    This is what the curriculum sampler should consume — a slug on disk but
    not in the YAML would crash at ``reset()`` with KeyError, and we already
    log a warning about it from ``MkwDolphinEnv._preflight_check_tracks``.

    Passing ``track_metadata_path=None`` skips the YAML intersection for
    back-compat (returns raw disk glob), but production callers should pass
    the same path used when constructing ``MkwDolphinEnv``.
    """
    d = Path(savestate_dir) if savestate_dir else _DEFAULT_SAVESTATE_DIR
    on_disk = sorted(p.stem for p in d.glob("*.sav"))
    if track_metadata_path is None:
        return on_disk
    meta = load_track_metadata(track_metadata_path)
    return [s for s in on_disk if s in meta]
