"""Live-Dolphin smoke test for ``MkwDolphinEnv``.

Instantiates the env, resets it on Luigi Circuit, runs ~20 no-op steps
(just hold A, no stick input), prints one line of diagnostic info per
step, then closes. Good for verifying the master/slave IPC + reward
pipeline end-to-end before writing training code on top.

Prereqs
-------
1. Dolphin pre-compiled binary present at the default path (run
   ``python3 scripts/download_dolphin.py`` under ``~/code/mkw/Wii-RL``
   if you haven't).
2. ISO at ``~/code/mkw/Wii-RL/game/mkw.iso``.
3. A savestate at ``data/savestates/luigi_circuit_tt.sav``. Stage one
   from VIPTankz's bundle (any of their Luigi states works)::

        cp ~/code/mkw/Wii-RL/MarioKartSaveStates/RMCP01.s02 \\
           data/savestates/luigi_circuit_tt.sav

4. The Python 3.13.5 stdlib symlink for Dolphin (see
   ``docs/PREFLIGHT.md`` Step 3) must already be in place. If you
   completed P-1, it is.

Usage
-----
::

    uv run python scripts/smoke_env.py

A Dolphin window will pop up — don't interact with it. The script
sends actions, prints per-step diagnostics, and shuts Dolphin down
cleanly via ``env.close()``.

What to look for
----------------
- Slave connects: ``[slave 0] connected to master :26330`` appears.
- Init handshake completes: ``[slave 0] init handshake complete``.
- Reset returns a non-zero observation (frame_stack filled with
  actual game frames, not zeros).
- Step loop prints race_completion ticking up as the default action
  (action 0: full-left stick, A held) does ... something.
- Finally, ``env.close()`` kills Dolphin cleanly.

Known rough edges (phase 2.1 TODOs being shaken out here)
---------------------------------------------------------
- Listener has no timeout, so if Dolphin fails to start we hang. Kill
  with Ctrl+C and check the Dolphin log for the real error.
- Action 0 is full-left-stick which will probably just drive the kart
  into a wall. That's fine for this smoke test — we just want the
  wire format working.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Bootstrap logging before importing our modules so we see the master-side logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("smoke_env")


def main() -> int:
    # Deferred import so the logging config above takes effect first.
    from mkw_rl.env.dolphin_env import MkwDolphinEnv

    project_root = Path(__file__).resolve().parents[1]
    savestate_dir = project_root / "data" / "savestates"
    luigi_sav = savestate_dir / "luigi_circuit_tt.sav"
    if not luigi_sav.exists():
        log.error(
            "missing savestate %s — stage one from VIPTankz's bundle first. See "
            "the script's docstring for the exact cp command.",
            luigi_sav,
        )
        return 1

    env = MkwDolphinEnv()

    try:
        log.info("constructing env done; calling reset()")
        t0 = time.time()
        obs, info = env.reset(options={"track_slug": "luigi_circuit_tt"})
        log.info(
            "reset OK in %.1fs: obs.shape=%s dtype=%s min=%d max=%d info=%s",
            time.time() - t0,
            obs.shape,
            obs.dtype,
            int(obs.min()),
            int(obs.max()),
            info,
        )

        log.info("stepping for ~20 actions (action=0 every step)")
        total_reward = 0.0
        for i in range(20):
            obs, reward, terminated, truncated, info = env.step(0)
            total_reward += reward
            rb = info.get("reward_breakdown", {})
            log.info(
                "step %02d: reward=%+0.4f total=%+0.4f term=%s trunc=%s race_com=%.4f stage=%s pos=%s "
                "rb={cp=%+0.4f off=%+0.4f wall=%+0.4f fin=%+0.4f pos=%+0.4f}",
                i,
                reward,
                total_reward,
                terminated,
                truncated,
                info.get("race_completion", float("nan")),
                info.get("stage"),
                info.get("position"),
                rb.get("checkpoint", 0.0),
                rb.get("offroad", 0.0),
                rb.get("wall", 0.0),
                rb.get("finish", 0.0),
                rb.get("position", 0.0),
            )
            if terminated or truncated:
                log.info("episode ended at step %d — not resetting, breaking", i)
                break

        log.info("smoke test complete; closing env")
        env.close()
        return 0
    except Exception:  # noqa: BLE001
        log.exception("smoke test failed")
        env.close()
        return 2


if __name__ == "__main__":
    sys.exit(main())
