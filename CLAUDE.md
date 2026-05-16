# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mkw-rl`: a multi-track Beyond-the-Rainbow (BTR) RL agent for Mario Kart Wii, run via the Dolphin emulator. Forks VIPTankz's published single-track BTR (`third_party/Wii-RL`) and extends it to **track-agnostic generalization across all 32 vanilla tracks with a single policy, no track-id conditioning** — research territory as of 2026-04. The BC scaffolding under `src/mkw_rl/dtm/` + `src/mkw_rl/bc/` is **dormant** (kept as future augmentation, not on the active path).

Primary reference docs (read these before non-trivial work):

- `MKW_RL_SPEC.md` — full spec, with fixed assumptions that should not be revisited.
- `docs/TRAINING_METHODOLOGY.md` — v2 training design (LSTM-on-BTR, variable checkpoints, progress-weighted sampler, lenient reset). Reward function and methodology live here.
- `docs/PIVOT_2026-04-17.md` — why BC was deferred for direct-to-BTR multi-track.
- `CHANGES.md` — running build-out log.
- `SETUP.md` — one-time dev setup + Vast.ai production setup.

## Commands

```bash
# Tests (fast; pure-Python, no Dolphin)
uv run pytest
uv run pytest tests/test_btr_training.py -k test_resume    # single test

# Lint
uv run ruff check .
uv run ruff format .

# Smoke test against live Dolphin (~1 min, Luigi Circuit, tiny model)
# Run before any real training launch.
uv run python scripts/train_btr.py --config configs/btr.yaml --testing

# Production training (CUDA / Vast.ai)
.venv/bin/python scripts/train_btr.py --config configs/btr.yaml --device cuda

# Resume (replay buffer is NOT checkpointed; first ~200K env steps post-resume are random-policy warmup)
.venv/bin/python scripts/train_btr.py --config configs/btr.yaml --device cuda \
  --resume runs/btr/btr_20260422_123456_final.pt
```

**On Linux/Vast.ai do NOT use `uv run` for training.** It silently prevents Dolphin's embedded scripting engine from initializing — emulator boots, burns CPU, but the Python slave never connects and training stalls at the header row. Invoke `.venv/bin/python` directly. (Reason not fully nailed down; documented in `SETUP.md`.)

## Architecture

**Master ↔ slave Dolphin loop.** `MkwDolphinEnv` (`src/mkw_rl/env/dolphin_env.py`) is a `gymnasium.Env` that launches Dolphin as a subprocess. Inside Dolphin, `src/mkw_rl/env/dolphin_script.py` runs in the emulator's embedded Python interpreter and talks to the master over a TCP socket (port `26330 + env_id`). The slave reads RAM (lap/checkpoint/position/velocity/wall/offroad), grabs framebuffers, and injects controller inputs. The master orchestrates resets via savestate loads.

**Reward** (`src/mkw_rl/env/reward.py`) is the v2 layered scheme: per-checkpoint × speed-bonus + per-frame off-road/wall penalties + finish bonus + position bonus + death-penalty (bottomless-pit fall) + lenient reset (truncate after 750 frames without progress). Methodology §5.

**BTR policy** (`src/mkw_rl/rl/model.py`): IMPALA-style CNN encoder + stateful LSTM (hidden=512) + IQN/Munchausen dueling heads with NoisyNets. Action space is discrete 40-way per VIPTankz. LSTM-on-top-of-BTR is the v2 deviation from VIPTankz's frame-stack-only `BTR.py` — see methodology §2.

**Replay** (`src/mkw_rl/rl/replay.py`): PER with **stored-hidden** transitions (Kapturowski 2019 §3.2 "stored state"). Each transition carries the `(h, c)` the rollout agent used at that timestep; learn-step is single-step from those hiddens, no sequence unroll. This is a 2026-04-23 refactor — *not* R2D2 burn-in sequences. Cuts encoder activation memory ~60×.

**Track curriculum** (`src/mkw_rl/rl/track_sampler.py`): `ProgressWeightedTrackSampler` weights tracks by inverse EMA return so harder tracks see more samples, with a `min_samples_per_track` floor (10) for cold start. Hot-add: new savestates dropped into `data/savestates/` while training is live get picked up via periodic polling. Tolerates `update()` calls for unknown slugs (silent skip — avoids races with crash-counter blacklist).

**Multi-env.** `env.num_envs > 1` runs N parallel Dolphin instances. Each needs its own `dolphin{i}/` directory because Dolphin's `portable.txt` forces per-binary `./User/` (shared dirs → shared shader/JIT caches → corruption). Use `scripts/setup_dolphin_instances.py` to clone from `dolphin0/`. Recommended baseline is 2; 4 is stable on Vast.ai EPYC + RTX 5080 after the `_cleanup_stale_x11_state` fix (auto-runs at `train()` start, wipes orphan `/tmp/.X11-unix/X*` sockets that otherwise SIGSEGV new Dolphins).

**Training loop** (`src/mkw_rl/rl/train.py`): IMPALA-style threaded rollouts under a single `agent_lock`. Periodic checkpoints rotate (keeps newest 5) but `_final.pt` and `_diverged.pt` are pinned. Checkpoint writes are atomic (`tmp + os.replace`) so SIGTERM mid-write doesn't truncate the resume payload. Signal handling: SIGTERM → graceful shutdown writes a `_final.pt`, training resumes cleanly via `--resume`.

## Fixed assumptions

These are load-bearing — confirm with the user before changing any:

- **PAL only** (Game ID `RMCP01`). Decision in `docs/REGION_DECISION.md`. Implies Retro Rewind (NTSC-U) is out of scope.
- **Python 3.12–3.13** (`pyproject.toml`). 3.12 floor is for Dolphin-built-from-source on Linux which links against `libpython3.12`.
- **Frame processing**: 140×75 grayscale, 4-frame stack, frameskip=4 (matches VIPTankz `DolphinEnv.py:91-92`). Don't change without a written reason.
- **Action space**: 40-way discrete per VIPTankz. No analog steering on the BTR path.
- **VIPTankz submodule SHA**: `third_party/Wii-RL` is pinned to `d8358cbc5feef41161522e51b60fba100506d489` (placeholder; spec §P-1 requires the user to confirm).

## Non-obvious gotchas

- **`uv run` on Linux breaks Dolphin scripting** (see Commands above). Direct `.venv/bin/python` only.
- **cuDNN library order**: torch 2.11+cu128 bundles cuDNN 9.19; Vast.ai's PyTorch template ships system cuDNN 9.8 which wins on `LD_LIBRARY_PATH` otherwise. Prepend the venv's bundled path — baked into Vast.ai's `.bashrc` per `SETUP.md`.
- **`--extra linux-cuda`** is required on Linux CUDA hosts (`uv sync --extra dev --extra linux-cuda`). Adds `nvidia-cusparselt-cu12` + `nvidia-nvshmem-cu12` which torch dlopens but doesn't pull as a hard dep.
- **Replay buffer is not checkpointed.** Resumes do ~200K env steps of random-policy warmup again (<0.1% of a 500M run). Acceptable; documented in `train_btr.py`.
- **Savestate naming**: per-track `.sav` files in `data/savestates/{slug}_NNN.sav` (12 per track at distributed race-completion values per `docs/SAVESTATE_PROTOCOL.md`). `env.reset()` samples one uniformly. Old single-savestate names in `data/savestates/_legacy/`.
- **Audit history**: `AUDIT.md` is the latest forensic review; treat its applied fixes (`199ec2e audit: apply blocker + high-severity fixes from AUDIT.md`) as ground truth, the original VIPTankz quirks as outdated context.

## Project status (2026-05-16)

- Phase 2.1 env fork: live-smoke tested on Luigi Circuit.
- Phase 2.2 BTR fork: passes 1–5 complete (stored-hidden replay refactor done 2026-04-23).
- Phase 2.3 BC eval / side-by-side video: complete.
- 4 tracks have savestates (`luigi_circuit_tt`, `moo_moo_meadows_tt`, `mushroom_gorge_tt`, `toads_factory_tt`); 28 more pending user-recorded.
- Vast.ai multi-env production runs ongoing; Xvfb crash recovery is lossy but functional.
