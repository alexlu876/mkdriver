# Claude Code prompt sequence (v3)

Feed these to Claude Code one at a time. Each prompt assumes Claude Code has read `MKW_RL_SPEC.md` and `docs/PIVOT_2026-04-17.md` in full. For each prompt after the first, start the session with: *"Re-read MKW_RL_SPEC.md and docs/PIVOT_2026-04-17.md before starting."*

Each prompt is designed to produce one reviewable PR. Do not chain prompts; wait for a clean result before moving on.

**What changed from v2 (pivot 2026-04-17)**: Project skipped the BC path entirely and jumped to multi-track BTR. Prompts 1a-2c are complete but produced **dormant** code (still tested, just not on the critical path). The new Prompts 3-5 drive the env fork, BTR fork, and Vast.ai training bring-up. See `docs/PIVOT_2026-04-17.md` for rationale.

---

## Prompt P-1 — Preflight (user-run, Claude Code authors the checklist only)

**Status: done (scaffolding). User-side execution still required before Prompt 0.**

> Read MKW_RL_SPEC.md §P-1. Generate `scripts/preflight.py` (a thin wrapper that runs the mechanical parts of the checklist — e.g., verifying a RAM read succeeds from a running Dolphin scripting session) and `docs/PREFLIGHT.md` with the full six-step human checklist. Do NOT attempt to run any of it; the user runs P-1 on their own machine. Report completion and wait for the user to confirm all six checks passed and report back the Dolphin fork commit SHA + Python interpreter path that the scripting API actually links against. Commit as "preflight scaffolding".

**User then runs P-1 manually and reports back the commit SHA and Python path. Update `SETUP.md` and `.gitmodules` with the real SHA before Prompt 3.**

---

## Prompt 0 — Bootstrap

**Status: done (2026-01 bootstrap commit).** No re-run needed.

---

## Prompts 1a–1e — `.dtm` + frame pipeline (DORMANT)

**Status: done and tested (124 tests passing). DORMANT — not on the critical path post-pivot.**

Prompts 1a (parser), 1b (sanity visualizer + replay scaffolding), 1c (pairing hardening), 1d (action encoding), 1e (sequence dataset) all shipped and committed. The code lives in `src/mkw_rl/dtm/` and is exercised by `tests/test_dtm_parser.py`, `tests/test_pairing.py`, `tests/test_action_encoding.py`, `tests/test_dataset.py`, `tests/test_pipeline_smoke.py`.

**Do not re-run these prompts.** The code stays in the repo as future BC-augmentation scaffolding (see `docs/PIVOT_2026-04-17.md`). If a future Prompt 6 (BC augmentation) is executed, it consumes this pipeline as-is.

---

## Prompts 2a–2c — BC training (DORMANT)

**Status: done and tested (124 tests passing). DORMANT — not on the critical path post-pivot.**

Prompts 2a (BC model), 2b (BC training loop with TBPTT), 2c (BC eval) all shipped and committed. Code in `src/mkw_rl/bc/`.

**Do not re-run these prompts.** The code stays as future BC-augmentation scaffolding.

---

## Prompt 3 — Fork VIPTankz env into `src/mkw_rl/env/` + implement v2 methodology

**Depends on: P-1 passed, submodule SHA pinned to real value, `data/track_metadata.yaml` populated with at least Luigi Circuit's WR time (others can be filled in incrementally).**

> Re-read MKW_RL_SPEC.md, `docs/PIVOT_2026-04-17.md`, `docs/TRAINING_METHODOLOGY.md`, and `docs/REGION_DECISION.md`. Fork `third_party/Wii-RL/DolphinEnv.py` → `src/mkw_rl/env/dolphin_env.py` and `third_party/Wii-RL/DolphinScript.py` → `src/mkw_rl/env/dolphin_script.py`. Preserve VIPTankz's PAL RAM pointers (`0x809BD730`, `0x809C18F8`, `0x809C3618`) verbatim — do not re-derive.
>
> **Apply the v2 methodology from `docs/TRAINING_METHODOLOGY.md`** — VIPTankz's published code is the v1 variant that failed on 19/32 tracks; we need the v2 changes:
>
> 1. **Variable checkpoints per track**: implement `src/mkw_rl/env/reward.py::checkpoint_count_for_track(slug)` returning `round(100 × wr_seconds / 60)`, reading from `data/track_metadata.yaml`. Do NOT hardcode 200.
> 2. **Reward function** per `TRAINING_METHODOLOGY.md` §5: variable-checkpoint × speed-bonus, off-road penalty (small), wall penalty (larger), finish bonus (large), position bonus (small), plus VIPTankz's existing progress shaping between checkpoints. Each component logged separately (`reward/checkpoint`, `reward/offroad`, `reward/wall`, `reward/finish`, `reward/position`).
> 3. **Lenient reset threshold**: episode terminates ONLY when the agent fails to make ≥1s of forward progress within a 15s window. No termination on edge-fall or wall-contact — those are reward penalties. Implement as a progress tracker reading the existing VIPTankz RAM reads.
> 4. **Track-slug reset interface**: `env.reset(track_slug=None)` loads the corresponding savestate from `data/savestates/<slug>.sav`. If `track_slug` is None, the caller (not the env) is expected to have sampled one via the progress-weighted sampler — env is dumb w.r.t. curriculum.
> 5. Adapt to our conventions: `logging` module not prints; gymnasium-compliant. Observation space: `Box(0, 255, shape=(4, 75, 140), dtype=uint8)`. Action space: `Discrete(40)`.
>
> Write `tests/test_env_smoke.py` covering: env instantiates offline (no Dolphin); `checkpoint_count_for_track` returns sensible values for known tracks; reward components sum correctly given hand-constructed RAM state; reset-threshold logic fires only under the right conditions. Full integration test (with live Dolphin) is human-driven, not in CI.
>
> Commit as "phase 2.1: env fork + v2 reward + lenient reset".

---

## Prompt 4 — Fork BTR into `src/mkw_rl/rl/` + add LSTM + progress-weighted sampler

**Depends on: Prompt 3 done; user has at least 3 savestates recorded (Luigi Circuit + two contrasting tracks for first multi-track smoke).**

> Re-read MKW_RL_SPEC.md, `docs/PIVOT_2026-04-17.md`, and `docs/TRAINING_METHODOLOGY.md`. Fork `third_party/Wii-RL/BTR.py` → `src/mkw_rl/rl/btr.py`. Keep `FactorizedNoisyLinear`, `Dueling`, `PER`, `SumTree`, and the Munchausen/IQN loss math verbatim — these are the validated BTR components. But **the architecture and training loop need two material changes from v1**:
>
> 1. **Add LSTM on top of the IMPALA encoder** per `docs/TRAINING_METHODOLOGY.md` §2. v1's BTR is frame-stack-only; v2 (which succeeded) added LSTM. Reuse the `ImpalaEncoder` and LSTM pattern from our existing BC model (`src/mkw_rl/bc/model.py`) — do NOT reimplement. The flow is: `(B, T, 4, 75, 140) → ImpalaEncoder per-timestep → LSTM → IQN dueling heads → Q-values over 40 actions`. The IQN head wraps the LSTM output, not the raw encoder output.
> 2. **Recurrent replay**: modify `PER` to sample burn-in-prefixed sequences (R2D2 pattern) — default burn-in 20 frames + 40 frames of learning window. The LSTM state at the start of the learning window is derived by forwarding through the burn-in; the burn-in frames do not contribute to the loss. If this complicates PER too much, fall back to storing LSTM hidden states at sequence boundaries and loading them at sample time; document the choice.
>
> Implement the **progress-weighted track sampler** at `src/mkw_rl/rl/track_sampler.py` per `docs/TRAINING_METHODOLOGY.md` §4. Exposes `sampler.sample()` which the rollout loop calls before each `env.reset(track_slug=...)`. Maintains EMA of episode reward per track; sampling weight ∝ `max_progress_across_tracks - track_progress + epsilon`. Log `track_sampler/{slug}/weight` per update.
>
> Extract the training loop into `scripts/train_btr.py`. Config via `configs/btr.yaml` mirroring VIPTankz's hyperparameters where they still apply (batch 256, lr 1e-4, discount 0.997, n-step 3, per_alpha 0.2, target replace 500, eps_steps 2M, framestack 4, input 75×140) plus new ones: `lstm_hidden: 512`, `lstm_layers: 1`, `burn_in_len: 20`, `learning_seq_len: 40`, `spectral: off` on MPS / `on` on CUDA. Wire logging to wandb if `WANDB_API_KEY` is set else CSV to `runs/btr/metrics.csv`. Log per-track episode rewards (`track/{slug}/episode_reward`) and per-component reward signals (`reward/*`) so we can diagnose each track separately.
>
> `tests/test_btr_smoke.py`: 100 random-action steps through a mocked-env stub, verify PER sample/update path works with the recurrent sequences, verify the track sampler distribution concentrates on low-progress tracks.
>
> Commit as "phase 2.2: BTR fork with LSTM + recurrent replay + progress-weighted sampler".

---

## Prompt 5 — Vast.ai bring-up + first real training run

**Depends on: Prompt 4 done, user has Vast.ai account + SSH key + payment method set up, user has at least 3 savestates recorded (Luigi Circuit + Moo Moo Meadows + Mushroom Gorge for first real multi-track smoke).**

> Re-read `docs/PIVOT_2026-04-17.md`. Write `docs/VAST_AI_SETUP.md` covering: (a) choosing an RTX 4090 instance, (b) SSHing in and rsync-uploading `src/`, `scripts/`, `configs/`, `data/savestates/`, and the VIPTankz submodule, (c) installing Dolphin's Linux build (VIPTankz's `scripts/build-dolphin-linux.sh`), (d) environment-variable setup (`WANDB_API_KEY`, `PYTORCH_CUDA_ALLOC_CONF`), (e) launching training via `python scripts/train_btr.py --config configs/btr.yaml --device cuda`, (f) periodic rsync-back of checkpoints to local `runs/btr/`. **Do not actually run any of this from Claude Code** — the user executes on Vast.ai, Claude Code writes the runbook. Also add a smoke-test section: user should run 10 minutes locally on M4 (device mps, batch_size 32, 2 envs) to verify the training loop doesn't crash before burning Vast.ai credits. Commit as "phase 2.3: Vast.ai runbook".

**User then runs the smoke test locally, reports results, then runs the first real training session on Vast.ai.**

---

## Checkpoint — after ~5M environment steps of multi-track training

**Stop and reconvene with the user.** At this point we have real training data to answer:

1. **Is the policy learning?** Reward curves trending up on at least Luigi Circuit? On any other tracks?
2. **Track-agnostic generalization holding up?** Or are some tracks stalled (flat reward) while others make progress?
3. **If generalization is failing:** add track-id conditioning (cheap retrofit — one extra channel or embedding input to the encoder). See `docs/PIVOT_2026-04-17.md` "Research-grade uncertainty" for the diagnostic framing.
4. **Compute burn rate acceptable?** At $0.40-$0.80/hr per 4090, what's the projected dollar cost to a useful policy, and is that within budget?
5. **Reward shaping holding?** VIPTankz's progress+position reward works on Luigi; do we need per-track reward tweaks for tracks with loops / anti-grav / shortcuts?

These decisions set direction for Phase 2.5+ (reward tweaks, architecture changes, BC augmentation).

---

## Prompts 6+ (stubs — re-spec before use)

- **Prompt 6** — **BC augmentation** (optional, post-BTR). Revive the dormant `src/mkw_rl/bc/` pipeline. Consume TAS demos from `data/raw/tas/` (user sources from MKWii TAS community; no personal play time required). Either (a) pretrain the IMPALA encoder with BC then load into BTR as a warm start, or (b) add BC loss as an auxiliary term during BTR training. Trigger: only if vanilla multi-track BTR fails to generalize, or if we want to close the gap to TAS-quality driving.
- **Prompt 7** — **Retro Rewind custom tracks.** Blocked on region conflict: Retro Rewind is NTSC-U-only, we are PAL. Out of scope unless we flip regions (which would redo Prompts 3-5).
- **Prompt 8** — **Autoresearch / arch search.** Only on Vast.ai with meaningful budget. Deferred until we have a working baseline to search around.

Re-read `MKW_RL_SPEC.md` and `docs/PIVOT_2026-04-17.md` before expanding any Prompt 6+ stub into a real prompt.
