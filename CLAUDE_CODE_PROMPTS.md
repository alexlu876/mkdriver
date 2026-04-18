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

## Prompt 3 — Fork VIPTankz env into `src/mkw_rl/env/`

**Depends on: P-1 passed, submodule SHA pinned to real value.**

> Re-read MKW_RL_SPEC.md §4 (renumbered to current Phase 2 per `docs/PIVOT_2026-04-17.md`) and `docs/REGION_DECISION.md`. Fork `third_party/Wii-RL/DolphinEnv.py` → `src/mkw_rl/env/dolphin_env.py` and `third_party/Wii-RL/DolphinScript.py` → `src/mkw_rl/env/dolphin_script.py`. Preserve VIPTankz's PAL RAM pointers (`0x809BD730`, `0x809C18F8`, `0x809C3618`) verbatim — do not re-derive. Port the reward function as-is (progress + position). Adapt the env class to: (a) use our logging conventions (standard `logging` module, not prints where avoidable), (b) accept a savestate directory pointing at `data/savestates/` so we can load any track's savestate by track slug, (c) extend `reset()` with a `track_slug` argument (optional; if None, sample uniformly from available savestates) that loads the corresponding savestate, (d) remain gymnasium-compliant. Observation space: `Box(0, 255, shape=(4, 75, 140), dtype=uint8)` per VIPTankz. Action space: `Discrete(40)` per VIPTankz. Write `tests/test_env_smoke.py` that verifies the env instantiates without Dolphin running (unit-testable parts only) — the full integration test requires a live Dolphin and is human-driven. Commit as "phase 2.1: env fork".

---

## Prompt 4 — Fork BTR into `src/mkw_rl/rl/`

**Depends on: Prompt 3 done, user has at least one savestate recorded (Luigi Circuit minimum for smoke testing).**

> Re-read MKW_RL_SPEC.md and `docs/PIVOT_2026-04-17.md`. Fork `third_party/Wii-RL/BTR.py` → `src/mkw_rl/rl/btr.py`. Keep the `FactorizedNoisyLinear`, `Dueling`, `ImpalaCNNLargeIQN`, `PER` / `SumTree`, and `Agent` classes verbatim (these are the validated BTR implementation). Extract the `main()` training loop into `scripts/train_btr.py` and replace hardcoded config with a `configs/btr.yaml` that mirrors VIPTankz's defaults (batch size 256, lr 1e-4, discount 0.997, n-step 3, per_alpha 0.2, target replace 500, spectral norm on for CUDA / off for MPS, eps_steps 2M, framestack 4, input 75×140). Wire logging to wandb if `WANDB_API_KEY` is set else CSV to `runs/btr/metrics.csv`. **Critical multi-track change**: modify the rollout loop so each `env.reset()` samples a random track_slug from available savestates. Log per-track episode rewards as separate wandb metrics (`track/{slug}/episode_reward`) so we can spot stalled tracks. Add a `tests/test_btr_smoke.py` that runs 100 random-action steps through a mocked-env stub and verifies the PER sample/update path works. Commit as "phase 2.2: BTR fork".

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
