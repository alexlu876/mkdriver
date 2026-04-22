# Autonomous build-out: change log

> **⚠️ Strategic pivot 2026-04-17**: the BC-first plan below is superseded. Project skipped BC and jumped to multi-track BTR directly. See [docs/PIVOT_2026-04-17.md](docs/PIVOT_2026-04-17.md). The code produced in Phases 1-2 is preserved as future BC-augmentation scaffolding but is not on the current critical path. Region was also changed from NTSC-U to PAL on the same day — see [docs/REGION_DECISION.md](docs/REGION_DECISION.md) — and propagated across the codebase.

All prompts executed autonomously in one session. Running log of decisions, deviations from spec, outstanding blockers, and things you should review before proceeding.

## Post-pivot passes (BTR, 2026-04-21 → 2026-04-22)

### Phase 2.1 — env fork

`src/mkw_rl/env/dolphin_env.py` + `dolphin_script.py`: master/slave split that launches Dolphin as a subprocess and talks to it over an authenticated Unix domain socket. Gym-compatible `reset(track_slug=...)` / `step(action)` API. Live-smoke-tested against Luigi Circuit on 2026-04-21 — 20-step rollout with reward ticking at checkpoints, clean close.

### Phase 2.2 — BTR fork (5 passes)

1. **Helper components** — `FactorizedNoisyLinear`, cos embedding for IQN, `ImpalaResidualBlock`, `ImpalaLargeEncoder`. Fully-tested leaves before composition.
2. **BTRPolicy** — IMPALA → LSTM (v2 addition) → IQN dueling head with NoisyLinear throughout. Kapturowski-style burn-in hidden-state carry.
3. **PER.sample_sequences()** — R2D2 recurrent replay sampling with seam-rejection for wrapped buffers. Fixes a pre-wrap sequence-seam bug found during the pass-3 audit.
4. **ProgressWeightedTrackSampler** — `weight[slug] = max(progress) - progress[slug] + ε` curriculum that self-corrects as tracks get solved.
5. **Training loop** — `src/mkw_rl/rl/train.py` (`BTRAgent.learn_step` with Munchausen-IQN loss, Dabney eq. 10 quantile Huber, R2D2 priority aggregation η·max + (1-η)·mean), `scripts/train_btr.py` CLI, `configs/btr.yaml` with a `testing:` subtree for smoke runs, and a 27-test unit-test suite for the pass-5 code paths.

### Post-pass-5 audit fixes (2026-04-21)

Multi-agent audit surfaced 6 blockers + 9 high-severity findings. Applied:

- **Target-net noise contract restored** — dropped a spurious `reset_noise()` on the target that contradicted the `disable_noise()` guarantee at build + sync.
- **Target LSTM hidden aligned to n-state** — burn-in now runs on `n_states[:, :burn_in]` so `hidden_target` corresponds to the timestep the target's learning forward actually consumes (was off by `n_step`).
- **Checkpoint-resume path** — `_save_checkpoint()`, `load_checkpoint()`, `--resume` CLI flag; restores online/target/optimizer/counters/sampler EMA. Replay is re-warmed from scratch (docstring explains trade-off).
- **Dolphin crash-restart** — outer loop catches socket EOF / BrokenPipeError / ConnectionResetError / OSError, tears down the env, relaunches, continues. Aborts after 5 consecutive crashes.
- **Graceful shutdown** — SIGTERM handler + KeyboardInterrupt path flip a flag polled at episode boundary; always writes a `*_final.pt` on exit. Second signal bypasses graceful path.
- **NaN/inf bail** — `learn_step` checks `isfinite(loss)` and `isfinite(grad_norm)` before optimizer step; skip + WARN log + abort after 50 consecutive failures.
- **log_every_grad_steps wired** — learn-step metrics emitted mid-episode at configured cadence; was previously parsed from YAML but never consumed.
- **Warmup progress log** — replay fill ratio logged as `replay/capacity`, `replay/fill_ratio` in every episode row; INFO line every ~5% of `min_sampling_size`.
- **CSV logger rewritten** — RFC-4180-compatible; rewrites file + rewrites header when new metric keys appear (was emitting `#`-prefixed malformed headers + silently dropping new columns).
- **sample_sequences retry fix** — rows are refilled in place on invalid hits (up to 20 attempts) instead of re-rolling the whole batch on any single miss. Converges in O(1) retries under tight capacity.
- **Online burn-in noise consistent** — online net's `reset_noise()` now fires above the burn-in so the LSTM warm-up uses the same noise realization as the learning forward.
- **Stdlib `random` seeded** alongside torch + numpy.
- **Env ↔ replay stack-order assertion** at episode start so a regression in frame-stack layout fails loudly.

### Tests

27 new unit tests in `tests/test_btr_training.py` covering: config merge, quantile-Huber (+Dabney axis), Munchausen reward bonus clamping, full loss pathway (including zero-weights → zero-grad, munch_alpha=0 vs 0.9 divergence, all-dones gating), `_CSVLogger` disjoint-key round-trip, `BTRAgent.act` / `sync_target` / `learn_step` (incl. noop-before-warmup + NaN bail + priority writeback), and full checkpoint save → load → match round-trip (weights + counters + sampler EMA). 269 tests total pass.

## Pre-pivot (BC) history — superseded

## TL;DR

- **11 commits**, one per prompt (+ lint cleanups). All on `main`.
- **112 tests passing, 1 skipped** (the skip is the opportunistic REAL_DTM_PATH round-trip test — activate by setting that env var).
- **Full pipeline is CI-green end-to-end on synthetic data**: parse .dtm → pair with frames → build sequence dataset → train BC policy for 2 epochs → eval metrics + side-by-side video. Loss decreases, LSTM grad-norm > 1e-4, all three spec §2.3 diagnostics fire correctly.
- **Phase 2 acceptance criteria met against synthetic data only.** The user-visible acceptance test — sanity visualizer MP4 matching real MKWii kart behavior — still requires you to record a real `.dtm` and watch the output.

## Commits

```
4871244 phase 2.3: BC eval + side-by-side video
992494b phase 2.2: BC training loop with TBPTT
bf3da38 lint: use PEP 695 'type' keyword for LstmState alias
6dc62e1 phase 2.1: BC model (IMPALA CNN + stateful LSTM + mixed heads)
203f497 lint: allow H/W/T/B uppercase variable names (ML convention)
3c06161 phase 1.5: sequence dataset + demo-aware sampler + parse_demo
de245a8 phase 1.4: action encoding
126a0a9 phase 1.3: pairing hardening
1a26a9b phase 1.2: sanity visualizer + minimal pairing + record/replay scaffolding
3de8b03 lint: remove unused imports via ruff --fix
a860a06 phase 1.1: .dtm parser
250d2b4 phase 0: bootstrap
22e8347 preflight scaffolding
```

## Deviations from the spec — decisions I made

1. **uv init in place (not `mkw-rl/` subdirectory).** The spec's `uv init mkw-rl --python 3.13` creates a subdirectory. I kept the existing `mkwii/` directory as the project root since it already had `.git/`, `scripts/preflight.py`, and `docs/PREFLIGHT.md`. Everything works; `uv run pytest` runs from the project root.

2. **Submodule SHA is a placeholder.** Pinned to `d8358cbc5feef41161522e51b60fba100506d489` (master HEAD as of 2026-01-06). **This must be replaced by the SHA you build against in P-1.** The Phase 1-2 code doesn't load the submodule so the wrong-SHA hazard only lands at Phase 4.

3. **Python version: 3.13 (not 3.13.5 exactly).** uv resolved to 3.13.13 (the latest 3.13.x). The v2 spec loosened this from 3.13.5 to ">=3.13" explicitly.

4. **Ruff config: ignore N806 (uppercase `H`, `W`, `T`, `B`).** These are conventional ML names (height, width, timesteps, batch). Fighting them is pointless churn.

5. **Ruff config: line-length 110.** Up from default 88. The spec doesn't mandate a line length; 110 makes the type-hint-heavy code readable without excessive wrapping.

6. **bc.yaml uses decimal floats, not scientific notation.** PyYAML parses `3e-4` as a string; `0.0003` is a float. Noted in the YAML file.

7. **Dataset skipped short demos warn rather than error.** If a demo has fewer samples than `seq_len`, we log a warning and exclude it. The alternative (erroring) would block training whenever one of 20+ demos happened to be too short. Consistent with the existing divergence-warning pattern in pairing.

8. **Training loop is a function (`train_epoch`), not a class.** One fewer layer of indirection for single-epoch debugging. If we need checkpointing-across-epochs state later we can add it without a rewrite.

9. **Sampler exhausts at `min(stream_length)`.** Longer streams' extra chunks get dropped this epoch rather than partial batches being yielded. Standard tradeoff; lets the training loop assume full-width batches.

10. **`is_continuation` is not a dataset output.** It's computed at runtime from `meta` in the training loop. This avoids coupling the dataset to sampler state and makes the API cleaner.

11. **Replay automation is skeleton.** `scripts/replay_demo.py` has a driver template that tries `dolphin.movie.play(...)` — this API may not exist at the pinned SHA. There's a `REPLAY_PROTOCOL.md` manual fallback. See "Outstanding" below.

## Outstanding / need your attention

These are things I could not do autonomously. Listed in the order they'll bite you.

### 🚨 Must do before any real data work

1. **Run P-1 by hand.** Download VIPTankz's pre-compiled Dolphin (`python3 scripts/download_dolphin.py`), boot PAL MKWii, run the scripting-API probe, verify savestate determinism, enable frame dumps. See `docs/PREFLIGHT.md`. Note: region was changed from NTSC-U to PAL on 2026-04-17 — see `docs/REGION_DECISION.md`. Report back:
   - Fork commit SHA actually built.
   - `sys.executable` path from the scripting-API test.
   - `sys.version` string.
   - Any scripting-API import line that differs from `from dolphin import memory, event`.

2. **Replace the submodule SHA pin.** Once P-1 gives you the real SHA, update `.gitmodules` and `SETUP.md`.

3. **Verify `replay_demo.py`'s driver template works against your fork.** The driver calls `from dolphin import movie; movie.play(dtm_path)`. If that API doesn't exist at your SHA, the replay is blocked and you'll need the manual `REPLAY_PROTOCOL.md` procedure until we know the right API.

### 🟠 Required before training on real data

4. **Create the Luigi Circuit savestate.** Follow `docs/SAVESTATE_PROTOCOL.md`. Write the JSON sidecar with the exact VI count — this feeds `skip_first_n` for the pairing step.

5. **Record demos and replay them.** See `scripts/capture_demo.md`. Aim for 20+ Luigi Circuit TT laps. Use `scripts/replay_demo.py` (or the manual protocol) to produce paired frame dumps.

6. **Watch the sanity visualizer MP4.** `scripts/sanity_check.py` produces the Phase 1 acceptance artifact. If the steering / A / R overlays don't match kart behavior, stop and debug pairing before training.

### 🟡 Worth reviewing at your leisure

7. **`replay_demo.py` contains TODOs.** Three explicit `TODO(verify)` comments mark unverified API assumptions. Skim it before running.

8. **The sampler's demo-to-stream distribution matters more with few demos.** If you have fewer demos than `batch_size`, some batch positions stall idle. With 20+ demos and batch_size 16 you're fine. With 2 demos and batch_size 4, not fine. The training loop will surface this via `stats.n_batches == 0`.

9. **MPS vs CPU on M4.** Config default is `cpu`. When you've verified the pipeline trains reasonably on CPU, switch to `mps` via `--device mps` or update `configs/bc.yaml`.

10. **Checkpoint size.** BCPolicy is 3.9M params → checkpoint file ~16MB. `configs/bc.yaml` writes every 5 epochs + best-val. If disk is tight, bump `checkpoint_every`.

11. **wandb is optional.** If `WANDB_API_KEY` isn't set, the logger writes a TSV at `runs/bc/metrics.csv`. No action needed unless you want wandb.

### 🟢 Nice-to-haves I deferred

12. **No real-file round-trip test for the parser.** `tests/test_dtm_parser.py::test_real_dtm_parses` is gated by the `REAL_DTM_PATH` env var. Set it once you have a .dtm on disk.

13. **No online / in-Dolphin BC eval.** Spec §2.4 marks this optional ("Phase 2.5"). The side-by-side video is the human-readable sanity check for now.

14. **Replay driver isn't tested.** We can't CI-test code that needs a running Dolphin binary. The scripting-API template is inferred from VIPTankz's own `DolphinScript.py` but not actually exercised.

## Test coverage summary

```
tests/test_action_encoding.py     15 tests   bins, round-trip, boundaries
tests/test_bc_eval.py              8 tests   metrics, chunked inference, video
tests/test_bc_model.py            12 tests   IMPALA, encoder, LSTM roundtrip, grad flow
tests/test_bc_training.py          6 tests   TBPTT loop, diagnostics, loss decrease
tests/test_dataset.py             19 tests   sequences, frame stack padding, sampler
tests/test_dtm_parser.py          24 tests   headers, buttons, analog, error paths (1 skip)
tests/test_pairing.py             17 tests   skip/trim/divergence/lag warnings
tests/test_pipeline_smoke.py      12 tests   end-to-end synthetic pipeline
─────────────────────────────────────
Total: 113 collected, 112 pass, 1 skip
```

Wall time for the full suite: ~17 seconds on M4 CPU.

## Spec acceptance criteria

| Phase | Acceptance | Status |
|---|---|---|
| P-1 | All 6 preflight checks pass | ⏳ **Blocked on human execution** |
| 0 | `import mkw_rl` succeeds, submodule present, SETUP.md exists, savestate loaded | ✅ Code/docs ready. Savestate blocked on human. |
| 1 | Sanity visualizer MP4 matches reality per user confirmation; tests green; at least 1 demo end-to-end | ⏳ Tests green. MP4 confirmation blocked on real recording. |
| 2 | All three dry-run diagnostics pass; offline eval metrics logged; overlay video produced | ✅ On synthetic data. Real-data confirmation blocked on demos. |

## The checkpoint decision

Per `CLAUDE_CODE_PROMPTS.md`, after Prompt 2c we're supposed to reconvene on four questions before Phase 3:

1. Does the BC model produce visually sensible driving? — **Unanswerable until you have real eval video.**
2. Single-track BC or jump to multi-track? — Defer to after #1.
3. PPO vs BTR for Phase 4? — Spec made action-space compatibility a non-issue (discretized from day one, bridges cleanly to either). Decision is now sample-efficiency / wall-clock driven.
4. Commit Vast.ai compute yet? — Defer until Phase 2 is validated on real data.

**My recommendation**: when you return, run P-1, record demos, produce the sanity-check MP4, train for real on local CPU (should be fast since IMPALA is tiny), then review the side-by-side eval video. That's the minimum required to answer #1 and make the rest of the decisions meaningfully.

## Files to skim before trusting the build-out

In priority order (highest first):

1. [MKW_RL_SPEC.md](MKW_RL_SPEC.md) — if you changed your mind about any fixed assumption, now's the time.
2. [CHANGES.md](CHANGES.md) — this file.
3. [src/mkw_rl/bc/train.py](src/mkw_rl/bc/train.py) — the TBPTT loop. This is the load-bearing code, and correctness here is hard to verify without real data.
4. [src/mkw_rl/dtm/dataset.py](src/mkw_rl/dtm/dataset.py) — frame stacking + demo-aware sampler. Same story: hard to verify without real demos.
5. [scripts/replay_demo.py](scripts/replay_demo.py) — unverified API assumptions.
6. The various `TODO(verify)` / `TODO(phase-1c)` comments (grep: `uv run ruff check --select TD` would surface them but I didn't enable that rule).

## If anything looks off

`git log --oneline` for the prompt-by-prompt history. Each commit message explains rationale. Reverting one prompt's commit will leave the others intact, so bisecting is easy.

`uv run pytest --co -q` lists all collected tests if you want to spot-check coverage.

Welcome back.
