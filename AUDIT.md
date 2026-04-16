# Code audit

Audit of the BC pipeline code (Phase 0 - 2) for bugs, spec violations, and failure modes that will surface once real Dolphin + ISO + demos are available.

**Scope**: every file under `src/`, `scripts/`, `docs/`, `tests/`, plus `pyproject.toml` and root-level docs.

**Method**: re-read every source file; cross-check against `MKW_RL_SPEC.md`, Dolphin's `Movie.cpp` layout (from community references), and VIPTankz's actual `DolphinScript.py` at the pinned SHA for API verification.

**Result**: 112/113 tests pass on synthetic data (1 opportunistic-skip). The pipeline is CI-green end-to-end. But real-data execution will surface **3 high-severity issues that block the record-then-replay workflow**, plus a handful of lower-severity issues worth fixing before spending compute on training.

**This report does not fix anything.** Findings are documented for your review; fixes are your call.

---

## Severity legend

- **🔴 BLOCKER** — code will fail / produce wrong results once real data flows. Must fix before Phase 1 end-to-end.
- **🟠 HIGH** — latent bug that hasn't fired in synthetic tests but will bite under real conditions.
- **🟡 MEDIUM** — correctness/robustness concern; won't block Phase 1 but will bite Phase 3+ or cause silent data issues.
- **🟢 LOW** — minor; document, code hygiene, edge cases.

---

## 🔴 BLOCKERS

### B-1. `scripts/replay_demo.py` uses APIs that don't exist in the fork

**File**: [scripts/replay_demo.py](scripts/replay_demo.py) (the `_DRIVER_TEMPLATE` inside)

The driver template does:

```python
from dolphin import movie         # module does NOT exist in VIPTankz fork
movie.play(DTM_PATH)
while movie.is_playing():
    event.on_frameadvance()
```

**Verification**: grepping `third_party/Wii-RL/DolphinScript.py` confirms the only imports are `from dolphin import event, gui, savestate, memory, controller`. There is no `movie` module. The `ImportError` path in our driver exits with code 2 and does nothing.

The "fallback option 2" mentioned in the TODO comment suggests `controller.set_gcpad_status(...)` — also wrong. VIPTankz's actual API is `controller.set_gc_buttons(port, button_dict)` (see `DolphinScript.py:490`).

**Impact**: `scripts/replay_demo.py` cannot replay a .dtm end-to-end as-is. The `REPLAY_PROTOCOL.md` manual fallback works (via Dolphin's `Movie → Play Input` GUI), but the automation is dead.

**Recommendation**: either delete `replay_demo.py` and rely on the manual protocol until Phase 4 RL introduces its own input-injection primitives, OR rewrite the driver to (a) parse the .dtm with our parser, (b) loop over frames, (c) call `controller.set_gc_buttons(...)` each frame. The second path is ~40 lines of glue code.

---

### B-2. Parser silently misparses multi-controller .dtm files

**File**: [src/mkw_rl/dtm/parser.py](src/mkw_rl/dtm/parser.py) — `_parse_header`, `parse_dtm`

We only check that `has_gcn_port_1` (bit 0) is set. If bits 1, 2, or 3 are ALSO set (multi-GCN), the .dtm body is `8 * num_gcn` bytes per frame, not 8. We read 8-byte chunks and treat each as one frame, effectively reading port-2's data as port-1's data for half the frames. If a Wiimote bit is set (bits 4-7), the body also contains Wiimote state which is variable-length — completely misaligned.

**Impact**: a Time Trial recording (1 GCN controller) works fine. But a Grand Prix or VS recording would silently corrupt all inputs. Phase 3 all-tracks work that uses non-TT modes would hit this.

**Recommendation**: add check that `controllers_bitfield == 0x01` (only GCN port 1, no other GCN ports, no Wiimotes). Raise `DtmFormatError` otherwise. This matches the spec's Fixed Assumption.

**Tests that would catch this if added**: `test_multi_controller_rejected` and `test_wiimote_rejected` in `tests/test_dtm_parser.py`.

---

### B-3. Frame dump ↔ .dtm alignment is fragile due to Dolphin dump ordering

**Files**: [src/mkw_rl/dtm/pairing.py](src/mkw_rl/dtm/pairing.py), [scripts/REPLAY_PROTOCOL.md](scripts/REPLAY_PROTOCOL.md)

Pairing assumes `frame_dump[0]` corresponds to `inputs[0]` (savestate load moment). This is true only if:

1. Dolphin's frame dumper starts producing PNGs exactly at the savestate load.
2. The user enabled Dump Frames AFTER any prior emulation stopped.
3. The dump dir was empty before the session.

REPLAY_PROTOCOL.md step 1 tells the user to enable Dump Frames, then step 3 says "File → Load State." If any emulation is already running when dumping is enabled, the first N PNGs will be pre-savestate frames. Our pairing then mis-aligns input 0 with a pre-savestate frame.

Additionally, **Dolphin's `Movie → Play Input` loads a savestate adjacent to the .dtm automatically if `from_savestate=True` and the .sav file lives next to the .dtm**. Step 3 ("manually load state") may conflict with step 4 ("Play Input").

**Impact**: end-to-end pairing may silently drift by tens of frames. Sanity visualizer will catch this if the user watches it — but they have to know to look for "does frame 0 show the pre-input anchor." The divergence warning threshold of `max(30, 2%)` won't catch a small misalignment.

**Recommendations**:

1. REPLAY_PROTOCOL.md should instruct: "launch Dolphin with NO emulation running, enable Dump Frames, then load the .dtm via Movie → Play Input. Do NOT separately load the savestate."
2. For robustness, place the savestate next to the .dtm with the matching name (`foo.dtm` + `foo.dtm.sav`) so Dolphin's auto-load works.
3. Longer-term: add a timestamp OCR check as spec §1.3 mentions.

---

## 🟠 HIGH

### H-1. Frame dump location may be in a game-ID subdirectory

**Files**: [src/mkw_rl/dtm/frames.py](src/mkw_rl/dtm/frames.py):load_frame_dump, [scripts/replay_demo.py](scripts/replay_demo.py):--dolphin-dump-dir default

Dolphin historically writes frame dumps to `<Dump/Frames>/<GAMEID>/` rather than `Dump/Frames/` directly. Our `load_frame_dump` uses non-recursive `glob("*.png")` — it won't find PNGs in the `RMCE01/` subdir.

Inconsistency: `scripts/preflight.py:check_frame_dump_dir` uses `rglob` (recursive) and WILL find them. So preflight reports PASS but pipeline errors out.

**Recommendation**: either always use `rglob`, or document that the user must point at `Frames/RMCE01/` directly.

---

### H-2. Training "best" checkpoint uses TRAIN loss, not val loss

**File**: [scripts/train_bc.py:175-180](scripts/train_bc.py#L175)

```python
if stats.loss_total < best_loss:
    best_loss = stats.loss_total
    torch.save(..., cfg.log_dir / "bc_best.pt", ...)
```

`stats.loss_total` is the training epoch's loss. As training progresses past the generalization peak, this keeps decreasing while val loss rises. `bc_best.pt` is therefore the last overfit model, not the best model. `scripts/eval_bc.py` loads `bc_best.pt` by default.

Also: val samples are computed but never used — `train_bc.py` builds a train loader only, not a val loader.

**Recommendation**: build a val loader, run a val pass each epoch (no-grad, carried hidden state), track best by val loss. If that's too expensive, at minimum track best by steering loss alone rather than total (steering is the slower metric to overfit).

---

### H-3. `CsvLogger.log()` adds columns mid-file without rewriting header

**File**: [src/mkw_rl/utils/logging.py:35-49](src/mkw_rl/utils/logging.py#L35)

On first `log()`, columns and header are fixed. On subsequent calls, if `metrics` has new keys, they're appended to `self._columns` but the already-written header row is not updated. Data rows for later calls will include values in new positions whose header is missing.

**Impact**: currently benign — we log the same key set every epoch. But the first refactor that adds a new metric will produce a silently corrupted CSV.

**Recommendation**: either enforce fixed columns at construction time (raise on new keys), or rewrite the file with updated headers on column change.

---

### H-4. wandb config may not accept PosixPath

**File**: [scripts/train_bc.py:137](scripts/train_bc.py#L137)

```python
logger = make_logger(..., config=cfg.__dict__, ...)
```

`cfg.__dict__` contains `log_dir: PosixPath('runs/bc')`. wandb.init's `config` param expects JSON-serializable values. Depending on wandb version this either coerces via `str()` silently, skips the key, or raises `TypeError`.

**Recommendation**: sanitize `cfg.__dict__` to serializable types before passing to wandb (e.g., `{k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()}`).

---

### H-5. Hidden state zeros when batch size shrinks

**File**: [src/mkw_rl/bc/train.py:222](src/mkw_rl/bc/train.py#L222)

```python
if hidden[0].shape[1] != B_actual:
    hidden = _hidden_zero_like(model, B_actual, device)
```

When a last-partial batch has B_actual < batch_size, we zero out all hidden state. The continuation from previous batch is lost — if the next epoch's first batch has the same demos in the same stream positions with carried state expected, we miss a chunk's worth of recurrent context.

**Impact**: minor in normal operation (sampler keeps batches uniform via `min(stream_length)`). Will bite if someone swaps in a sampler that yields ragged batches.

**Recommendation**: truncate `hidden` along dim=1 to `B_actual` instead of zeroing. For added correctness, track hidden per stream position, not per tensor-batch-position.

---

## 🟡 MEDIUM

### M-1. LSTM grad norm is a sum-of-per-param-norms, not global L2

**File**: [src/mkw_rl/bc/train.py:239-242](src/mkw_rl/bc/train.py#L239)

```python
for p in model.lstm.parameters():
    if p.grad is not None:
        lstm_grad_norm += float(p.grad.norm().item())
```

This sums per-tensor L2 norms, which is upper-bounded above the true global L2 norm of concatenated gradient vectors (by triangle inequality on individual tensors' contributions). The "> 1e-4" spec threshold is easier to pass than intended.

**Impact**: diagnostic under-reports "gradient dying" conditions. Not a bug per se but calibration drift.

**Recommendation**: use `torch.nn.utils.clip_grad_norm_(..., max_norm=1e9)` which returns the true concatenated norm, or compute it via `sqrt(sum(p.grad.pow(2).sum() for p in model.lstm.parameters() if p.grad is not None))`.

---

### M-2. `torch.manual_seed(cfg.seed)` doesn't seed numpy

**File**: [scripts/train_bc.py:113](scripts/train_bc.py#L113)

`DemoAwareBatchSampler` uses `np.random.default_rng(seed)` for per-epoch demo shuffling. We DO pass `seed` to the sampler, so that's seeded. But if any torch code paths use numpy indirectly (e.g., via PIL resize → some numpy random), they won't be deterministic.

**Impact**: low; training isn't exactly reproducible but close enough. Flag if you care about bit-identical reruns.

**Recommendation**: also call `np.random.seed(cfg.seed)` and `random.seed(cfg.seed)` at entry.

---

### M-3. Eval rebuilds BCPolicy with default config, ignores saved checkpoint config

**File**: [scripts/eval_bc.py:61-62](scripts/eval_bc.py#L61)

```python
model = BCPolicy(BCPolicyConfig(stack_size=args.stack_size))
model.load_state_dict(ckpt["model"])
```

We saved `ckpt["config"]` during training (as `cfg.__dict__`) but don't read it back. If a future TrainConfig exposes `lstm_hidden`/`feature_dim`/etc. and someone trains with non-defaults, eval will build a model with DEFAULT shape and fail `load_state_dict`.

**Impact**: currently benign (TrainConfig doesn't expose model-shape knobs). Fragile against future changes.

**Recommendation**: reconstruct `BCPolicyConfig` from `ckpt["config"]` in eval, or save `BCPolicyConfig` explicitly alongside `TrainConfig` in the checkpoint.

---

### M-4. JSON savestate sidecar is documented but nothing reads it

**Files**: [docs/SAVESTATE_PROTOCOL.md](docs/SAVESTATE_PROTOCOL.md), [scripts/parse_demo.py](scripts/parse_demo.py)

SAVESTATE_PROTOCOL.md mandates a JSON sidecar with `skip_first_n`. `parse_demo.py` takes `--skip-first-n` on the CLI but never reads the sidecar. The user has to keep the two in sync manually.

**Recommendation**: `parse_demo.py` should look for `data/savestates/<track_slug>.json` and pull `skip_first_n` from it. Derive `track_slug` from either a CLI arg or the savestate filename.

---

### M-5. Frame stacking loads the same PNG up to `stack_size` times per chunk

**File**: [src/mkw_rl/dtm/dataset.py:182-191](src/mkw_rl/dtm/dataset.py#L182), [src/mkw_rl/bc/eval.py:121-125](src/mkw_rl/bc/eval.py#L121)

For `seq_len=32, stack_size=4, frame_skip=4`, one `__getitem__` call loads 128 PNGs, but consecutive timesteps share 3/4 of their stack. Naive reload is 3–4× disk read amplification.

**Impact**: I/O-bound at M4-Mini scale. Training will be slower than necessary.

**Recommendation**: sliding-window cache within `__getitem__`: load `seq_len + (stack_size-1)*frame_skip` unique PNGs once, then index into the cache. Roughly 4x faster at seq_len=32.

---

### M-6. REPLAY_PROTOCOL.md ordering doesn't prevent pre-savestate dump frames

**File**: [scripts/REPLAY_PROTOCOL.md](scripts/REPLAY_PROTOCOL.md)

See B-3 for details. Separate issue because REPLAY_PROTOCOL.md is the fix locus, while the underlying fragility is in pairing.py's assumptions.

---

### M-7. PREFLIGHT.md's step-3 scripting probe may use wrong UI path

**File**: [docs/PREFLIGHT.md:92](docs/PREFLIGHT.md#L92)

We say `Scripting → Run Script → ~/code/mkw/scripting_test.py`. VIPTankz actually invokes scripts via CLI flag `dolphin-emu --script <path>` (see `third_party/Wii-RL/DolphinEnv.py:202`). The Scripting menu may or may not exist in their fork.

**Recommendation**: document the CLI path as the primary, GUI as "if the fork has added a menu."

---

### M-8. PREFLIGHT.md step 5 contradicts the record-then-replay discipline

**File**: [docs/PREFLIGHT.md:140-147](docs/PREFLIGHT.md#L140)

Step 5 tells the user to "boot into Luigi Circuit and play with dumping enabled." That's fine for verifying the dumper works, but it directly contradicts `scripts/capture_demo.md` which says "Never record with frame dumping enabled." A first-time user reading PREFLIGHT then capture_demo will be confused.

**Recommendation**: add a note in step 5 that this is a dumper-verification only, not a recording workflow.

---

### M-9. Default `tail_margin=10` aggressive for short-lap demos

**File**: [src/mkw_rl/dtm/pairing.py](src/mkw_rl/dtm/pairing.py)

10 frames at 60fps = 167 ms. For a 90s Luigi Circuit TT (~5400 frames), losing 10 is ~0.2%. Fine. But if someone's demos are shorter than a full lap (e.g., diagnostic short clips), 10 frames may not leave enough signal. Worth surfacing the tradeoff in docs.

---

### M-10. Parser doesn't warn when actual frames differ from header.input_count

**File**: [src/mkw_rl/dtm/parser.py:216-225](src/mkw_rl/dtm/parser.py#L216)

Comment says "we trust the file" if `header.input_count != len(states)`. Silent. The pairing module has a divergence warning for frames-vs-inputs, but the parser itself doesn't flag header/body disagreement.

**Recommendation**: log at WARNING when `len(states) != header.input_count`, with both numbers.

---

### M-11. Parser doesn't warn if `from_savestate == False`

**File**: [src/mkw_rl/dtm/pairing.py](src/mkw_rl/dtm/pairing.py), [src/mkw_rl/dtm/parser.py](src/mkw_rl/dtm/parser.py)

TAS .dtms often don't have `from_savestate=True` — they start from power-on. Pairing assumes savestate anchor (start-alignment). For a power-on-anchored .dtm the anchor is wrong.

**Recommendation**: warn at WARNING in `pair_dtm_and_frames` if `header.from_savestate == False`.

---

### M-12. `WandbLogger` claims lazy init but is eager

**File**: [src/mkw_rl/utils/logging.py:58-64](src/mkw_rl/utils/logging.py#L58)

Docstring says "Initialized lazily on first log()" but `__init__` calls `wandb.init(...)`. Misleading; also means if `make_logger` is called but no `log()` ever fires (e.g., aborted before first epoch), wandb still creates a run.

**Recommendation**: either genuinely defer the `wandb.init` call until first `log()`, or update the docstring.

---

## 🟢 LOW

### L-1. `build_samples_by_demo` is dead code

**File**: [src/mkw_rl/dtm/dataset.py:357-364](src/mkw_rl/dtm/dataset.py#L357). Not called anywhere. Drop it or document.

### L-2. `_frame_sort_key` regex `(\d+)` matches first digit sequence; brittle for odd filenames

**File**: [src/mkw_rl/dtm/frames.py:23](src/mkw_rl/dtm/frames.py#L23). `RMCE01_framedump_0.png` would sort on `01` not `0`. Dolphin's actual naming is benign but document assumption.

### L-3. `test_perfect_prediction` escape valve

**File**: [tests/test_bc_eval.py:59](tests/test_bc_eval.py#L59). Uses `f1 == 1.0 or f1 == 0.0` — passes even if F1 is 0 on some buttons because GT had no positives. Tighten the test by seeding buttons with at least one positive each.

### L-4. `preflight.py check_frame_dump_dir` uses rglob, `frames.py load_frame_dump` uses glob

**Files**: cross-module consistency. See H-1. Mark here for completeness.

### L-5. `scripts/train_bc.py` doesn't expanduser on `demo_glob`

**File**: [scripts/train_bc.py:116](scripts/train_bc.py#L116). `glob("~/data/*")` returns nothing. `Path(...).expanduser()` first if needed.

### L-6. `preflight.py:check_dolphin_binary` marks missing "scripting" marker as WARN

Correct — VIPTankz's fork may not include the word "scripting" in `--version`. But the comment is ambiguous; clarify that this is advisory only.

### L-7. `CsvLogger` overwrites existing log file on every training run

**File**: [src/mkw_rl/utils/logging.py:38](src/mkw_rl/utils/logging.py#L38). Opens with `"w"`. Previous training run's metrics are lost. Timestamp the file or use `"a"`.

### L-8. `bc_loss`'s button list is hardcoded but model's `_BUTTON_NAMES` is also hardcoded

**File**: [src/mkw_rl/bc/model.py](src/mkw_rl/bc/model.py). Two tuples to keep in sync. Minor coupling; fine at current scope.

### L-9. Replay driver calls `event.on_frameadvance()` before any emulation is running

**File**: [scripts/replay_demo.py](scripts/replay_demo.py) `_DRIVER_TEMPLATE`. The call may hang indefinitely waiting for a frame that never comes. Related to B-1.

### L-10. `DtmHeader.has_gcn_port_1` is a property but not tested explicitly

**File**: [tests/test_dtm_parser.py](tests/test_dtm_parser.py). Tested via the GCN-port-1 validation, but no dedicated property test. Minor.

### L-11. Tests don't cover the REAL_DTM_PATH opportunistic round-trip on a known-good fixture

Once you record a first .dtm, add it to the repo (if small) or to a private fixtures path, and exercise `REAL_DTM_PATH=... uv run pytest`. Catches parser regressions against real-world bit patterns.

### L-12. Sampler has O(D*N) setup per epoch via `chunks_for_demo`

**File**: [src/mkw_rl/dtm/dataset.py:165-167](src/mkw_rl/dtm/dataset.py#L165). Fine for current scale; precompute to O(D+N) if multi-track data grows this large.

### L-13. No test reproduces the "LSTM carries across chunks in training" invariant

`test_chunked_matches_single_pass` in `test_bc_eval.py` covers inference, not the training loop's hidden-state plumbing. Worth a direct test: run `train_epoch` on deterministic synth data with known hidden-state expectations.

---

## Spec alignment check

Going line-by-line against `MKW_RL_SPEC.md`:

| Spec section | Status |
|---|---|
| §P-1 preflight checklist | ✅ Implemented in `docs/PREFLIGHT.md` + `scripts/preflight.py` (see M-7, M-8 for improvements) |
| §0.1 uv init | ✅ |
| §0.2 Dolphin install instructions | ✅ `SETUP.md` |
| §0.3 submodule pinned | ✅ (placeholder SHA; user must replace after P-1) |
| §0.4 savestate protocol with VI count sidecar | ⚠️ Protocol documented; sidecar JSON not consumed by pipeline (M-4) |
| §1.1 .dtm parser | ✅ (single-controller only; see B-2) |
| §1.2 record-then-replay | ⚠️ Capture protocol correct; replay automation broken (B-1); manual replay ordering fragile (B-3) |
| §1.3 frames loader | ⚠️ Non-recursive glob misses game-ID subdirs (H-1) |
| §1.4 pairing hardened | ✅ (M-11 minor gap on from_savestate warning) |
| §1.5 sanity visualizer | ✅ |
| §1.6 sequence dataset | ✅ (M-5 performance) |
| §1.7 action encoding | ✅ |
| §2.1 data collection, user vs TAS separation | ✅ Documented; separation enforced at directory level, not code level |
| §2.2 BC model IMPALA + stateful LSTM | ✅ |
| §2.3 TBPTT training with 3 diagnostics | ✅ (M-1 grad norm is over-reported; H-2 val loss not actually validated) |
| §2.4 BC eval with side-by-side video | ✅ |

No spec section is materially un-implemented. The gaps are around robustness (B-1, B-3, H-1) and observability (M-4, M-11) rather than missing features.

---

## Will it run bug-free on real data?

**Honestly: no, not without addressing at least the blockers.** Specifically:

1. **B-1 will fail on first attempt** to run `replay_demo.py`. The user WILL need to fall back to `REPLAY_PROTOCOL.md` immediately.
2. **B-3 is a silent bug** — the user may produce a paired dataset that looks reasonable but is offset by tens of frames. Only the sanity visualizer catches this, and only if the user knows to watch the first frame specifically.
3. **H-1 frame dump subdir** is likely to bite: Dolphin's default for MKWii is typically `.../Dump/Frames/RMCE01/`. User points `--frames` at the wrong dir, gets "no PNGs," wastes time.

After those three, the data pipeline is sound. Training will run correctly for one demo on synthetic-scale data — this is already proven by `tests/test_bc_training.py`. The remaining HIGH issues (H-2, H-3, H-4, H-5) affect training quality and reproducibility but not crashes.

---

## Recommended order of operations when you return

1. **Run P-1 by hand.** (No changes needed here.)
2. **Replace submodule SHA pin** with the one you built. Edit `.gitmodules` and `SETUP.md`.
3. **Fix B-3** (REPLAY_PROTOCOL.md ordering) before recording any real demos. A 3-line doc change.
4. **Fix B-2** in the parser (reject multi-controller). ~5 LOC.
5. **Record your first demo** per `capture_demo.md`.
6. **Before running replay** — decide: fix B-1 in `replay_demo.py` driver (~40 LOC) or use manual protocol only (0 LOC).
7. **Fix H-1** (frame loader recursive glob) before running `parse_demo.py`. 1-line change.
8. **Run `sanity_check.py` and watch the MP4.** This is the real Phase 1 gate.
9. If the MP4 looks right: produce N demos, parse, train, eval.
10. Between demos and training: consider M-4 (sidecar consumption) to reduce manual typing errors.
11. Before declaring Phase 2 done: consider H-2 (val loss tracking) and M-3 (config preservation in eval).

---

## Summary

- **13 commits in the repo, 112/113 tests passing.**
- **3 blockers** (B-1, B-2, B-3) that will surface on first real-data run.
- **5 high-severity** issues (H-1 through H-5) that degrade training reliability.
- **12 medium-severity** issues around observability, reproducibility, performance.
- **13 low-severity** cosmetic/edge-case issues.

Total: 33 flagged items. None of them revert the architectural design or require a rewrite. The bulk are small, targeted patches — roughly one hour of focused fixes would address every BLOCKER and HIGH.

The test suite is a fair but incomplete safety net. It caught what synthetic data can catch. It cannot catch B-1, B-3, H-1, M-4, M-7, M-8, M-11 — those require either real data or API verification that we can't do from a CI-only environment. Keep that gap in mind when deciding whether to fix blockers before or after first real-data run.

**Nothing here should scare you.** The architecture is sound. The pipeline is correct in the parts that have been tested against real-ish stimuli (full TBPTT loop with hidden state, parser against hand-built .dtms, dataset + sampler). The blockers are concentrated at the Dolphin-boundary layer, which is exactly where they're hardest to verify without your hardware — and exactly where the spec explicitly called out "unverified" and "skeleton" during prompt-by-prompt build-out.
