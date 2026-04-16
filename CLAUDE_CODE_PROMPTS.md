# Claude Code prompt sequence (v2)

Feed these to Claude Code one at a time. Each prompt assumes Claude Code has read `MKW_RL_SPEC.md` in full. For each prompt after the first, start the session with: *"Re-read MKW_RL_SPEC.md before starting."*

Each prompt is designed to produce one reviewable PR. Do not chain prompts; wait for a clean result before moving on.

**What changed from v1**: Prompt P-1 (preflight) is new and mandatory before Prompt 0. Prompt 1d (action encoding) is new. Prompts 2a and 2b have been updated for IMPALA CNN, discretized steering, and TBPTT.

---

## Prompt P-1 — Preflight (user-run, Claude Code authors the checklist only)

> Read MKW_RL_SPEC.md §P-1. Generate `scripts/preflight.py` (a thin wrapper that runs the mechanical parts of the checklist — e.g., verifying a RAM read succeeds from a running Dolphin scripting session) and `docs/PREFLIGHT.md` with the full six-step human checklist. Do NOT attempt to run any of it; the user runs P-1 on their own machine. Report completion and wait for the user to confirm all six checks passed and report back the Dolphin fork commit SHA + Python interpreter path that the scripting API actually links against. Commit as "preflight scaffolding".

**User then runs P-1 manually and reports back the commit SHA and Python path. Update MKW_RL_SPEC.md §0.1 and §0.3 with those values before Prompt 0.**

---

## Prompt 0 — Bootstrap

> Re-read MKW_RL_SPEC.md. The user has completed P-1 and reported back `<SHA>` for the Dolphin fork and `<python-path>` for the interpreter. Execute Phase 0 in full: init the repo with `uv` (using `<python-path>` if it differs from a uv-managed 3.13), install dependencies as specified in §0.1, create the directory layout from the "Repo layout" section, add the VIPTankz/Wii-RL submodule **pinned to `<SHA>`**, write `SETUP.md` with the macOS Dolphin fork install steps and the pinned SHA prominently recorded. Create `docs/SAVESTATE_PROTOCOL.md` with the Luigi Circuit savestate procedure from §0.4, including the requirement to record the exact VI count in a JSON sidecar. Stop and report status; do not proceed to Phase 1.

---

## Prompt 1a — `.dtm` parser

> Re-read MKW_RL_SPEC.md §1.1. Implement `src/mkw_rl/dtm/parser.py` with the `ControllerState` and `DtmHeader` dataclasses and `parse_dtm()` function. Header must surface `from_savestate`, `vi_count`, `input_count`, `lag_count`. Write `tests/test_dtm_parser.py` alongside with hand-constructed binary blobs (hold-A, hold-left, truncated, wrong game_id). Use `@pytest.mark.skipif(not path.exists())` for any test that requires a real `.dtm` file. **Before finalizing the parser, read `Source/Core/Core/Movie.cpp` in the VIPTankz submodule to resolve any TASVideos-wiki ambiguity about field widths or VI/input count semantics.** Verify by running `uv run pytest`. Commit as "phase 1.1: .dtm parser".

---

## Prompt 1b — Sanity visualizer (intentionally built before full pairing)

> Re-read MKW_RL_SPEC.md §1.3, §1.4, and §1.5. Implement `src/mkw_rl/dtm/frames.py` (`load_frame_dump` + `load_frame`), `src/mkw_rl/dtm/viz.py` (the overlay renderer), and a **minimal** `src/mkw_rl/dtm/pairing.py` that aligns from the **start** (frame 0 ↔ input frame 0, both truncated to `min(len)` from the start with no `skip_first_n` or tail margin yet). Leave a TODO pointing at §1.4 for the full alignment logic. Create `scripts/sanity_check.py` that takes a `.dtm` + frame dir as args and outputs an MP4 of the first 30 seconds with overlays. Also generate `scripts/capture_demo.md` (user-facing recording protocol from §1.2 — record first, replay with dumps second, NEVER both at once) and `scripts/replay_demo.py` (or `REPLAY_PROTOCOL.md` if scripting API limitations block automation). Stop and wait for the user to run sanity_check against a real recording and visually confirm correctness before moving to Prompt 1c.

---

## Prompt 1c — Pairing hardening

> Re-read MKW_RL_SPEC.md §1.4. The visualizer confirmed alignment works. Now harden `pairing.py` per the full §1.4 strategy: `skip_first_n` read from the savestate JSON sidecar, `tail_margin` default 10, length-divergence warning threshold `max(30, 0.02 * len(inputs))`, `lag_count > 0` warning, alignment from the start. Add `tests/test_pairing.py` covering: empty sequences, length mismatch warnings, `skip_first_n` truncation, `tail_margin` truncation, and divergence past threshold. Commit as "phase 1.2: pairing hardening".

---

## Prompt 1d — Action encoding

> Re-read MKW_RL_SPEC.md §1.7. Implement `src/mkw_rl/dtm/action_encoding.py` with `N_STEERING_BINS = 21`, `encode_steering`, `decode_steering`. Write `tests/test_action_encoding.py` covering round-trip error bounds, edge values (-1, 0, 1), and inverse consistency. Commit as "phase 1.3: action encoding".

---

## Prompt 1e — Sequence dataset

> Re-read MKW_RL_SPEC.md §1.6. Implement `src/mkw_rl/dtm/dataset.py` with the `MkwBCDataset` class that returns fixed-length sequences with `is_continuation` metadata, respecting demo boundaries (no splicing across demos). Also implement a `DemoAwareBatchSampler` that pins each batch position to a demo and consumes windows in order. Write `scripts/parse_demo.py` that takes a `.dtm` + frame dir, runs parser → pairing → action encoding → dataset, and pickles the result to `data/processed/user_demos/`. Add pytest coverage for dataset edge cases: demo shorter than `seq_len`, stacking padding at demo start, `is_continuation` flag correctness. Commit as "phase 1.4: sequence dataset".

---

## Prompt 2a — BC model (IMPALA + stateful LSTM)

> Re-read MKW_RL_SPEC.md §2.2. Implement `src/mkw_rl/bc/model.py` with the `BCPolicy` class per the architecture diagram: IMPALA-style CNN encoder (3 blocks with max-pool + 2 residual blocks each, 16/32/32 channels, 4-channel grayscale input), applied per-timestep over a `(B, T, 4, H, W)` input; LSTM with `hidden_dim=512`, 1 layer, stateful; heads for 21-bin steering (linear to 21 logits), and four binary buttons (linear to 1 logit each). `forward(frames, hidden) -> (logits_dict, new_hidden)`. Write `tests/test_bc_model.py` with: output shape smoke test at `(B=2, T=8, 4, 114, 140)`; hidden state round-trip determinism (same input + hidden → same output); parameter count sanity check (<5M params). Commit as "phase 2.1: BC model".

---

## Prompt 2b — BC training loop with TBPTT

> Re-read MKW_RL_SPEC.md §2.3. Implement `src/mkw_rl/bc/train.py` and `scripts/train_bc.py` + `configs/bc.yaml`. Key requirements: (1) TBPTT loop that detaches hidden state after each backward, (2) hidden state reset only at demo boundaries per the `is_continuation` flag, never at arbitrary batch boundaries, (3) AdamW + cosine schedule + grad clipping at 1.0, (4) wandb logging if `WANDB_API_KEY` is set else CSV, (5) checkpoint every N epochs and best-val-loss. Run the dry-run test specified in §2.3: 1 epoch on the user's processed demos. **Report back all three sanity diagnostics** explicitly: (a) per-bin steering CE decreasing across bins (not just bin 10), (b) A-button training F1 > 0.6, (c) LSTM gradient norm > 1e-4. If any of the three fail, stop and ask before iterating. Commit as "phase 2.2: BC training".

---

## Prompt 2c — BC eval

> Re-read MKW_RL_SPEC.md §2.4. Implement `src/mkw_rl/bc/eval.py` + `scripts/eval_bc.py`. Compute offline metrics (steering top-1 and top-3 accuracy, per-button F1, joint accuracy with ±1 bin tolerance) on a held-out demo. Produce a side-by-side overlay video: left pane = ground-truth controller state (decoded from the held-out demo's action tensors), right pane = predicted controller state (model run with carried hidden state across the full held-out demo, not reset per-window). Commit as "phase 2.3: BC eval".

---

## Checkpoint — do not proceed past this without user review

After Prompt 2c succeeds, **stop and reconvene with the user** before starting Phase 3. Decisions to make at this checkpoint:

1. Does the BC model produce visually sensible driving on the eval video?
2. Is single-track BC good enough, or do we jump to Phase 3 (multi-track) before Phase 4 (RL)?
3. Which RL algo for Phase 4: PPO or BTR? Both can consume the discretized action output from §1.7, so the bridge is cleaner than v1 implied — decision is now driven by sample efficiency and wall-clock on the target compute, not action-space compatibility.
4. Is the user ready to commit compute (Vast.ai) for Phase 4, or still iterating locally?
5. Is an online smoke test (Phase 2.5, §2.4) warranted before Phase 4, or do offline metrics look unambiguous?

These decisions change the Phase 3/4 spec meaningfully. Do not let Claude Code pick them unilaterally.

---

## Prompts 3+ (stubs — re-spec before use)

- **Prompt 3** — Multi-track data collection + training. Requires 32 savestates (all with documented VI counts) and demo pool. Substantial user work upstream. Architecture is unchanged from Phase 2 (stateful LSTM already supports multi-track).
- **Prompt 4a** — Fork VIPTankz env into `src/mkw_rl/env/`. Verify RAM reads match expected values on NTSC-U.
- **Prompt 4b** — Reward function + gym.Env wrapper. Factored vs joint action space decision made here.
- **Prompt 4c** — RL fine-tune with BC init. Algo choice made at the checkpoint above.
- **Prompt 5** — Retro Rewind support.
- **Prompt 6** — Autoresearch / arch search.

Re-read MKW_RL_SPEC.md and update the appendix stubs before writing prompts 3+.
