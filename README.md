# mkw-rl

Multi-track Beyond-the-Rainbow (BTR) RL agent for Mario Kart Wii via the Dolphin emulator. Behavioral cloning scaffolding is preserved for future augmentation but is not on the active path.

See:

- [MKW_RL_SPEC.md](MKW_RL_SPEC.md) — full project spec.
- [docs/PIVOT_2026-04-17.md](docs/PIVOT_2026-04-17.md) — strategic pivot from BC-first to multi-track BTR.
- [docs/TRAINING_METHODOLOGY.md](docs/TRAINING_METHODOLOGY.md) — v2 training design (LSTM on BTR, variable checkpoints, progress-weighted sampler, lenient reset).
- [docs/REGION_DECISION.md](docs/REGION_DECISION.md) — PAL (RMCP01) region decision.
- [CLAUDE_CODE_PROMPTS.md](CLAUDE_CODE_PROMPTS.md) — ordered prompt sequence for build-out (v3).
- [SETUP.md](SETUP.md) — one-time developer setup.
- [docs/PREFLIGHT.md](docs/PREFLIGHT.md) — mandatory P-1 checklist (run before Phase 0).
- [docs/SAVESTATE_PROTOCOL.md](docs/SAVESTATE_PROTOCOL.md) — how to create reproducible per-track savestates.

## Status

- **P-1 preflight**: complete on Apple Silicon (2026-04-17).
- **Phase 2.1 env fork**: live-smoke-tested against Dolphin on Luigi Circuit (2026-04-21).
- **Phase 2.2 BTR fork**: passes 1–5 complete — helper components, `BTRPolicy` (IMPALA+LSTM+IQN), `PER.sample_sequences()` for R2D2 recurrent replay, `ProgressWeightedTrackSampler` for multi-track curriculum, and a full training loop with Munchausen-IQN loss, checkpoint-resume, graceful shutdown, NaN-bail, and env crash-restart. Post-pass-5 audit findings (2026-04-21) applied inline.
- **Vast.ai runbook**: pending (next deliverable before production run).
- **Phase 3+ (curriculum tuning, multi-env scaling)**: downstream of the Vast.ai runbook.

See [CHANGES.md](CHANGES.md) for the running build-out log.

## Quick start

```bash
brew install uv
git clone <this-repo> && cd mkwii
git submodule update --init --recursive
uv sync --extra dev
uv run pytest

# Smoke-test the full training loop against live Dolphin (~1 min, Luigi Circuit
# only, tiny model). Run this before any real training launch.
uv run python scripts/train_btr.py --config configs/btr.yaml --testing

# Production run (500M env steps — consume Vast.ai compute).
uv run python scripts/train_btr.py --config configs/btr.yaml --device cuda

# Resume from a checkpoint after preemption / crash. Replay buffer is NOT
# checkpointed, so the first ~200K env steps post-resume are random-policy
# warmup (<0.1% of a 500M-frame run). run_name is inferred from the ckpt
# filename so wandb charts + CSV logs stitch across resumes.
uv run python scripts/train_btr.py --config configs/btr.yaml --device cuda \
  --resume runs/btr/btr_20260422_123456_final.pt
```
