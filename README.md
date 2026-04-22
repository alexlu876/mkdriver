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
- **Phase 2.2 BTR fork**: passes 1–2 complete (helper components + `BTRPolicy` with IMPALA+LSTM+IQN). Pass 3 (R2D2 recurrent replay) in progress.
- **Phase 3+ (training loop, Vast.ai runbook, curriculum)**: pending.

See [CHANGES.md](CHANGES.md) for the running build-out log.

## Quick start

```bash
brew install uv
git clone <this-repo> && cd mkwii
git submodule update --init --recursive
uv sync --extra dev
uv run pytest
```
