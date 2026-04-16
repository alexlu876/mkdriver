# mkw-rl

Behavioral cloning + RL agent for Mario Kart Wii via the Dolphin emulator.

See:

- [MKW_RL_SPEC.md](MKW_RL_SPEC.md) — full project spec.
- [CLAUDE_CODE_PROMPTS.md](CLAUDE_CODE_PROMPTS.md) — ordered prompt sequence for build-out.
- [SETUP.md](SETUP.md) — one-time developer setup.
- [docs/PREFLIGHT.md](docs/PREFLIGHT.md) — mandatory P-1 checklist (run before Phase 0).
- [docs/SAVESTATE_PROTOCOL.md](docs/SAVESTATE_PROTOCOL.md) — how to create reproducible per-track savestates.

## Status

Phase 0 bootstrap complete. See [CHANGES.md](CHANGES.md) for a running log of what was built and what's outstanding.

## Quick start

```bash
brew install uv
git clone <this-repo> && cd mkwii
git submodule update --init --recursive
uv sync --extra dev
uv run pytest
```
