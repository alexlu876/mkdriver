# Setup

This document covers the one-time setup a developer needs before working on mkw-rl. Run [P-1](docs/PREFLIGHT.md) first — it validates the toolchain — then come back here to finish Phase 0.

## Pinned versions

| Component | Pinned value |
|---|---|
| VIPTankz/Wii-RL submodule | `d8358cbc5feef41161522e51b60fba100506d489` (2026-01-06 master HEAD at time of initial bootstrap) |
| Python | `>=3.13` (via uv; see [.python-version](.python-version)) |
| Game ID | `RMCE01` (NTSC-U MKWii only) |

**⚠️ Submodule SHA is a placeholder.** The spec (§P-1) requires the user to build the fork themselves and report back the SHA they built against. Replace `d8358cb...` with that SHA before Phase 4. For Phase 1-2 work (BC pipeline), the submodule content isn't actually loaded or executed, so the placeholder is safe.

## Prerequisites

1. macOS 14+ on Apple Silicon (M-series). Intel Macs are untested.
2. Xcode command-line tools: `xcode-select --install`.
3. Homebrew: <https://brew.sh>.
4. A legitimate NTSC-U MKWii ISO on disk.

## Install uv

```bash
brew install uv
```

## Clone and sync

```bash
git clone <this-repo-url> mkwii
cd mkwii
git submodule update --init --recursive
uv sync --extra dev
```

Verify:

```bash
uv run python -c "import mkw_rl; print(mkw_rl.__version__)"
# → 0.1.0
uv run pytest
```

## Install VIPTankz's Dolphin scripting fork

This is the external dependency that can't be installed via uv. The fork provides a Python API (`from dolphin import memory, event, savestate, controller, gui`) which the Phase 4 gym wrapper depends on, and `.dtm` recording/replay which the Phase 1 data pipeline depends on.

1. Clone the fork to a working directory **outside** this repo:

   ```bash
   mkdir -p ~/code/mkw
   cd ~/code/mkw
   git clone https://github.com/VIPTankz/Wii-RL.git
   cd Wii-RL
   git checkout d8358cbc5feef41161522e51b60fba100506d489  # matches the submodule pin
   ```

2. Follow the fork's own `README.md` for the Dolphin build. This is macOS-specific and changes as the upstream fork evolves — don't transcribe the steps here, they'll drift.

3. When the build finishes, note the path to the `dolphin-emu` binary and check:

   ```bash
   /path/to/dolphin-emu --version
   ```

   The version string should include the word "scripting" or similar. If it doesn't, you may have built mainline Dolphin by accident — rebuild from the fork's branch.

4. Run the scripting-API probe from [docs/PREFLIGHT.md](docs/PREFLIGHT.md) to confirm the Python bridge works.

## Luigi Circuit savestate

See [docs/SAVESTATE_PROTOCOL.md](docs/SAVESTATE_PROTOCOL.md). The first savestate (`data/savestates/luigi_circuit_tt.sav` + its JSON sidecar) is created manually by the user before Phase 1.

## Running preflight

```bash
uv run python scripts/preflight.py --dolphin /path/to/dolphin-emu
```

The mechanical checks (Python version, binary existence, frame dump dir) will pass/warn automatically. The human-driven steps (fork builds, NTSC-U boots, scripting API works, savestate determinism) are in [docs/PREFLIGHT.md](docs/PREFLIGHT.md) and have to be done by hand.

## What's next

- Complete [P-1](docs/PREFLIGHT.md) and record its output.
- Create the Luigi Circuit savestate per [docs/SAVESTATE_PROTOCOL.md](docs/SAVESTATE_PROTOCOL.md).
- Proceed to Phase 1 via `CLAUDE_CODE_PROMPTS.md` (Prompt 1a).
