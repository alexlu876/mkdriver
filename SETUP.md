# Setup

This document covers the one-time setup a developer needs before working on mkw-rl. Run [P-1](docs/PREFLIGHT.md) first — it validates the toolchain — then come back here to finish Phase 0.

## Pinned versions

| Component | Pinned value |
|---|---|
| VIPTankz/Wii-RL submodule | `d8358cbc5feef41161522e51b60fba100506d489` (2026-01-06 master HEAD at time of initial bootstrap) |
| Python | `>=3.13` (via uv; see [.python-version](.python-version)) |
| Game ID | `RMCP01` (PAL MKWii only — see [docs/REGION_DECISION.md](docs/REGION_DECISION.md)) |

**⚠️ Submodule SHA is a placeholder.** The spec (§P-1) requires the user to build the fork themselves and report back the SHA they built against. Replace `d8358cb...` with that SHA before Phase 4. For Phase 1-2 work (BC pipeline), the submodule content isn't actually loaded or executed, so the placeholder is safe.

## Prerequisites

1. macOS 14+ on Apple Silicon (M-series). Intel Macs are untested.
2. Xcode command-line tools: `xcode-select --install`.
3. Homebrew: <https://brew.sh>.
4. A legitimate PAL MKWii ISO on disk (`mkw.iso`, MD5 `e7b1ff1fabb0789482ce2cb0661d986e`, 4.38 GB). See [docs/REGION_DECISION.md](docs/REGION_DECISION.md) for why we follow VIPTankz's PAL setup.

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

## Install VIPTankz's Dolphin scripting distribution

This is the external dependency that can't be installed via uv. VIPTankz wraps Felk's scripting fork and ships a pre-compiled arm64 macOS build via `scripts/download_dolphin.py`. The distribution provides a Python API (`from dolphin import memory, event, savestate, controller, gui`) which the Phase 4 gym wrapper depends on, and `.dtm` recording/replay which the Phase 1 data pipeline depends on.

1. Clone the repo to a working directory **outside** this project:

   ```bash
   mkdir -p ~/code/mkw
   cd ~/code/mkw
   git clone https://github.com/VIPTankz/Wii-RL.git
   cd Wii-RL
   git checkout d8358cbc5feef41161522e51b60fba100506d489  # matches the submodule pin
   ```

2. Install their Python deps and download pre-compiled Dolphin:

   ```bash
   pip install -r requirements.txt
   python3 scripts/download_dolphin.py
   ```

   That places `dolphin0/DolphinQt.app` next to the repo root. Building from source (`bash scripts/build-dolphin-mac.sh`) is only a fallback — their build script explicitly doesn't work on macOS 26+.

3. Run the scripting-API probe from [docs/PREFLIGHT.md](docs/PREFLIGHT.md) to confirm the Python bridge works. The app-bundle invocation is `open dolphin0/DolphinQt.app --args --script <your-script>.py --exec=<your-mkw.iso>`.

## Luigi Circuit savestate

See [docs/SAVESTATE_PROTOCOL.md](docs/SAVESTATE_PROTOCOL.md). The first savestate (`data/savestates/luigi_circuit_tt.sav` + its JSON sidecar) is created manually by the user before Phase 1.

## Running preflight

```bash
uv run python scripts/preflight.py --dolphin /path/to/dolphin-emu
```

The mechanical checks (Python version, binary existence, frame dump dir) will pass/warn automatically. The human-driven steps (Dolphin launches, PAL MKWii boots, scripting API works, savestate determinism) are in [docs/PREFLIGHT.md](docs/PREFLIGHT.md) and have to be done by hand.

## What's next

- Complete [P-1](docs/PREFLIGHT.md) and record its output.
- Create the Luigi Circuit savestate per [docs/SAVESTATE_PROTOCOL.md](docs/SAVESTATE_PROTOCOL.md).
- Proceed to Phase 1 via `CLAUDE_CODE_PROMPTS.md` (Prompt 1a).
