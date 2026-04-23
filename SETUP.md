# Setup

This document covers the one-time setup a developer needs before working on mkw-rl. Run [P-1](docs/PREFLIGHT.md) first — it validates the toolchain — then come back here to finish Phase 0.

## Pinned versions

| Component | Pinned value |
|---|---|
| VIPTankz/Wii-RL submodule | `d8358cbc5feef41161522e51b60fba100506d489` (2026-01-06 master HEAD at time of initial bootstrap) |
| Python | `>=3.13` (via uv; see [.python-version](.python-version)) |
| Game ID | `RMCP01` (PAL MKWii only — see [docs/REGION_DECISION.md](docs/REGION_DECISION.md)) |

**⚠️ Submodule SHA is a placeholder.** The spec (§P-1) requires the user to build the fork themselves and report back the SHA they built against. Replace `d8358cb...` with that SHA before live training runs (new Phase 2; was Phase 4 pre-pivot). For the dormant BC pipeline the submodule content isn't actually loaded or executed, so the placeholder is safe.

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

This is the external dependency that can't be installed via uv. VIPTankz wraps Felk's scripting fork and ships a pre-compiled arm64 macOS build via `scripts/download_dolphin.py`. The distribution provides a Python API (`from dolphin import memory, event, savestate, controller, gui`) which the gym wrapper in the active training path (`src/mkw_rl/env/`) depends on, and `.dtm` recording/replay which the dormant BC demo pipeline (`src/mkw_rl/dtm/`, `src/mkw_rl/bc/`) depends on.

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

## Linux / Vast.ai setup (production training)

The pipeline also runs on Linux CUDA hosts (Vast.ai). The env module auto-detects platform and switches between Darwin (`.app` bundle) and Linux (direct binary + `QT_QPA_PLATFORM=offscreen` for headless). Before launching:

1. Build or install Dolphin-with-scripting on the target host. The binary must accept `--script` and `--no-python-subinterpreters` flags (the same fork VIPTankz ships). Common binary names we auto-detect: on Linux `dolphin-emu-nogui` (headless, preferred), `DolphinQt`, `dolphin-emu`.
2. Install `xvfb` (`apt install xvfb`). Required on Linux — `dolphin-emu-nogui` still initializes a Qt platform plugin + video backend that need an X display; we wrap the binary in `xvfb-run` automatically. Without xvfb, Dolphin fails with "No X11 display found / No platform found."
3. Place your PAL MKWii ISO somewhere writable.
4. Uncomment + set the env paths in `configs/btr.yaml`:

   ```yaml
   env:
     env_id: 0
     dolphin_app: "/opt/dolphin"            # directory containing the binary, OR direct binary path
     iso: "/opt/mkw/mkw.iso"
     mkw_rl_src: "/workspace/mkwii/src"     # absolute path to this repo's src/ dir
   ```

5. Copy the savestate bundle to `data/savestates/` on the host (the glob is intersected with `data/track_metadata.yaml` at build time, so only slugs present in both will be sampled).

6. Install CUDA runtime libs with the Linux extra — torch 2.11+cu128 dlopens `libcusparseLt` / `libnvshmem_host` but doesn't pull them as a hard dep, so a plain `uv sync --extra dev` will `import torch`-fail on a fresh Linux CUDA host:

   ```bash
   uv sync --extra dev --extra linux-cuda
   ```

7. Prepend torch's bundled cuDNN to `LD_LIBRARY_PATH` (Vast.ai's PyTorch template ships cuDNN 9.8 system-wide; torch 2.11 was built against 9.19 and bundles the right version in its venv). Add to your shell's rc file:

   ```bash
   export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
   ```

8. Launch — **invoke the venv python directly, not via `uv run`**:

   ```bash
   WANDB_API_KEY=… .venv/bin/python scripts/train_btr.py --config configs/btr.yaml --device cuda
   ```

   Running through `uv run` on Vast.ai silently prevents Dolphin's embedded scripting engine from initializing — the emulator boots fine and burns ~700% CPU running the game, but the Python slave never executes, so the master socket never sees a connection and training stalls at the header row. Direct `.venv/bin/python` launches produce identical results to `uv run` for all other code but fix scripting. Reason not fully nailed down (likely `uv run`'s env/fd manipulation interacting with Dolphin's embedded CPython init via `subprocess.Popen`); bypassing `uv run` is the only reliable workaround found.

   Dolphin's stdout/stderr is captured to `{log_dir}/dolphin_env_0.log` for post-mortem diagnosis. Checkpoints land in `{log_dir}/{run_name}_grad{N}.pt` with rotation keeping the newest 5; `_final.pt` / `_diverged.pt` are never pruned. Resume with `--resume {path} [--run-name {name}]`.

## What's next

- Complete [P-1](docs/PREFLIGHT.md) and record its output.
- Create the Luigi Circuit savestate per [docs/SAVESTATE_PROTOCOL.md](docs/SAVESTATE_PROTOCOL.md).
- Proceed to Phase 1 via `CLAUDE_CODE_PROMPTS.md` (Prompt 1a).
