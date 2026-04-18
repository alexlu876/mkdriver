# Preflight checklist (P-1)

This checklist runs **before** Phase 0 of the project (see [MKW_RL_SPEC.md](../MKW_RL_SPEC.md) §P-1). Its job is to verify that VIPTankz's Dolphin scripting fork distribution actually runs on your machine, boots Mario Kart Wii, exposes a working Python scripting API, and produces the artifacts (savestates, frame dumps) that the downstream pipeline depends on.

You (the human operator) run this yourself. Claude Code cannot do it — several steps require controlling a GUI emulator and visually confirming on-screen behavior.

Only proceed to Phase 0 after **all six checks pass** and you've filled in the "Report back" section at the bottom.

> **ROM-region decision: PAL (`RMCP01`).** See [REGION_DECISION.md](REGION_DECISION.md) for the rationale. The project follows VIPTankz's PAL setup. Savestates come from their `scripts/download_savestates.py` rather than being built manually (the manual protocol in [SAVESTATE_PROTOCOL.md](SAVESTATE_PROTOCOL.md) is now a fallback / Phase 3 tool).

---

## Before you start

- Have your own legitimate PAL Mario Kart Wii ISO on disk (`mkw.iso`, `RMCP01`, MD5 `e7b1ff1fabb0789482ce2cb0661d986e`, 4.38 GB). Place it at `~/code/mkw/Wii-RL/game/mkw.iso` after you've cloned the repo in Step 1.
- Have Xcode command-line tools installed: `xcode-select --install`.
- Have Homebrew installed: <https://brew.sh/>.
- Have **Python 3.13.5 specifically** installed — via `uv python install 3.13.5`, **not** via Homebrew. (VIPTankz's README says `brew install python@3.13.5` but that's not a real Homebrew formula; `brew install python@3.13` gives you whatever 3.13.x Homebrew currently ships, which may not be 3.13.5.) The pre-compiled Dolphin's embedded Python is 3.13.5 exactly and hardcodes the stdlib prefix — see Step 3 for the symlink workaround.
- Have ~10 GB free disk space for Dolphin + a few frame dumps.

**macOS 26.x caveat**: on macOS Tahoe (26.x) and newer, building Dolphin from source is broken (AGL was removed). The pre-compiled download path in Step 1 is the **only** viable install method. Do not try the fallback build script.

---

## Step 1 — Dolphin distribution ready on your M4 Mac

VIPTankz ships a pre-compiled arm64 Dolphin (sourced from `unexploredtest/dolphin`) that you download via their script. Building from source is now the fallback, not the primary path — `scripts/build-dolphin-mac.sh` in their repo carries a comment noting it doesn't work on macOS 26+ because AGL was removed.

**What to do (primary — pre-compiled download):**

```bash
# Clone somewhere convenient (NOT inside this project dir).
mkdir -p ~/code/mkw && cd ~/code/mkw
git clone https://github.com/VIPTankz/Wii-RL.git
cd Wii-RL
git rev-parse HEAD   # record the repo SHA (below)

# Create a Python 3.13 venv for VIPTankz's Python deps.
uv venv --python 3.13 .venv
source .venv/bin/activate

# Use requirements_cpu.txt on macOS — requirements.txt pins torch==2.7.1+cu118
# which has no macOS wheel.
uv pip install -r requirements_cpu.txt

# download_dolphin.py needs requests + tqdm which aren't in either
# requirements file. Install them separately.
uv pip install requests tqdm

# Download pre-compiled Dolphin.
python3 scripts/download_dolphin.py
```

That places a `dolphin0/` directory next to the repo root containing `DolphinQt.app` (the macOS app bundle). Record that path.

**Fallback — build from source (only if download fails on pre-macOS-26):**

```bash
bash scripts/build-dolphin-mac.sh
```

⚠️ This script is broken on macOS 26+ (uses removed AGL framework). On modern macOS the download path is mandatory.

**How to verify:**

```bash
open ~/code/mkw/Wii-RL/dolphin0/DolphinQt.app
```

should launch the Dolphin GUI. If it fails to launch, or Gatekeeper blocks the binary, resolve that before continuing. (Typical fix: System Settings → Privacy & Security → "Open Anyway" after the first block.)

**Record for reporting:**

- [ ] VIPTankz/Wii-RL repo SHA (`git rev-parse HEAD` in their repo): `__________________________________________`
- [ ] Path to `DolphinQt.app`: `________________________________`
- [ ] Install method: `downloaded` / `built-from-source`
- [ ] macOS version (`sw_vers`): `__________________________________________`

---

## Step 2 — Dolphin boots PAL MKWii

**What to do:** launch `DolphinQt.app`, open your PAL ISO, confirm you reach the title screen with Koopa / Mario announcer audio. Check the Dolphin log window (Tools → Log Configuration → enable "Game"): it should show `Game ID: RMCP01`. Any other game ID means you're running the wrong ISO — stop.

**How to verify:**

- Title screen visible.
- Log shows `Game ID: RMCP01`.
- Can reach the main menu (Single Player → Time Trial → Luigi Circuit is loadable). You don't need to actually race yet.

**Record for reporting:**

- [ ] Title screen reached on first boot.
- [ ] Log shows `RMCP01`.
- [ ] Luigi Circuit Time Trial is loadable.

---

## Step 3 — Python scripting API works

This is the load-bearing check. If it fails, the whole plan is wrong.

### Critical prerequisite — the Python stdlib symlink

Before anything else, you need to work around a compile-time hardcoding in the pre-compiled Dolphin. The embedded Python dylib has this path baked in as its stdlib prefix:

```
/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13
```

This path doesn't exist on your machine (Homebrew has no `python@3.13.5` formula; only `python@3.13` which ships whatever 3.13.x is current). Without the stdlib present there, you'll see:

```
E[Scripting]: Failed to initialize python from config: Failed to import encodings module
E[Scripting]: ModuleNotFoundError: No module named 'traceback'
```

and the probe script will never execute. `PYTHONHOME` is **not** respected — Dolphin uses compiled-in `PyConfig` which overrides env vars.

**Fix (one-time, reversible):**

```bash
# Install 3.13.5 specifically via uv (real 3.13.5 with matching ABI).
uv python install 3.13.5

# Symlink the expected Homebrew path to uv's install.
UVROOT=$(dirname "$(dirname "$(uv python find 3.13.5)")")
TARGET=/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13
mkdir -p "$(dirname "$TARGET")"
ln -s "$UVROOT" "$TARGET"

# Verify the stdlib is reachable.
ls "$TARGET/lib/python3.13/encodings" | head -3
```

If `/opt/homebrew/Cellar/` is root-owned on your system (unusual on Apple Silicon), prepend `sudo` to the `mkdir` and `ln -s`.

### The probe script

Save this as `~/code/mkw/scripting_test.py`:

```python
"""Scripting-API probe + Python linkage report.

Note: event.on_frameadvance takes a callback function — it is
callback-registration, NOT a blocking "sleep one frame" primitive.
"""

import sys
import traceback

print(f"[preflight] sys.executable = {sys.executable}", flush=True)
print(f"[preflight] sys.version = {sys.version}", flush=True)

try:
    from dolphin import memory, event
    print("[preflight] imported dolphin.memory and dolphin.event OK", flush=True)
except Exception:
    print("[preflight] IMPORT FAILED:", flush=True)
    print(traceback.format_exc(), flush=True)
    raise

_frame_count = 0
_done = False

def on_frame() -> None:
    global _frame_count, _done
    _frame_count += 1
    if _frame_count == 60 and not _done:
        _done = True
        addr = 0x80000000
        val = memory.read_u8(addr)
        print(f"[preflight] memory.read_u8(0x{addr:08x}) = 0x{val:02x}", flush=True)
        print("[preflight] probe complete", flush=True)

event.on_frameadvance(on_frame)
print("[preflight] registered on_frameadvance callback", flush=True)
```

### Launching Dolphin with the script

Use the **direct binary invocation** (not `open --args ...`) — it keeps stdout attached to your terminal so the probe's prints are visible:

```bash
~/code/mkw/Wii-RL/dolphin0/DolphinQt.app/Contents/MacOS/DolphinQt \
    --no-python-subinterpreters \
    --script ~/code/mkw/scripting_test.py \
    --exec=$HOME/code/mkw/Wii-RL/game/mkw.iso
```

Let the game boot (takes ~1-2 seconds of emulation to reach frame 60). The seven `[preflight]` lines should print to your terminal; close Dolphin once you see `probe complete`.

(VIPTankz's own code launches via `open --args` per `DolphinEnv.create_dolphin`, but that detaches stdout and hides useful errors. Prefer direct invocation for the probe; use `open --args` only for their env's production launch path.)

**Things that can go wrong:**

- `Failed to import encodings module` → you skipped the stdlib symlink above.
- `TypeError: function takes exactly 1 argument (0 given)` on `event.on_frameadvance()` → you're calling it as a blocking wait instead of registering a callback. Use the callback pattern shown above.
- `ImportError` on `from dolphin import …` → the fork's API names drifted. Check `~/code/mkw/Wii-RL/DolphinScript.py` for the current import line.
- Silent (nothing prints) when launching via `open --args ...` → macOS's `open` detaches stdout. Use the direct binary path instead.
- Crash → report the backtrace.

**Record for reporting:**

- [ ] Seven `[preflight]` lines printed (including `memory.read_u8(0x80000000) = 0xXX`).
- [ ] Adjusted import line (if different from template): `_________________________`

---

## Step 4 — Python linkage determined

The Step 3 probe already prints `sys.executable` and `sys.version`. Capture those values here.

**On VIPTankz's pre-compiled Dolphin**, the expected output is:

- `sys.executable = <path to DolphinQt binary itself>` (Python is statically embedded into the Dolphin binary).
- `sys.version = 3.13.5 (main, Jun 11 2025, 15:36:57) [Clang 16.0.0 (clang-1600.0.26.6)]`.

That's the fixed, compiled-in interpreter — there's no "pointing Dolphin at a different Python." The only lever you have is making the stdlib available at the hardcoded path (Step 3's symlink).

**Implication for Phase 0's training-side venv**: it's a separate process, so it's independent of Dolphin's embedded interpreter. You can use any Python 3.13.x via uv. The one thing that matters for cross-process compatibility is `DolphinEnv.py`'s `shared_site.txt` mechanism — when you run their env, it writes the training-side Python's `site.getsitepackages()` to a file, and Dolphin's embedded Python appends that path to `sys.path`. The C extensions (numpy, torch, PIL) in that forwarded site-packages must be ABI-compatible with Dolphin's 3.13.5. Within 3.13.x the stable ABI mostly works, but if you see import errors from C extensions during Phase 2 training, pin your training venv to 3.13.5 exactly (`uv venv --python 3.13.5`).

**Record for reporting:**

- [ ] `sys.executable` output: `__________________________________________`
- [ ] `sys.version` output: `__________________________________________`

---

## Step 5 — Frame dump produces PNGs

**Where the toggle actually lives**: this fork moved the frame-dump enable out of `Config → Graphics → Advanced` (only the output settings — resolution type, PNG compression level — live there). The on/off toggle is under the **Movie** menu in the top menu bar: `Movie → Start Frame Dump` (or keyboard shortcut).

**Where the output goes**: this distribution ships `portable.txt` in the `dolphin0/` directory, which tells Dolphin to use its own `User/` subdirectory as the config/data root. Frame dumps therefore land in:

```
~/code/mkw/Wii-RL/dolphin0/User/Dump/Frames/
```

**not** the default macOS path `~/Library/Application Support/Dolphin/Dump/Frames/`.

**Layout**: this fork writes frames flat (`framedump_0.png`, `framedump_1.png`, …) directly in `Frames/`, not under a `RMCP01/` game-id subdirectory like some Dolphin versions. Our `load_frame_dump` uses `rglob` so both layouts work transparently.

**What to do:**

1. Launch Dolphin normally (no `--script` this time): `open ~/code/mkw/Wii-RL/dolphin0/DolphinQt.app`.
2. Boot `mkw.iso` → Single Player → Time Trial → Luigi Circuit.
3. Enable frame dumping via `Movie → Start Frame Dump`.
4. Let it run for ~10 seconds of actual gameplay (countdown + first few seconds of driving is fine).
5. `Emulation → Stop` (this also stops the dump).

**How to verify:**

```bash
find ~/code/mkw/Wii-RL/dolphin0/User/Dump/Frames -name "*.png" | wc -l
# Expect ~500+ PNGs per 10s of gameplay at native speed.
```

Or run the mechanical wrapper with an explicit `--frames-dir`:

```bash
uv run python scripts/preflight.py \
    --dolphin ~/code/mkw/Wii-RL/dolphin0/DolphinQt.app \
    --frames-dir ~/code/mkw/Wii-RL/dolphin0/User/Dump/Frames
# Should print [PASS] frame-dump-dir with a PNG count.
```

**Note — speed during dumping**: frame dumping slows emulation on M4. This is expected and the reason §1.2 of the spec separates record-and-replay. Do not worry if the game feels sluggish during this check.

**Important — this is a dumper verification only, not the actual demo recording workflow.** Our demo pipeline NEVER records `.dtm` and dumps frames simultaneously — dumping slows emulation below realtime, which makes recorded inputs reflect slow-speed play rather than clean-speed play. See `scripts/capture_demo.md` (record clean, no dumps) and `scripts/REPLAY_PROTOCOL.md` (replay with dumps afterwards). This P-1 step is just making sure the dumper produces PNGs at all.

**Record for reporting:**

- [ ] At least one frame dump session produced PNGs.
- [ ] PNG count from `scripts/preflight.py`: `__________`

---

## Step 6 — Savestate save/load works deterministically

**What to do:**

1. Boot Luigi Circuit Time Trial. Press `F1` (or your configured save-state hotkey) mid-race to save state to slot 1.
2. Let the race run for another ~2 seconds so the kart moves visibly.
3. Press `F5` (or configured load hotkey) to reload slot 1.
4. The kart should jump back to exactly the position it was in when you saved.
5. Repeat load 2-3 times to confirm determinism — same load, same resulting frame.

**How to verify:** visual confirmation that load produces a reproducible, identical post-load frame. If the kart lands in slightly different positions on repeat loads, determinism is broken and something is wrong with the fork's savestate code — stop and report.

**Record for reporting:**

- [ ] Savestate save/load works.
- [ ] Reloading the same slot produces visually identical post-load state across at least 3 trials.

---

## Report back

After all six checks pass, reply to the user-facing conversation (or fill in `docs/PREFLIGHT_REPORT.md` if working asynchronously) with:

```
P-1 complete.

  Fork commit SHA: <SHA>
  Dolphin binary path: <path>
  Python sys.executable: <path>
  Python version: <version>
  Platform: <Darwin arm64 / other>
  Scripting API probe: PASS (or list issues)
  Frame dump count: <N>
  Savestate determinism: PASS (or list issues)

  Adjusted scripting API import (if template needed changes):
    <line>

  Notable gotchas hit during P-1:
    <free-form notes>
```

Those values feed directly into Prompt 0's `--python` arg and the `<SHA>` pin in `.gitmodules`. Don't proceed to Phase 0 without them.

If any step **fails** rather than passes: stop. The project is not ready for Phase 0. Common failure modes:

- Build fails on M4 → the fork may not have been tested on Apple Silicon. File an issue upstream or look for a community patchset. Do not work around it locally.
- Scripting API probe errors on import → their API drifted. Check their README / examples and update the template script in this doc before Phase 0.
- Savestate determinism broken → savestates are load-bearing for the entire RL pipeline. Cannot proceed.

---

## What this file does *not* cover

These are verified later, not now:

- `.dtm` recording / parsing — Phase 1.
- Replay-with-frame-dump workflow — Phase 1.
- RAM addresses for lap / checkpoint / position — Phase 4.
- Multi-instance Dolphin — Phase 4.

If you start poking at any of those during P-1 and something breaks, that's fine but not a blocker for Phase 0. Note it and move on.
