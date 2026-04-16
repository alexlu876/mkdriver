# Preflight checklist (P-1)

This checklist runs **before** Phase 0 of the project (see [MKW_RL_SPEC.md](../MKW_RL_SPEC.md) §P-1). Its job is to verify that the VIPTankz Dolphin scripting fork actually builds, boots NTSC-U Mario Kart Wii, exposes a working Python scripting API, and produces the artifacts (savestates, frame dumps) that the downstream pipeline depends on.

You (the human operator) run this yourself. Claude Code cannot do it — several steps require controlling a GUI emulator and visually confirming on-screen behavior.

Only proceed to Phase 0 after **all six checks pass** and you've filled in the "Report back" section at the bottom.

---

## Before you start

- Have your own legitimate NTSC-U Mario Kart Wii ISO (`RMCE01`) on disk. No PAL / NTSC-J / Korean copies.
- Have Xcode command-line tools installed: `xcode-select --install`.
- Have Homebrew installed: <https://brew.sh/>.
- Have ~20 GB free disk space for Dolphin's build artifacts and a few frame dumps.

---

## Step 1 — Fork builds on your M4 Mac

**What to do:**

```bash
# Clone somewhere convenient (NOT inside this project dir).
mkdir -p ~/code/mkw && cd ~/code/mkw
git clone https://github.com/VIPTankz/Wii-RL.git
cd Wii-RL
# Follow their README's macOS build instructions exactly. DO NOT freelance.
# When the build completes, record the exact commit SHA you built against.
git rev-parse HEAD
```

**How to verify:** `dolphin-emu --version` (or whatever the fork's build places on PATH) runs and prints a version string. If it crashes, errors, or can't be found, stop and debug the build before continuing.

**Record for reporting:**

- [ ] Fork commit SHA: `__________________________________________`
- [ ] Path to the built `dolphin-emu` binary: `________________________________`
- [ ] Output of `dolphin-emu --version`: `________________________________`

---

## Step 2 — Fork boots NTSC-U MKWii

**What to do:** launch the fork's Dolphin, open your NTSC-U ISO, confirm you reach the title screen with Koopa / Mario announcer audio. Check the Dolphin log window (Tools → Log Configuration → enable "Game"): it should show `Game ID: RMCE01`. Any other game ID means you're running the wrong ISO — stop.

**How to verify:**

- Title screen visible.
- Log shows `Game ID: RMCE01`.
- Can reach the main menu (Single Player → Time Trial → Luigi Circuit is loadable). You don't need to actually race yet.

**Record for reporting:**

- [ ] Title screen reached on first boot.
- [ ] Log shows `RMCE01`.
- [ ] Luigi Circuit Time Trial is loadable.

---

## Step 3 — Python scripting API works

This is the load-bearing check. If it fails, the whole plan is wrong.

**What to do:** VIPTankz's fork exposes a Python scripting API with the import pattern `from dolphin import event, gui, savestate, memory, controller` (verified against their `DolphinScript.py` at the pinned SHA — see [SETUP.md](../SETUP.md)). Run the minimal test script below.

Save this as `~/code/mkw/scripting_test.py`:

```python
"""Minimal scripting-API probe. Prints one RAM byte and exits.

Verified against VIPTankz/Wii-RL @ d8358cb... — if the fork's API has
drifted by the time you run this, adapt per their README.
"""

from dolphin import memory, event

def main():
    # Wait until the game has fully booted past the title screen.
    event.on_frameadvance()
    # Arbitrary address in MEM1. Any address in 0x80000000-0x817FFFFF
    # is valid; the exact value doesn't matter for this test — we only
    # care that the read returns without error.
    addr = 0x80000000
    val = memory.read_u8(addr)
    print(f"[preflight] memory.read_u8(0x{addr:08x}) = 0x{val:02x}")

main()
```

Then run Dolphin with the script via CLI (this is VIPTankz's documented invocation path — see `third_party/Wii-RL/DolphinEnv.py`):

```bash
/path/to/dolphin-emu --script ~/code/mkw/scripting_test.py
```

The game should advance one frame and the printed line should appear in the Dolphin log / console. (If the fork has also added a `Scripting → Run Script` menu in the UI, that works too, but the CLI is the primary path.)

**Things that can go wrong:**

- Import error → the fork's API names differ from the template above. Read their README, fix the imports, retry.
- Python version mismatch → see Step 4.
- Silent hang → the API's `event.on_frameadvance()` may not be the right primitive; check their examples.
- Crash → report the backtrace back to the user before continuing.

**Record for reporting:**

- [ ] Scripting API probe prints the expected line without error.
- [ ] Adjusted import line (if different from template): `_________________________`

---

## Step 4 — Python linkage determined

This tells us whether Phase 0's `uv init` can use uv-managed Pythons or must pin to a system/Homebrew interpreter.

**What to do:** from inside the scripting test (step 3), add one extra line before the memory read:

```python
import sys
print(f"[preflight] sys.executable = {sys.executable}")
print(f"[preflight] sys.version = {sys.version}")
```

Re-run the script. The printed path tells us which Python the fork embedded/linked against.

**Possible outcomes:**

| What `sys.executable` prints | What it means | Phase 0 implication |
|---|---|---|
| `/opt/homebrew/bin/python3.13` or similar | Fork uses Homebrew Python | `uv init --python /opt/homebrew/bin/python3.13` |
| `/usr/bin/python3` | Fork uses system Python | Pin to that in Phase 0; do not use uv-managed Python |
| Path inside the Dolphin app bundle | Fork embeds its own Python | Pin to that; uv venv will need `--python` override |
| A uv / conda path | Uv-managed Python is compatible | Use uv default in Phase 0 |

**Record for reporting:**

- [ ] `sys.executable` output: `__________________________________________`
- [ ] `sys.version` output: `__________________________________________`

---

## Step 5 — Frame dump produces PNGs

**What to do:**

1. In the fork's Dolphin: Config → Graphics → Advanced. Enable:
   - `Dump Frames`
   - `Dump Frames as Images` (forces PNG over AVI)
2. Boot into Luigi Circuit Time Trial. Let it run for ~10 seconds of actual gameplay (the countdown + first few seconds of driving is fine).
3. Stop the game.
4. Look in the frame dump directory (usually `~/Library/Application Support/Dolphin/Dump/Frames/`). There should be hundreds of sequentially numbered `.png` files.

**How to verify:**

```bash
ls ~/Library/Application\ Support/Dolphin/Dump/Frames/ | head
# Should see a sequence like framedump_0.png, framedump_1.png, ...
```

Or run the mechanical wrapper:

```bash
python scripts/preflight.py
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
