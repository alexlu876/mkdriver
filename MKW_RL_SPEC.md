# Mario Kart Wii RL Agent — Project Spec (v2)

This document is the working spec for an RL agent that plays Mario Kart Wii via the Dolphin emulator. It is written to be consumed by Claude Code as a long-running reference. Read this entire document before making architectural decisions.

> **⚠️ Strategic pivot 2026-04-17: BC path deferred.** The project is now going direct to multi-track BTR with a **track-agnostic policy**, skipping behavioral cloning. See [docs/PIVOT_2026-04-17.md](docs/PIVOT_2026-04-17.md) for rationale and implications. Phase 1 (`src/mkw_rl/dtm/`) and Phase 2 (`src/mkw_rl/bc/`) are complete and tested (124 tests green) but **dormant** — they stay as future BC-augmentation scaffolding. The new critical path is Phase 2 (renumbered from old Phase 4): fork VIPTankz's env + BTR into `src/mkw_rl/env/` and `src/mkw_rl/rl/`, train on all 32 vanilla tracks with no track-id conditioning, on Vast.ai.

**Revision note (v2)**: This spec supersedes v1. Changes from v1 are concentrated in §P-1 (pre-bootstrap sanity checks), §1.2 (record-then-replay), §1.3 (alignment from start), §1.5 (sequence-returning dataset), and §2.2 (IMPALA CNN + discretized steering + stateful LSTM with TBPTT). Rationale for each change is inline. Sections on `.dtm` parsing, pairing, action encoding, and BC training are preserved for future BC augmentation work; they are not on the current critical path.

---

## Attribution & prior work

This project is a layered extension of existing work. Credit where due:

- **VIPTankz / AI-Tango** — [`Wii-RL`](https://github.com/VIPTankz/Wii-RL) and [`BTR`](https://github.com/VIPTankz/BTR). Their Dolphin scripting fork, environment wrapper, savestate-based reset mechanic, RAM-reading reward function, and Luigi Circuit single-track PPO/BTR baseline are the foundation we fork from. Their paper (Beyond The Rainbow, ICML 2025 Poster, [arXiv:2411.03820](https://arxiv.org/abs/2411.03820)) is the algorithmic baseline.
- **TASVideos / heinermann** — [`.dtm` format spec](https://tasvideos.org/EmulatorResources/Dolphin/DTM) and [DTMText](https://github.com/heinermann/DTMText) are the canonical references for `.dtm` parsing.
- **Dolphin project** — [`Source/Core/Core/Movie.cpp`](https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/Core/Movie.cpp) is ground truth for `.dtm` semantics when the TASVideos wiki is ambiguous. **When in doubt, read Movie.cpp, not the wiki.** The wiki has drifted from the source in known places.
- **MKWii hacking community** — RAM address tables for lap / checkpoint / position / velocity are community-maintained. Cite the specific source in code comments when you use an address.

Our contribution (what makes this more than a fork):

1. **Track-agnostic generalization across all 32 vanilla MKWii tracks** with a single policy, no track-id conditioning. VIPTankz published only a single-track (Luigi Circuit) BTR result; multi-track generalization in MKWii RL is, as of 2026-04, unpublished. This is research territory (see `docs/PIVOT_2026-04-17.md` for the honest risk assessment).
2. **Behavioral cloning augmentation from TAS `.dtm` demonstrations** (deferred — `src/mkw_rl/dtm/` and `src/mkw_rl/bc/` are built and tested but dormant). Hypothesis on hold: BC augmentation accelerates RL convergence and enables approaching TAS-quality Time Trials. Re-examined only after a working multi-track BTR baseline exists.
3. **Extension to Retro Rewind custom tracks** (Phase 5 stretch — blocked on region; Retro Rewind is NTSC-U and we are PAL).
4. **Autoresearch stretch** (Phase 6, miolini/autoresearch-macos) — attribution deferred until implementation.

---

## Fixed assumptions

Do not revisit these without explicit user approval.

- **Game region**: PAL only. Game ID `RMCP01`. Do not support NTSC-U / NTSC-J / Korean. Decision recorded in [docs/REGION_DECISION.md](docs/REGION_DECISION.md) — we follow VIPTankz's PAL setup to get their RAM addresses, savestates, and pre-trained model for free. (Trade-off: Retro Rewind is NTSC-U-only, so Phase 5 is out of scope unless we flip regions later.)
- **Controller**: Emulated GameCube controller (GCN), via `.dtm` GameCube controller section. 8 bytes per frame. No Wii Remote / Classic Controller / Nunchuck paths.
- **Dolphin build**: VIPTankz's scripting fork, pinned by commit SHA (not branch-tracked). Mainline Dolphin is not acceptable — we need their Python scripting API for RAM reads, screen grabs, savestate manipulation, and input injection.
- **Python**: 3.13 on macOS, matching VIPTankz's macOS fork requirement. Target interpreter is determined in §P-1 (may be Homebrew `python@3.13` or uv-managed standalone, depending on Dolphin fork linkage). `uv` is used for dependency management and virtualenvs. Do not use `conda`, `poetry`, or raw `pip+venv`.
- **Frame processing**: 140×75 grayscale, 4-frame stack. Matches VIPTankz's MKWii BTR implementation (`DolphinEnv.py:91-92`, `BTR.py:1258`). Do not change without a written reason.
- **Frameskip**: 4 (BTR convention, Atari-style).
- **Bring-up track**: Luigi Circuit for initial smoke testing (matches VIPTankz), then all 32 vanilla tracks simultaneously with **track-agnostic policy** (no track-id input). See `docs/PIVOT_2026-04-17.md` for why we jumped to multi-track early.
- **Recording discipline** (dormant): `.dtm` is always recorded with frame dumping *disabled*, then replayed with frame dumping *enabled* to produce paired frames. Never record with both at once. See §1.2. Applies only to the BC-augmentation future work.
- **Pairing direction** (dormant): alignment from the start of the savestate anchor, with `skip_first_n` for menu/countdown frames and tail-trimming for stop-recording ragged edge. See §1.3. Applies only to the BC-augmentation future work.
- **Policy architecture**:
  - **Active path (BTR)**: IMPALA-style CNN encoder + **stateful LSTM (hidden=512, 1 layer)** + IQN/Munchausen/PER/NoisyNet dueling heads. Discrete 40-way action space per VIPTankz. LSTM is added on top of VIPTankz's published frame-stack-only `BTR.py` because v2 methodology (see `docs/TRAINING_METHODOLOGY.md` §2) identified LSTM as the single biggest change that made multi-track generalization work. Replay uses burn-in-prefixed sequences (R2D2 pattern). See §4 and `docs/TRAINING_METHODOLOGY.md`.
  - **Dormant path (BC)**: same IMPALA encoder + stateful LSTM + discretized 21-bin steering + per-button binary heads. See §2.2. Shares the encoder + LSTM with BTR so a future BC-to-BTR warm-start is direct (load encoder+LSTM weights, swap only the heads).
- **Compute**: M4 Mac Mini (16GB) for code iteration, small smoke tests (Luigi Circuit only), and local multi-env debugging. **Vast.ai (RTX 4090 class) for real training runs** — committed upfront given the track-agnostic multi-track scope. Do not assume a local GPU for anything beyond smoke tests.

---

## Repo layout (greenfield)

```
mkw-rl/
├── README.md
├── pyproject.toml              # uv-managed, Python 3.13
├── uv.lock
├── .python-version             # "3.13"
├── .gitignore
├── src/mkw_rl/
│   ├── __init__.py
│   ├── dtm/                    # Phase 1: .dtm parsing & BC data pipeline
│   │   ├── __init__.py
│   │   ├── parser.py           # binary .dtm → structured ControllerState per frame
│   │   ├── frames.py           # Dolphin frame dump → frame index
│   │   ├── pairing.py          # align .dtm inputs ↔ rendered frames from savestate start
│   │   ├── dataset.py          # PyTorch Dataset: sequence-returning, TBPTT-friendly
│   │   ├── action_encoding.py  # steering discretization / inverse
│   │   └── viz.py              # overlay controller state on frame for sanity check
│   ├── bc/                     # Phase 2: behavioral cloning
│   │   ├── model.py            # IMPALA CNN + stateful LSTM + mixed action heads
│   │   ├── train.py            # TBPTT training loop, sequence-aware sampler
│   │   └── eval.py
│   ├── env/                    # Phase 4: gymnasium.Env wrapping VIPTankz's fork
│   │   ├── __init__.py
│   │   ├── dolphin_env.py
│   │   ├── ram.py              # RAM address tables, per-region (PAL only)
│   │   ├── reward.py           # reward shaping
│   │   └── savestates.py
│   ├── rl/                     # Phase 4: BTR / PPO
│   │   ├── btr.py
│   │   └── ppo.py
│   ├── multitrack/             # Phase 3: all-32-tracks support
│   └── utils/
│       ├── logging.py
│       └── config.py
├── scripts/
│   ├── preflight.py            # §P-1: verify Dolphin fork + Python linkage before Phase 0
│   ├── capture_demo.md         # user-facing guide for recording .dtm (no dump)
│   ├── replay_demo.py          # replay .dtm with frame dump enabled → paired frames
│   ├── parse_demo.py           # end-to-end: .dtm + frame dump → sequence dataset
│   ├── sanity_check.py         # runs viz.py over a few frames to verify alignment
│   └── train_bc.py
├── data/
│   ├── raw/
│   │   ├── demos/              # .dtm files (user-recorded)
│   │   ├── tas/                # .dtm files (TAS community, attribution tracked)
│   │   └── frames/             # frame dumps (gitignored, regeneratable from .dtm)
│   ├── processed/              # packed sequence datasets
│   └── savestates/             # .sav files per track, with exact VI count documented
├── configs/
│   ├── bc.yaml
│   └── btr.yaml
├── tests/
│   ├── test_dtm_parser.py
│   ├── test_pairing.py
│   ├── test_action_encoding.py
│   └── test_bc_model.py
└── third_party/
    └── Wii-RL/                 # git submodule, pinned to specific SHA
```

**Key layout notes:**

- `third_party/Wii-RL` is a git submodule **pinned to a specific commit SHA**, not tracking `main`. Upstream drift is a known hazard; we accept the cost of manual update. Record the pinned SHA in `SETUP.md`.
- `data/` is gitignored. Frame dumps are regeneratable from `.dtm` + savestate via `scripts/replay_demo.py`, so we don't need to back them up. `.dtm` and savestates are the only source-of-truth artifacts.
- `data/raw/demos/` and `data/raw/tas/` are separate intentionally — see §2.1 for why user and TAS recordings are not mixed naively.
- `configs/` uses YAML, loaded via a thin `utils/config.py` wrapper. No Hydra, no Lightning config nonsense.

---

## Phase P-1 — Pre-bootstrap sanity checks (do this before Phase 0)

**Rationale**: v1 committed to VIPTankz's macOS fork and Python 3.13.5 under uv as Fixed Assumptions, without verifying they compose. Dolphin scripting forks historically have been Linux/Windows-first; "macOS support" in a README is often aspirational. If the fork doesn't build or doesn't link against a uv-managed Python, it's much cheaper to find out before we lay down uv + pyproject.toml than after.

Claude Code should generate `scripts/preflight.py` and a `docs/PREFLIGHT.md` checklist that the **user** runs. Do not attempt any of these checks from Claude Code directly — they require a human at the keyboard.

The checklist must verify, in order:

1. **Fork builds on the user's M4 Mac.** User clones VIPTankz/Wii-RL, follows their README to build the Dolphin fork, and confirms `dolphin-emu --version` reports the scripting build. Record the commit SHA they built against.
2. **Dolphin boots PAL MKWii.** User loads their own RMCP01 ISO and reaches the title screen. Confirms game ID in Dolphin logs.
3. **Python scripting API works.** Run a minimal script (provided in `docs/PREFLIGHT.md`) that starts Dolphin, reads one RAM address, and exits cleanly. This is the load-bearing check — if this fails, the whole plan is wrong.
4. **Python linkage determined.** Identify which Python the scripting fork loads: embedded, Homebrew, or system. This determines whether uv-managed Pythons are usable or we must pin to `/opt/homebrew/bin/python3.13`.
5. **Frame dump produces PNGs.** User enables `Dump Frames` + `Dump Frames as Images`, runs the game for ~10s, confirms PNGs land in `~/Library/Application Support/Dolphin/Dump/Frames/`.
6. **Savestate save/load works.** User makes a savestate, loads it, confirms determinism (same inputs → same outcome).

**Exit criterion for P-1**: all six checks pass. `docs/PREFLIGHT.md` is updated with the actual commit SHA, Python path, and any platform-specific gotchas the user hit. Only then proceed to Phase 0.

If P-1 fails on step 1 or 2, stop the project and reconvene with the user. Don't try to work around a broken fork; debug it first.

---

## Phase 0 — Bootstrap

### 0.1 Project init

```bash
uv init mkw-rl --python 3.13
cd mkw-rl
uv add torch torchvision numpy pillow pyyaml mss pygame gymnasium \
       stable-baselines3 opencv-python tqdm wandb imageio[ffmpeg]
uv add --dev pytest pytest-cov ruff
git init && git add . && git commit -m "init"
```

If P-1 step 4 determined uv-managed Pythons don't link, replace `--python 3.13` with `--python /opt/homebrew/bin/python3.13` (or whatever P-1 identified) and note this deviation in `SETUP.md`.

### 0.2 Dolphin fork install instructions

Claude Code generates `SETUP.md` with the exact steps from VIPTankz's README **using the commit SHA pinned in P-1**, adapted for macOS. Include the platform-specific gotchas surfaced in P-1.

Claude Code: do NOT attempt to install or build Dolphin itself. Generate instructions; the user has already run them in P-1.

### 0.3 Submodule (pinned)

```bash
git submodule add https://github.com/VIPTankz/Wii-RL third_party/Wii-RL
cd third_party/Wii-RL
git checkout <SHA-from-P1>
cd ../..
git add .gitmodules third_party/Wii-RL
git commit -m "pin VIPTankz/Wii-RL to <SHA-from-P1>"
```

Record the SHA in `SETUP.md` prominently. Do not use `git submodule update --remote` in this repo; that would track `main` and break the pin.

### 0.4 Savestate for Luigi Circuit

User must manually:
1. Boot MKWii (PAL, `RMCP01`) in VIPTankz's Dolphin distribution.
2. Select Time Trial → Luigi Circuit → Standard Kart, Mario.
3. Save state at a **specific, reproducible frame**. The simplest reproducible anchor is: the first frame after "GO!" disappears and kart inputs apply. Find this by frame-stepping from the "3" of the countdown. Record the exact VI count (from Dolphin's frame counter) in `docs/SAVESTATE_PROTOCOL.md`.
4. Export the state to `data/savestates/luigi_circuit_tt.sav`. Also export the exact VI count to `data/savestates/luigi_circuit_tt.json` as `{"vi_count": <N>, "track": "luigi_circuit", "game_id": "RMCP01"}`.

**Why the exact VI count matters**: savestate encodes RNG. Savestates made at slightly different frames have different RNG and will produce slightly different behavior under the same inputs. For multi-track consistency and reward reproducibility, every per-track savestate must be made at a documented, reproducible VI count.

Claude Code: generate `docs/SAVESTATE_PROTOCOL.md` with this procedure so it's reproducible per track later. The VI count field is mandatory, not optional.

---

## Phase 1 — `.dtm` + frame pipeline

**Goal**: Given a `.dtm` + matching Dolphin frame dump (produced by replay, not live record), produce a PyTorch `Dataset` that returns **sequences** of `(stacked_frames, controller_state)` pairs suitable for TBPTT, with a sanity-check visualizer confirming alignment.

### 1.1 `.dtm` parser (`src/mkw_rl/dtm/parser.py`)

Reference: [TASVideos DTM spec](https://tasvideos.org/EmulatorResources/Dolphin/DTM) and `Source/Core/Core/Movie.cpp`. **When the wiki and `Movie.cpp` disagree, `Movie.cpp` wins.** Known drift points: header field widths, semantics of `vi_count` vs `input_count` around lag frames. Read the source.

**Header** (first 0x100 bytes, little-endian):

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0x000 | 4 | signature | must equal `b"DTM\x1a"` — validate |
| 0x004 | 6 | game_id | must equal `b"RMCP01"` — validate, raise `DtmRegionError` on mismatch |
| 0x00A | 1 | is_wii | must be 1 for MKWii |
| 0x00B | 1 | controllers_bitfield | bits 0-3 = GC ports 1-4, bits 4-7 = Wiimotes. Must have bit 0 set (GC port 1). |
| 0x00C | 1 | from_savestate | flag; surface this on the returned header — TAS files often have this unset |
| 0x00D | 8 | vi_count | total VIs |
| 0x015 | 8 | input_count | total input samples |
| 0x01D | 8 | lag_count | |
| 0x025 | 8 | reserved | skip |
| 0x02D | 4 | rerecord_count | |
| 0x031 | 32 | author | UTF-8, NUL-padded |
| ... | | | see full table in TASVideos spec *and* `Movie.cpp` |
| 0x100 | — | controller_data | starts here |

**GameCube controller data** (8 bytes per frame, starting at offset 0x100):

| Byte | Bit(s) | Field |
|---|---|---|
| 0 | 0 | Start |
| 0 | 1 | A |
| 0 | 2 | B |
| 0 | 3 | X |
| 0 | 4 | Y |
| 0 | 5 | Z |
| 0 | 6 | D-Up |
| 0 | 7 | D-Down |
| 1 | 0 | D-Left |
| 1 | 1 | D-Right |
| 1 | 2 | L (digital) |
| 1 | 3 | R (digital) |
| 1 | 4 | disc_change (console, not relevant) |
| 1 | 5 | reset (console, not relevant) |
| 1 | 6 | controller_connected |
| 1 | 7 | origin_reset |
| 2 | 0-7 | L pressure (0-255) |
| 3 | 0-7 | R pressure (0-255) |
| 4 | 0-7 | Analog X (1-255; 128 = neutral) |
| 5 | 0-7 | Analog Y (1-255; 128 = neutral) |
| 6 | 0-7 | C-stick X (1-255) |
| 7 | 0-7 | C-stick Y (1-255) |

**⚠️ Bit numbering gotcha**: Bits are numbered LSB = bit 0. "Start at byte 0 bit 0" = `byte & 0x01`. Write an explicit bit-extraction helper, unit-tested against a hand-constructed blob.

**MKWii GCN mapping**:

| In-game action | GCN button | Analog? |
|---|---|---|
| Accelerate | A | no |
| Brake / reverse | B | no |
| Drift / hop | R (digital press) | no |
| Use item | L (digital press) | no |
| Look behind | X | no |
| Steering | Analog X (-1 to 1, normalized from 1-255 with 128 neutral) | yes |
| Pause | Start | no — exclude from training |

Produce a normalized `ControllerState` per frame:

```python
@dataclass(frozen=True)
class ControllerState:
    frame_idx: int           # 0-indexed input frame (not VI)
    steering: float          # analog X, normalized to [-1, 1]
    accelerate: bool         # A
    brake: bool              # B
    drift: bool              # R digital
    item: bool               # L digital
    look_behind: bool        # X
    # Raw fields preserved for debugging:
    _raw_analog_x: int
    _raw_analog_y: int
    _raw_byte0: int
    _raw_byte1: int
```

**Don't collapse trigger pressure to boolean** without first verifying MKWii doesn't use analog L/R for anything meaningful. (It doesn't, but verify once and cite in a code comment.)

**Parser API:**

```python
def parse_dtm(path: Path) -> tuple[DtmHeader, list[ControllerState]]:
    """Parse a .dtm file. Validates PAL (RMCP01). Raises on malformed input."""
```

`DtmHeader` must surface `from_savestate`, `vi_count`, `input_count`, `lag_count` — downstream code needs these.

**Unit tests** (`tests/test_dtm_parser.py`):
- Hand-constructed binary blob: 100 frames, hold only A → parse, assert `accelerate=True` everywhere, steering `≈0`.
- Hand-constructed: hold full-left → `steering < -0.9` everywhere.
- Malformed: truncated header → `DtmFormatError`. Wrong game ID → `DtmRegionError`.
- Round-trip tests against real `.dtm` files are marked `@pytest.mark.skipif(not path.exists())` and run opportunistically.

### 1.2 Record-then-replay pipeline (`scripts/capture_demo.md` and `scripts/replay_demo.py`)

**Why two steps, not one**: Dolphin's PNG frame dump slows emulation below realtime on a Mac Mini. If `.dtm` is recorded with dumping enabled, the emulator runs slowly during recording, the user's inputs reflect the slowdown, and the resulting `.dtm` + frames pair is distorted relative to a clean-speed recording. The fix is separation: record at full speed, replay at whatever speed with dumping.

**User workflow (`scripts/capture_demo.md`)**:
1. Load `data/savestates/luigi_circuit_tt.sav`.
2. `Movie → Start Recording Input`. **Frame dump must be disabled.**
3. Play Luigi Circuit Time Trial.
4. `Movie → Stop Recording`. Save to `data/raw/demos/{timestamp}.dtm`.

**Replay step (`scripts/replay_demo.py`)**:
1. Take `.dtm` path + matching savestate as args.
2. Launch Dolphin in headless/scripting mode with frame dump enabled (PNG sequence).
3. Load the savestate referenced by the `.dtm` header.
4. Play back the `.dtm` to completion.
5. Frames land in `data/raw/frames/{demo_timestamp}/`.
6. Write a sidecar `{demo_timestamp}.replay.json` containing: replay wall-clock duration, frame count produced, any Dolphin warnings.

If the replay script can't be fully automated in v1 (scripting API limitations), generate a `REPLAY_PROTOCOL.md` with the manual steps and leave a TODO.

### 1.3 Frame dump reader (`src/mkw_rl/dtm/frames.py`)

Dolphin's frame dump on macOS outputs to `~/Library/Application Support/Dolphin/Dump/Frames/`. Prefer PNG sequence output over AVI. Configure via Dolphin's GFX config: `Dump Frames` enabled, `Dump Frames as Images` enabled.

**Do not run ffmpeg** on the frame dump. Load PNGs directly via PIL. JPEG is forbidden in this pipeline — lossy compression on training data is a known generalization hazard.

```python
@dataclass
class FrameDump:
    frame_dir: Path
    frame_paths: list[Path]  # sorted by frame index

def load_frame_dump(frame_dir: Path) -> FrameDump: ...
def load_frame(path: Path, size: tuple[int, int] = (140, 75), grayscale: bool = True) -> np.ndarray: ...
```

### 1.4 Pairing (`src/mkw_rl/dtm/pairing.py`)

**This is where silent data corruption lives. Read this section carefully.**

**Alignment strategy** (revised from v1):

Because the `.dtm` is recorded *from a known savestate* (see §0.4) and replayed *from the same savestate* (see §1.2), the start of both sequences is deterministic and clean. It is the **end** that is ragged: the user stops recording at an arbitrary point, and the replay may flush a few frames late.

Therefore:

1. **Align from the start.** Frame 0 of the frame dump corresponds to input frame 0 of the `.dtm` (the first input after savestate load).
2. **Skip menu/HUD-noisy frames.** Countdown animation runs for ~300 frames from savestate start depending on where the savestate was anchored. Use `skip_first_n: int` to drop these — defaults vary by savestate, document per-savestate in the savestate JSON sidecar from §0.4.
3. **Trim the tail.** Take `min(len(inputs), len(frames)) - tail_margin` where `tail_margin` defaults to 10 frames.
4. **Sanity check**: compute `abs(len(frames) - len(inputs))`. If it exceeds `max(30, 0.02 * len(inputs))`, warn loudly with the actual counts. This threshold is looser than v1 because legitimate replay-side buffering can stretch beyond v1's `max(10, 0.01*)`.

**Open question**: `vi_count` vs `input_count` semantics in the header. The frame dump indexes by rendered frame (VI). The controller data indexes by input frame. Under ideal emulation without lag these are equal. With lag frames they diverge; `lag_count` tracks the delta. If `lag_count > 0`, log a warning and consider this demo second-class. **Verify against `Movie.cpp` before writing the alignment math**; the TASVideos wiki has been unreliable here.

**Output of pairing:**

```python
@dataclass
class PairedSample:
    frame_idx: int           # 0-indexed, after skip_first_n applied
    input_frame_idx: int     # index into .dtm inputs
    frame_path: Path
    controller: ControllerState

def pair_dtm_and_frames(
    dtm_path: Path,
    frame_dir: Path,
    skip_first_n: int = 0,   # default 0; caller supplies from savestate sidecar
    tail_margin: int = 10,
) -> list[PairedSample]: ...
```

### 1.5 Sanity visualizer (`src/mkw_rl/dtm/viz.py`) — BUILD THIS FIRST

**Build this before the hardened pairing and before the dataset.** If the visualizer looks right, the parser and alignment are right. If it looks wrong, nothing downstream can be trusted.

Given a `PairedSample`, render the frame with overlays:
- **Steering indicator**: horizontal bar at the bottom, filled left or right proportional to `steering`.
- **A button**: green dot in the bottom-right when `accelerate=True`.
- **B button**: red dot when `brake=True`.
- **R (drift)**: blue dot when `drift=True`.
- **L (item)**: yellow dot when `item=True`.
- **Frame index** in the top-left.

Output an MP4 of the first 30 seconds of a paired recording using `imageio[ffmpeg]`. User watches it back and visually confirms: steering matches kart turning, A lights up during acceleration, drift lights up during drifts.

**This video is the acceptance criterion for Phase 1.** Do not proceed to Phase 2 until the user confirms the overlay matches reality.

Prompt 1b builds viz + a minimal pairing (just `len(min)` trim from start) so the visualizer works end-to-end. Prompt 1c then returns to §1.4 to harden pairing.

### 1.6 Sequence-returning dataset (`src/mkw_rl/dtm/dataset.py`)

**This section is materially rewritten from v1**. The v1 dataset returned independent `(frame_stack, action)` pairs, which is incompatible with the stateful LSTM in §2.2 (the LSTM would see length-1 sequences and degenerate to a no-op linear layer). The v2 dataset returns **sequences** of `(frame_stack, action)` timesteps, and training uses truncated backpropagation through time.

**Why keep both frame-stack AND LSTM**: the 4-frame stack captures sub-second dynamics (drift state, mini-turbo charging, hop frames). The LSTM captures multi-second context (upcoming turn sequences, item inventory timing, lap-phase awareness). These serve different horizons. This architectural choice is made with Phase 3 (all 32 tracks) in mind — track identity can be inferred over seconds of footage, and the LSTM state is the right place for this.

```python
class MkwBCDataset(Dataset):
    """
    Returns fixed-length sequences for TBPTT. Each item is a contiguous slice
    from one demo, respecting demo boundaries (never splices across demos).

    Returns per __getitem__:
        frames: torch.Tensor, shape (T, stack_size=4, H=75, W=140), grayscale, float32 in [0,1]
        action: dict of tensors with shape (T,) each:
            'steering_bin': long, ∈ [0, 20]          # 21-bin discretization
            'accelerate':  float, ∈ {0, 1}
            'brake':       float, ∈ {0, 1}
            'drift':       float, ∈ {0, 1}
            'item':        float, ∈ {0, 1}
        meta: dict with
            'demo_id': str
            'seq_start': int   # index of first timestep within the demo
            'is_continuation': bool  # True if this seq continues a prior one (for state carry)
    """
    def __init__(
        self,
        samples_by_demo: dict[str, list[PairedSample]],
        stack_size: int = 4,
        frame_skip: int = 4,
        seq_len: int = 32,           # TBPTT window length
        transform: Callable | None = None,
    ): ...
```

**Frame stacking within a sequence**: for timestep `t` in the sequence, observation is `[frames[t-3*frameskip], frames[t-2*frameskip], frames[t-frameskip], frames[t]]`. Pad with copies of frame 0 at the start of each demo only — never pad across a demo boundary.

**Sequence sampling and ordering**:

- Sampling is demo-aware. `__len__` returns the total number of non-overlapping `seq_len` windows across all demos, minus any trailing partial window.
- `is_continuation` tells the training loop whether to carry LSTM hidden state from the previous batch item for this demo or reinitialize it.
- **Sampler**: use a custom `BatchSampler` that groups timesteps by demo so each batch position tracks one demo across calls. Batch position `b` always sees demo `demos[b % len(demos)]` (or similar), with hidden state carried across batches for that position. Shuffling happens at the demo level between epochs, not at the sequence level within a demo.

See §2.3 for how the training loop consumes this.

### 1.7 Action encoding (`src/mkw_rl/dtm/action_encoding.py`)

**Rationale**: MKWii steering is effectively bimodal — hard-left and hard-right dominate during drifts, neutral during straights. Regression with MSE on bimodal targets converges to the mean, producing a policy that drives straight into walls. Discretize steering into 21 bins (equal-width over `[-1, 1]`), train with cross-entropy, decode to the bin center at inference.

```python
N_STEERING_BINS = 21  # odd so zero is a bin center

def encode_steering(x: float) -> int:
    """Map x ∈ [-1, 1] to bin index ∈ [0, N_STEERING_BINS - 1]."""

def decode_steering(bin_idx: int) -> float:
    """Map bin index back to bin center in [-1, 1]."""
```

Unit tests: `decode_steering(encode_steering(x))` within `1 / N_STEERING_BINS` of `x` for 1000 random inputs. `encode_steering(-1) == 0`, `encode_steering(1) == N_STEERING_BINS - 1`, `encode_steering(0) == 10`.

---

## Phase 2 — Behavioral cloning

**Goal**: Train an IMPALA-CNN + stateful-LSTM policy that, given stacked frames and carried hidden state, predicts discretized steering and four binary buttons. This is the initialization for Phase 4's RL fine-tune.

### 2.1 Data collection

Two separate pools, not mixed naively:

**User demos** (`data/raw/demos/`): User records own gameplay per §1.2. Target: 20+ Luigi Circuit Time Trial laps. Reflects the user's play distribution; this is the primary training data.

**TAS demos** (`data/raw/tas/`): `.dtm` files from the MKWii TAS community for Time Trial records. Each must be attributed in a `data/raw/tas/ATTRIBUTION.md` file with source URL, author, and license. **These are a different distribution from user demos** — they are frame-perfect and may have been recorded from a different savestate or from boot (`from_savestate=0`). Do not mix into the training pool without a matching savestate anchor.

For Phase 2 initial bring-up: **user demos only**. TAS demos are Phase 2.5 / Phase 3 material once we have a working baseline and can actually measure whether they help.

Claude Code: generate `scripts/capture_demo.md` with the user-facing protocol from §1.2. Do not attempt to automate recording; the user is in the loop.

### 2.2 Model (`src/mkw_rl/bc/model.py`)

**Architecture (revised from v1)**:

```
Input: (B, T, stack_size=4, H=75, W=140), grayscale, float in [0,1]

Encoder: IMPALA-style CNN applied per-timestep
    Block 1: Conv 16 ch, 3x3, stride 1  → MaxPool 3x3 stride 2 → 2x residual blocks (16 ch)
    Block 2: Conv 32 ch, 3x3, stride 1  → MaxPool 3x3 stride 2 → 2x residual blocks (32 ch)
    Block 3: Conv 32 ch, 3x3, stride 1  → MaxPool 3x3 stride 2 → 2x residual blocks (32 ch)
    Flatten → Linear → ReLU → 256-dim per-timestep feature

Temporal: LSTM, hidden_dim=512, 1 layer, stateful across TBPTT windows

Heads (per timestep, consuming LSTM output):
    steering_head:  Linear(512 → 21)    # cross-entropy over 21 bins
    accelerate_head: Linear(512 → 1)    # BCE with logits
    brake_head:      Linear(512 → 1)
    drift_head:      Linear(512 → 1)
    item_head:       Linear(512 → 1)

Forward signature:
    forward(frames: (B, T, 4, H, W), hidden: (h, c)) -> (logits_dict, new_hidden)
```

**Why IMPALA CNN, not ResNet-18**: IMPALA convs are the standard for pixel-based RL (Atari, DMLab, Procgen) at a fraction of the parameter count. ResNet-18 is 11M params; IMPALA is ~1M. On 140×75 grayscale inputs the extra depth is wasted, and ResNet's ImageNet init is half-destroyed by the 4-channel first-conv reinit anyway. IMPALA trains faster, generalizes at least as well on this scale of input, and is what Phase 4's RL code will want to drop in.

**Why stateful LSTM with TBPTT**: LSTM hidden state is the right place to carry multi-second context: track identity (for Phase 3), item inventory timing, lap phase, recent-turn history beyond what the 4-frame stack sees. Training must preserve hidden state across TBPTT windows within a demo (see §2.3), otherwise the LSTM is trained on length-`seq_len` contexts only and loses the ability to integrate over longer horizons at inference.

**Why discretized steering**: see §1.7. Bimodal target, MSE regresses to mean.

**Loss**:
- Steering: cross-entropy over 21 bins.
- Buttons: BCE with logits per button.
- Total: weighted sum: `L = steering_weight * ce_steering + button_weight * mean(bce_buttons)`. Start with `steering_weight = 1.0, button_weight = 1.0`.

Smoke test (Prompt 2a): forward random noise at `(B=2, T=8, 4, 75, 140)`, assert all output shapes correct, assert hidden state round-trips (second call with returned hidden gives deterministic next output).

### 2.3 Training (`scripts/train_bc.py`, `src/mkw_rl/bc/train.py`)

**Truncated BPTT loop**:

```
for epoch in range(n_epochs):
    hidden = {b: zeros for b in range(batch_size)}
    sampler = DemoAwareBatchSampler(dataset, batch_size, seq_len=32)
    for batch in sampler:
        # batch[b] is a seq_len-long slice from the demo assigned to batch position b.
        # is_continuation[b] tells us whether to keep hidden[b] or reset it.
        for b in range(batch_size):
            if not batch.is_continuation[b]:
                hidden[b] = zeros
        hidden_stacked = stack(hidden)
        logits, new_hidden = model(batch.frames, hidden_stacked)
        loss = compute_loss(logits, batch.actions)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        hidden = detach_and_unstack(new_hidden)  # detach for TBPTT, keep values
```

Three things matter:

1. **`detach()` hidden state after backward.** Otherwise the graph grows without bound across windows and you OOM.
2. **Reset hidden state at demo boundaries**, not at batch boundaries. Batch position `b` might spend 10 consecutive batches inside demo X, then switch to demo Y when demo X's windows are exhausted.
3. **Epoch-level demo shuffling** only. Within a demo, windows are consumed in order.

Config (`configs/bc.yaml`):

```yaml
data:
  demo_glob: "data/processed/user_demos/*.pt"
  train_val_split: 0.9
  batch_size: 16           # lower than v1's 64: sequences are bigger
  num_workers: 4
  stack_size: 4
  frame_skip: 4
  seq_len: 32              # TBPTT window

model:
  encoder: impala
  lstm_hidden: 512
  lstm_layers: 1
  n_steering_bins: 21

optim:
  lr: 3e-4
  weight_decay: 1e-5
  epochs: 50
  scheduler: cosine
  grad_clip: 1.0           # important with LSTMs

loss:
  steering_weight: 1.0
  button_weight: 1.0

logging:
  wandb_project: mkw-rl
  checkpoint_every: 5
```

**Dry-run sanity** (Prompt 2b acceptance): train for 1 epoch on the user's processed demos. Before celebrating a loss decrease, verify **three diagnostics**:

1. Per-bin steering loss is decreasing (not just overall CE). If only bin 10 (neutral) is improving, the model is collapsing — flag and stop.
2. Per-button F1 on training data is above 0.6 for at least A (accelerate) after one epoch. A lower bound on learnability.
3. LSTM gradient norm is non-trivial (> 1e-4). If the LSTM is getting zero gradients, TBPTT plumbing is broken.

### 2.4 Evaluation (`src/mkw_rl/bc/eval.py`)

Offline eval metrics (on held-out demos):
- Steering top-1 accuracy over 21 bins.
- Steering top-3 accuracy (more forgiving; often the right diagnostic).
- Per-button F1.
- Joint prediction accuracy: "all buttons correct AND steering bin within ±1".

**Side-by-side overlay video**: left pane = ground-truth controller state, right pane = predicted. Model is run with carried hidden state across the full held-out demo (not reset per-window). This is the human-readable sanity check.

**Online eval (Phase 2.5, optional before Phase 4)**: a minimal "load savestate, drive the policy for 60s, measure whether it finishes a lap" script. Does not require the full Phase 4 gym env — just input injection via the scripting API. Defer unless offline metrics look ambiguous and you need a tiebreaker.

Phase 2 ends when offline metrics plateau and the user spot-checks a few predicted trajectories against held-out demos. Online eval is a nice-to-have, not a gate.

---

## Phase 3 — All-32-tracks generalization (appendix stub)

Phase 3 converts the single-track BC model into a 32-track model. Work:

- Build 32 savestates (one per vanilla track), Luigi Circuit → Rainbow Road, documented via `SAVESTATE_PROTOCOL.md` with exact VI counts per §0.4.
- Collect demos per track. Budget: 10-20 laps per track = 320-640 demos total. Consider TAS `.dtm` imports for tracks the user drives poorly — now a real option since the LSTM can absorb the distribution shift if user + TAS demos share savestate anchors.
- Train a single multi-track BC model, same architecture as Phase 2. Stateful LSTM from day one means no architectural change. Start with naive mixing; if underperforms, add a track-ID embedding concatenated to LSTM input.
- Evaluate per-track and on a held-out "test track" (e.g., hold out DK Summit) to measure zero-shot generalization.

**Do not attempt Phase 3 before Phase 4's RL env is working on Luigi Circuit.** Multi-track BC with no RL fine-tune is a less interesting result than single-track BC → RL.

---

## Phase 4 — Gym env + BTR/PPO RL fine-tune (appendix stub — superseded)

> **Status**: superseded by the 2026-04-17 pivot (see `docs/PIVOT_2026-04-17.md`). What was Phase 4 here is now the active **Phase 2**. Content below is preserved as historical record but is no longer authoritative.

Original Phase 4 scope (pre-pivot):

- `src/mkw_rl/env/dolphin_env.py` — `gymnasium.Env` wrapping their scripting fork's Python API. Observation: `(4, 75, 140)` uint8 frames. **Action space: `Discrete(40)` per VIPTankz** (5 stick positions × 2 drift × 2 up × 2 item, A always held — see `docs/TRAINING_METHODOLOGY.md`). Earlier drafts of this line proposed a 21-way steering × 16-way button space; that's the BC action space, not BTR's.
- `src/mkw_rl/env/reward.py` — reward shaping per `docs/TRAINING_METHODOLOGY.md` §5 (variable-per-track checkpoints × speed bonus, off-road/wall penalties, finish + position bonuses, lenient reset).
- `src/mkw_rl/env/dolphin_script.py` — slave-side RAM reads. PAL RMCP01 pointer chains ported verbatim from VIPTankz's `DolphinScript.py` (`0x809BD730`, `0x809C18F8`, `0x809C3618`).
- `src/mkw_rl/rl/btr.py` — BTR fork with LSTM added on top of the IMPALA encoder + R2D2-style burn-in replay. Future BC warm-start path: load BC's encoder+LSTM weights only (not heads — different action spaces).

---

## Phase 5 — Retro Rewind custom tracks (appendix stub)

**⚠️ Retro Rewind is NTSC-U only.** With the PAL region decision (see docs/REGION_DECISION.md), this phase requires either flipping to NTSC-U (and redoing Phase 4 RAM work) or dropping RR entirely. If kept, likely work:
- Flip project region to NTSC-U, or stand up a parallel NTSC-U track.
- Custom savestates per RR track.
- Verify RAM addresses (would need NTSC-U port of VIPTankz's PAL pointers).
- Retrain multi-track model on combined vanilla + RR pool.

Out of scope for initial spec. Revisit after Phase 4 with a deliberate region-flip decision.

---

## Phase 6 — Autoresearch stretch (appendix stub)

[miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) for architecture search. Realistic only on Vast.ai, not on 16GB Mac Minis. Deferred.

---

## Distributed training note

2× M4 Mac Minis (16GB each) as distributed rollout workers: useful for Phase 4, not Phase 1-2. BC training is single-machine and I/O-bound at Mac Mini scale. Don't over-engineer Phase 1-2 for distribution.

For Phase 4, VIPTankz's repo already supports running multiple Dolphin instances per machine via `clone_dolphins.py`. Extending this across machines needs a message bus (Redis / ZeroMQ) and is Phase 4+ scope.

---

## Non-goals

- Wii Remote / Nunchuck / Classic Controller support.
- PAL / Japanese / Korean regions.
- Online multiplayer.
- Battle mode.
- AI that beats the player in local split-screen (different problem, different infra).
- Continuous steering output. Discretize everywhere.

---

## Explicit decisions deferred to implementation

- Exact optimizer (AdamW almost certainly, but tune).
- Whether to freeze IMPALA encoder during later BC epochs (probably no, but try).
- Whether BC should predict the *next* input or the *current* input given current frame (probably current; verify with offline eval).
- Factored vs joint action space for Phase 4 (21-way steering × 2^4 buttons = 336 joint actions — possibly tractable; otherwise factored).
- Whether to add a track-ID embedding in Phase 3 or hope the LSTM infers it.

---

## Acceptance criteria per phase

| Phase | Done when |
|---|---|
| P-1 | All six preflight checks pass on user's machine. Pinned SHA recorded in `docs/PREFLIGHT.md`. Python linkage determined. |
| 0 | `uv run python -c "import mkw_rl"` succeeds. Submodule present at pinned SHA. `SETUP.md` exists. Luigi Circuit savestate loaded correctly in Dolphin fork, VI count documented. |
| 1 | Sanity visualizer MP4 matches reality per user confirmation. `tests/test_dtm_parser.py` and `tests/test_pairing.py` all green. Record-replay pipeline produces at least one paired demo end-to-end. |
| 2 | BC model trains with all three dry-run diagnostics passing (per-bin steering loss decreasing, A-button F1 > 0.6, LSTM grad-norm > 1e-4). Offline eval metrics logged. Side-by-side overlay video confirms qualitative driving behavior on held-out demo. |
| 3+ | Deferred; re-spec before starting. |

---

## For Claude Code: working style

- **Prefer small, reviewable PRs** — one module per commit. Don't batch Phase 1.1 + 1.2 + 1.3 into one giant diff.
- **Write the tests alongside the module**, not after. `test_dtm_parser.py` ships with `parser.py`.
- **Prototype-first for model / training code, test-driven for parsing / data code.** Parser correctness is load-bearing; training hyperparams aren't.
- **Don't touch Phase 3+ code until user explicitly asks.** The appendix stubs are for context, not for implementation.
- **When in doubt about `.dtm` semantics, check Dolphin's `Movie.cpp` directly, not the TASVideos wiki.** This is called out in §1.1 and §1.4 for good reason.
- **Run `ruff check` and `ruff format` before every commit.**
- **If something takes more than 2 iterations to get right, stop and ask.** Silent thrashing is worse than asking.
- **Never mix user demos and TAS demos without matching savestate anchors.** See §2.1.
- **Record and replay are two separate steps.** See §1.2. If Claude Code is ever tempted to write a "record with dumping enabled" script, it's wrong.
