# Region decision: NTSC-U (RMCE01) vs PAL (RMCP01)

**Decision (2026-04-17): PAL (`RMCP01`).** Propagated across `SETUP.md`, `docs/PREFLIGHT.md`, `docs/SAVESTATE_PROTOCOL.md`, `MKW_RL_SPEC.md`, `scripts/preflight.py`, `scripts/capture_demo.md`, `scripts/REPLAY_PROTOCOL.md`, `scripts/replay_demo.py`, `src/mkw_rl/dtm/parser.py` (and test fixtures), `src/mkw_rl/dtm/frames.py`, `src/mkw_rl/dtm/viz.py`, `CLAUDE_CODE_PROMPTS.md`, `CHANGES.md`. Rationale below was the input to the decision; preserved as record. Primary driver: save 1-2 days of Phase 4 RAM reverse-engineering, gain access to VIPTankz's pre-trained baseline. Cost: Retro Rewind (Phase 5) is now out of scope without a second region flip.

---

**Status (historical):** open. Must be resolved before P-1 execution, because the choice determines which ROM you acquire, which savestates you build, and which RAM addresses Phase 4 uses.

**Current spec assumption:** NTSC-U (RMCE01). Baked into `SETUP.md:11`, `docs/SAVESTATE_PROTOCOL.md:19,60,75,154`, `scripts/preflight.py:167`, `AUDIT.md:93`. The assumption predates closer inspection of VIPTankz's repo.

**VIPTankz's assumption:** PAL (RMCP01). Stated in their README §2. All their artifacts (pre-compiled Dolphin aside — that's region-agnostic, it's the binary itself) are PAL-specific: savestates from `scripts/download_savestates.py`, RAM addresses hardcoded in `DolphinScript.py` with no region branching (`0x809BD730`, `0x809C18F8`, `0x809C3618` base pointers), the pre-trained model, and `DolphinEnv._check_iso_validity` (which MD5-checks against their PAL ISO hash and warns otherwise but does not hard-fail).

---

## What's actually region-specific

| Thing | Region-dependent? | Notes |
|---|---|---|
| Dolphin binary itself | No | Same binary boots any region. |
| Scripting API (`from dolphin import ...`) | No | Fork feature, not game feature. |
| BC pipeline (`src/mkw_rl/bc/`, `src/mkw_rl/dtm/`) | No | Reads frames + .dtm inputs. Frame contents look ~identical between regions at gameplay level; .dtm format is the same. BC training would work on either region's demos, provided demos + replays + eval all run on the same region. |
| `.dtm` format | No | Dolphin-wide format, region-agnostic. |
| Savestate files | **Yes** | A PAL savestate will not load on NTSC-U. VIPTankz's `MarioKartSaveStates` is PAL only. |
| RAM addresses (lap, position, speed, etc.) | **Yes** | Different base pointers per region. `DolphinScript.py` hardcodes PAL addresses with no region check. Reading them on NTSC-U returns garbage. |
| Frame rate | **Yes** | NTSC-U ~59.94 fps, PAL 50 fps. Affects real-time-per-VI and reward shaping timescales. |
| Luigi Circuit lap time baselines | Yes | PAL laps are ~20% slower in wall-clock than NTSC-U due to 50 vs 60 Hz. |
| Pre-trained BTR model | **Yes** | Trained on PAL frames + PAL rewards + PAL RAM reads. Effectively useless on NTSC-U. |

**Takeaway:** BC (Phase 2) is basically region-agnostic as long as you're internally consistent. Phase 4 (RL environment with reward = function of RAM reads) is where region locks in.

---

## Option A — Stay NTSC-U (current spec)

### Costs
- Build Luigi Circuit savestate ourselves (`docs/SAVESTATE_PROTOCOL.md` already documents the procedure — ~30 minutes once).
- Re-derive RAM addresses for NTSC-U in Phase 4. Options:
  - Mine MKWii decomp / WiiBrew / MKWii Community for NTSC-U addresses. Lap counter and kart position are well-known; VIPTankz's 20+ pointers (internal velocity, miniturbo charge, surface flags, wheelie frames, etc.) require more research.
  - Use a debugger + memory search in Dolphin to translate their PAL pointer chains to NTSC-U. Mechanical but tedious.
- Cannot use VIPTankz's pre-trained model as a warm start or sanity baseline.
- Cannot drop their `DolphinEnv.py` straight in — must port RAM reads.

### Benefits
- NTSC-U is the standard for US MKWii community, TAS community, speedrunning. More community resources per address, more existing mods, more likely the human operator (you) already has a legit ISO.
- 60 fps means faster wall-clock per training frame — every environment step is ~20% cheaper than PAL.
- No conflict with existing spec; the work we've already shipped (`MKW_RL_SPEC.md`, `docs/SAVESTATE_PROTOCOL.md`, etc.) stands.
- Our demos, if you've already started recording, stay valid.

### Effort estimate
- Phase 0-2: zero change. Already built, already passing.
- Phase 4 RAM-reading work: maybe 1-2 days of focused reverse-engineering on top of what was already planned. The spec was already assuming we'd do this — VIPTankz's addresses are a helpful reference even if not directly reusable.

---

## Option B — Switch to PAL (follow VIPTankz)

### Costs
- Update `SETUP.md`, `docs/PREFLIGHT.md`, `docs/SAVESTATE_PROTOCOL.md`, `scripts/preflight.py`, `AUDIT.md` to reference RMCP01.
- Acquire a PAL RMCP01 ISO (`mkw.iso`, MD5 `e7b1ff1fabb0789482ce2cb0661d986e`, 4.38 GB). You may or may not have one.
- If you've already recorded NTSC-U demos: discard them. Re-record on PAL.
- Commit to 50 fps timing throughout. Update any `frame_skip` / reward-timescale assumptions that may have implicitly assumed 60 Hz.

### Benefits
- Phase 4 is ~plug-and-play: VIPTankz's `DolphinEnv.py` + `DolphinScript.py` run as-is, savestates drop in, their RAM pointer chains work, their pre-trained model is a valid baseline / warm-start candidate.
- Skip the RAM reverse-engineering entirely (or at least defer it indefinitely — add addresses only if you extend beyond what VIPTankz exposed).
- Directly comparable to their published results. If we diverge from their training curve, we know it's our changes, not our environment.
- If you eventually want to reproduce their "first place in ~1 day on RTX 4090" claim, you need their exact setup.

### Effort estimate
- Doc + config churn: half a day.
- Demo re-record: whatever we'd already spent × 1, if any (currently zero — no real demos recorded yet per CHANGES.md).
- Phase 4: massively reduced. Maybe 1-2 days instead of 3-5.

---

## Decision matrix

|  | Option A (NTSC-U) | Option B (PAL) |
|---|---|---|
| Already-have ISO? | 👤 you | 👤 you |
| Phase 4 effort | Higher (RAM work) | Lower (drop in VIPTankz) |
| Pretrained-model warm start | ❌ | ✅ |
| Community resources for region | ✅ (US modding hub) | ✅ (VIPTankz + EU scene) |
| Wall-clock training speed | ~20% faster (60 vs 50 fps) | Slower |
| Already-baked in our spec | ✅ | Requires doc churn |
| Demo re-record | ✅ valid | Discard any existing |

---

## Recommendation

**Default to Option B (PAL) unless you specifically want the NTSC-U training speed advantage or already have a legit NTSC-U ISO but not a PAL one.** The core value of this project is getting an RL agent that drives — not reproducing NTSC-U specifically. VIPTankz has done the hard RAM-reversing work for PAL and published a validated baseline; burning 1-2 days to redo that on NTSC-U buys us nothing unless we have a specific reason.

The ~20% wall-clock advantage from 60 fps is real but small compared to the risk of getting RAM reads subtly wrong and poisoning reward signal for weeks before noticing.

**Decision factors that would flip me back to Option A:**
- You already have NTSC-U ISO and acquiring PAL is friction.
- You've already recorded demos you don't want to discard.
- You plan to deviate significantly from VIPTankz's reward shaping anyway (in which case their RAM pointer work is less load-bearing).
- NTSC-U speedrun community comparability matters more to you than published RL baseline comparability.

---

## Action items once decided

### If Option A (NTSC-U):
- No doc changes needed. PREFLIGHT.md already targets NTSC-U.
- Proceed to P-1 Step 1 with NTSC-U ISO in hand.
- Park a follow-up task for Phase 4: "port VIPTankz RAM pointers to NTSC-U."

### If Option B (PAL):
- Global find-replace `RMCE01` → `RMCP01`, `NTSC-U` → `PAL`, `NTSC-U MKWii` → `PAL MKWii` across `SETUP.md`, `docs/PREFLIGHT.md`, `docs/SAVESTATE_PROTOCOL.md`, `scripts/preflight.py`, `AUDIT.md`, `MKW_RL_SPEC.md`.
- Grep for hard-coded `60` or `59.94` in frame-timing logic, audit for 50 Hz appropriateness.
- Acquire PAL ISO, verify MD5.
- After P-1, run `python3 scripts/download_savestates.py` from `~/code/mkw/Wii-RL` to get their pre-built Luigi Circuit savestate — skips our whole manual savestate procedure.
