# Savestate protocol

Every per-track savestate used in this project must be created following this exact protocol. Deviations break RNG determinism and cascade into silent data corruption in the BC and RL pipelines.

## Why this matters

Dolphin savestates capture the full emulator state including:

- CPU registers and memory.
- GPU state.
- **RNG internal state.** Mario Kart Wii uses RNG for item distribution, CPU racer behavior, and some physics tie-breaks.

Two savestates made at "the same moment" but a few VIs apart will have different RNG states. Replays from those savestates will diverge under identical inputs. For reproducibility — and for the Phase 4 RL reward to converge — every savestate must be anchored to a precisely documented VI count.

## Protocol

### Step 1 — Prepare Dolphin

Run VIPTankz's fork. Load your NTSC-U `RMCE01` ISO.

Pre-conditions on the Dolphin side:

- Enable the on-screen frame counter: `View → Show Frame Counter` (or whatever the fork calls it; the value you need is VI count).
- Disable frame dumping. Savestate creation is clean-speed.
- Set graphics backend to something reproducible (Metal + Hardware Rendering, default settings).
- Controller profile: GameCube controller, port 1. This is load-bearing — `.dtm` recording depends on GCN port 1 being active.

### Step 2 — Reach the pre-input anchor frame

Different tracks differ in menu flow, but for vanilla tracks the anchor point is the same:

1. `Single Player → Time Trial → [track] → Standard Kart, Mario, no ghost`.
2. Wait for the countdown to complete: "3", "2", "1", "GO!".
3. Frame-step (F3 or configured hotkey) through the "GO!" banner until the banner disappears and **the first frame on which kart inputs apply** is on screen.

The "first frame on which kart inputs apply" is the anchor frame. It's deterministic per track — under identical Dolphin settings and ISO, every user hits the same VI count at this frame.

### Step 3 — Save the state

1. Save state to slot 1 (or any slot).
2. Export the state to `data/savestates/{track_slug}.sav`.
3. Note the VI count at the exact frame of the save. Read it from the frame counter overlay.

**Naming convention**: `{track_slug}.sav` where `track_slug` is lowercase with underscores. Examples:

| Track | Slug |
|---|---|
| Luigi Circuit | `luigi_circuit_tt` |
| Moo Moo Meadows | `moo_moo_meadows_tt` |
| Rainbow Road | `rainbow_road_tt` |

The `_tt` suffix marks it as a Time Trial savestate. If we later add Grand Prix or VS savestates, use `_gp`, `_vs`, etc.

### Step 4 — Write the JSON sidecar

Create `data/savestates/{track_slug}.json` with this exact structure:

```json
{
  "game_id": "RMCE01",
  "track": "luigi_circuit",
  "mode": "time_trial",
  "character": "mario",
  "vehicle": "standard_kart",
  "vi_count": 3427,
  "skip_first_n": 0,
  "notes": "Saved at first frame kart inputs apply. Pre-race RNG is pre-item-roll."
}
```

Field reference:

| Field | Required | Notes |
|---|---|---|
| `game_id` | yes | Always `RMCE01` — sanity check. |
| `track` | yes | Lowercase slug without `_tt`. |
| `mode` | yes | `time_trial`, `gp`, `vs`, `battle`. |
| `character` | yes | Driver. For vanilla tracks use `mario` unless a specific experiment needs otherwise. |
| `vehicle` | yes | Kart / bike. Default `standard_kart`. |
| `vi_count` | yes | Exact VI count at save. Used to diagnose drift if anyone else reproduces. |
| `skip_first_n` | yes | Number of initial input frames to drop during §1.4 pairing. For a clean "first frame inputs apply" anchor, this is `0`. For a slightly-earlier anchor (e.g., during the "GO!" banner), set this to however many input frames are between save and actual input responsiveness. |
| `notes` | yes | Free-form. Note anything that might be relevant to future-you about the anchor choice. |

### Step 5 — Verify determinism

Before committing the savestate:

1. Load the state. Let it run for ~60 VIs without inputs.
2. Take a screenshot.
3. Load the state again. Run for ~60 VIs without inputs.
4. Take another screenshot.
5. `diff` the two screenshots. They should be bit-for-bit identical.

If they differ: something in Dolphin's setup is non-deterministic. Fix it before proceeding — this invalidates every downstream training run if left unfixed.

Common culprits:

- Audio backend differences (use the same audio config every session).
- Frame limiter on vs. off (set the same way consistently).
- Hardware cursor overlaying the frame dump (Dolphin bug; disable if it happens).

### Step 6 — Record where the VI count came from

Commit the `.sav` file and its JSON sidecar in the same commit. In the commit message, include:

- Track slug.
- VI count.
- Dolphin fork SHA the savestate was made under (should match the submodule pin).

Example commit message:

```
savestate: Luigi Circuit TT anchor

track: luigi_circuit_tt
vi_count: 3427
dolphin fork SHA: d8358cb...
character: Mario / Standard Kart
mode: Time Trial, no ghost
```

## Re-anchoring

If the Dolphin fork is upgraded to a new SHA:

1. Make a fresh savestate per this protocol under the new SHA.
2. Verify the old SHA's savestates still replay deterministically on the new SHA. Usually they do, but not always.
3. If they don't, **re-record every demo** with the new savestate. Do not mix demos across Dolphin SHAs.

## Per-track savestate checklist

Phase 1-2 scope:

- [ ] `luigi_circuit_tt.sav` + `.json` — Mario / Standard Kart / Time Trial.

Phase 3 scope (all 32 vanilla tracks):

- [ ] Mushroom Cup (4 tracks).
- [ ] Flower Cup (4).
- [ ] Star Cup (4).
- [ ] Special Cup (4).
- [ ] Shell Cup (4 retro).
- [ ] Banana Cup (4 retro).
- [ ] Leaf Cup (4 retro).
- [ ] Lightning Cup (4 retro).

Maintain this list as tracks are covered.

## Anti-patterns

Do NOT:

- Save at an arbitrary mid-race frame. Always use the consistent pre-input anchor.
- Share savestates across regions. A PAL savestate will not load on NTSC-U even if the VI count matches.
- Skip the JSON sidecar. The VI count is not optional.
- Use Dolphin's slot 0 (slot 0 is commonly overwritten by accident). Slot 1+.
