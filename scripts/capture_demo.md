# Capturing a demo

This is the user-facing protocol for recording a `.dtm` input log suitable for the BC pipeline.

**Critical rule**: record first, replay with frame dumping second. Never record with frame dumping enabled — it slows the emulator below realtime, and the resulting inputs will not match clean-speed replay. See MKW_RL_SPEC.md §1.2.

## Setup (per session)

1. Launch VIPTankz's Dolphin scripting fork.
2. Load your NTSC-U MKWii ISO (`RMCE01`).
3. `Config → Graphics → Advanced`. Ensure **Dump Frames is OFF**. Recording has nothing to do with dumping; these are two separate Dolphin features and they do not compose well during recording.
4. Verify the controller profile: `Controllers → GameCube Controllers → Port 1 → Standard Controller`. The `.dtm` format only supports GameCube controllers in this project (see MKW_RL_SPEC.md Fixed Assumptions).

## Recording

1. `File → Load State` → select your anchor, e.g. `data/savestates/luigi_circuit_tt.sav`. The game should land at the exact VI count documented in `data/savestates/luigi_circuit_tt.json`.
2. `Movie → Start Recording Input`. Dolphin will prompt you; use "Record from existing state" so the `.dtm` header's `from_savestate` flag is set correctly.
3. Play the race. Any finishing time is acceptable — we want demonstrations from a range of driving quality. Crashed or failed laps are not inherently bad demos; they teach the model what recovery looks like.
4. When you cross the finish line (or you've decided the demo is done), `Movie → Stop Recording`.
5. Save the resulting `.dtm` to `data/raw/demos/<YYYY-MM-DD_HHMMSS>.dtm`. Use a timestamp name so demos don't collide.

## Quick sanity check (no replay yet)

```bash
uv run python -c "
from mkw_rl.dtm.parser import parse_dtm
header, states = parse_dtm('data/raw/demos/<your_file>.dtm')
print(f'author: {header.author!r}')
print(f'game_id: {header.game_id}')
print(f'from_savestate: {header.from_savestate}')
print(f'input frames: {len(states)}')
print(f'vi_count in header: {header.vi_count}')
print(f'lag_count in header: {header.lag_count}')
"
```

Expected:

- `game_id` == `b'RMCE01'`.
- `from_savestate` == `True`.
- `input frames` in the thousands (a 90-second lap at 60fps is ~5400 frames).
- `lag_count` ideally 0 — if nonzero, log it in your notes. The demo is still usable but may have alignment issues.

If parsing fails at this point: the `.dtm` is malformed, or you've recorded against a non-NTSC-U copy. Stop and re-record.

## Next: replay to produce paired frames

After recording, run [scripts/replay_demo.py](replay_demo.py) (or the manual `REPLAY_PROTOCOL.md` fallback if automation isn't wired up yet) to replay the `.dtm` with frame dumping enabled. That produces the `data/raw/frames/<timestamp>/` PNG sequence that pairs with the `.dtm`.

## Multi-demo batching

For Phase 2 we target 20+ demos of Luigi Circuit. Don't record 20 in a row — fatigue shows up in the inputs. Record in batches of 5 across multiple sessions for a more natural distribution.

## Attribution of TAS demos

If you're also importing `.dtm` files from the MKWii TAS community, store them in `data/raw/tas/`, not `data/raw/demos/`. Record attribution in `data/raw/tas/ATTRIBUTION.md` with source URL, author, and license. TAS demos may not have `from_savestate=True` — they might start from power-on. Those are **not interchangeable with your recordings** and should not be pooled naively; see MKW_RL_SPEC.md §2.1.
