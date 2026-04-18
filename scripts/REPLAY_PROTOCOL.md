# Manual replay protocol

Fallback for when `scripts/replay_demo.py` doesn't fully automate — typically because the VIPTankz fork's Python API for movie playback differs from what the driver template expects.

Use this if `replay_demo.py` errors out, or until the driver is verified against real hardware.

## Ordering is load-bearing

Frame 0 of the dump must correspond to input 0 of the `.dtm`. This only works if:

1. No emulation is running before you enable Dump Frames.
2. The dump directory is empty when you start.
3. You let Dolphin auto-load the savestate via `Movie → Play Input` — do NOT separately `File → Load State` first. `Play Input` reads the `from_savestate` flag in the `.dtm` and loads the adjacent `<name>.dtm.sav` (or prompts for one). Separately loading a state first makes the dump include pre-.dtm frames.

Violating any of these silently shifts frame-to-input alignment by tens of frames. The sanity visualizer will show the misalignment only if you watch frame 0 and confirm it's the pre-input anchor.

## Steps

1. **Place the savestate next to the `.dtm`** so Dolphin's `Play Input` picks it up automatically:
   ```bash
   cp data/savestates/luigi_circuit_tt.sav data/raw/demos/<demo_timestamp>.dtm.sav
   ```
   (Dolphin looks for `<dtm-filename>.sav` alongside the `.dtm`.)

2. **Launch Dolphin fresh.** No ROM loaded. No emulation running. If Dolphin is already open with an emulation session, `Emulation → Stop` first.

3. **Clear any previous dump** so the new replay starts from an empty directory:
   ```bash
   rm -rf ~/Library/Application\ Support/Dolphin/Dump/Frames/*
   ```

4. **Enable Dump Frames now**, while no emulation is running. `Config → Graphics → Advanced`:
   - `Dump Frames` ON
   - `Dump Frames as Images` ON (forces PNG over AVI)

5. Load the ISO: `File → Open` → your PAL MKWii ISO.

6. **Immediately pause**: `Emulation → Pause`. Dolphin will boot into the title screen but we want to block it before it produces pre-.dtm frames.

7. `Movie → Play Input` → select your `.dtm`. Dolphin reads `from_savestate=True` from the header and loads `<demo>.dtm.sav` automatically. Emulation resumes and the first produced frame corresponds to input 0 of the `.dtm`.

8. Wait for playback to finish. The game may continue running after the `.dtm` exhausts; `Emulation → Stop` once the final recorded input has played (the Dolphin title bar shows the current frame index).

9. Move the dumped PNGs into `data/raw/frames/<demo_timestamp>/`:
   ```bash
   mkdir -p data/raw/frames/2026-04-16_120000
   # Dolphin may write to a game-ID subdir — move whatever structure it used.
   cp -R ~/Library/Application\ Support/Dolphin/Dump/Frames/ data/raw/frames/2026-04-16_120000/
   ```

10. Run the sanity visualizer:
    ```bash
    uv run python scripts/sanity_check.py \
      --dtm data/raw/demos/2026-04-16_120000.dtm \
      --frames data/raw/frames/2026-04-16_120000/ \
      --output sanity_out/2026-04-16_120000.mp4
    ```

11. **Open the MP4 and confirm alignment at frame 0.** The first frame of the video must show the same kart position as your savestate's anchor (the first-input-applies moment, per `docs/SAVESTATE_PROTOCOL.md`). If frame 0 is a title screen / menu / countdown frame, your dump includes pre-.dtm content — alignment is off by N frames. Re-do steps 2-9 with a cleaner ordering.

12. Through the rest of the video: the steering overlay should match the kart's turning, A should light up during acceleration, R should light up during drifts.

## Known gotchas

- **Emulator feels slow during replay**: expected. Frame dumping is heavy. Do not panic.
- **Fewer PNGs than input frames in the `.dtm`**: expected (lag frames don't render). The pairing module trims to `min(len)`; small discrepancies are fine.
- **Way fewer PNGs (e.g., half)**: something went wrong. Probably Dolphin's dump buffer overflowed or you stopped emulation before playback completed. Re-record or let the replay run to completion.
- **PNGs named oddly**: Dolphin has used multiple naming schemes over the years. The loader sorts numerically on any sequence of digits in the filename, so `framedump_1.png`, `frame_0001.png`, and plain `1.png` all work.
- **PNGs in `Dump/Frames/RMCP01/` subdir, not `Dump/Frames/` directly**: expected on some Dolphin versions. The loader uses recursive glob (`rglob`) so both layouts are handled.
- **Dolphin complains the `.sav` doesn't match the `.dtm`**: the `.dtm` has a hash of the savestate in its header. If the savestate was made under a different Dolphin SHA, the hash won't match. Re-anchor the savestate per `docs/SAVESTATE_PROTOCOL.md`.
