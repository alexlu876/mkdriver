# Manual replay protocol

Fallback for when `scripts/replay_demo.py` doesn't fully automate — typically because the VIPTankz fork's Python API for movie playback differs from what the driver template expects.

Use this if `replay_demo.py` errors out, or until the driver is verified against real hardware.

## Steps

1. In Dolphin: `Config → Graphics → Advanced`. Enable:
   - `Dump Frames`
   - `Dump Frames as Images`

2. Clear any previous dump so you can tell what this replay produced:
   ```bash
   rm ~/Library/Application\ Support/Dolphin/Dump/Frames/*.png
   ```

3. In Dolphin: `File → Load State` → select the savestate that was active when the `.dtm` was recorded.

4. `Movie → Play Input` → select your `.dtm`. Dolphin will begin playback.

5. Wait for playback to finish. The game will continue running after the `.dtm` exhausts; `Emulation → Stop` once the final recorded input has played (the Dolphin title bar shows the current frame index).

6. Move the dumped PNGs into `data/raw/frames/<demo_timestamp>/`:
   ```bash
   mkdir -p data/raw/frames/2026-04-16_120000
   mv ~/Library/Application\ Support/Dolphin/Dump/Frames/*.png data/raw/frames/2026-04-16_120000/
   ```

7. Run the sanity visualizer:
   ```bash
   uv run python scripts/sanity_check.py \
     --dtm data/raw/demos/2026-04-16_120000.dtm \
     --frames data/raw/frames/2026-04-16_120000/ \
     --output sanity_out/2026-04-16_120000.mp4
   ```

8. Open the MP4 and watch it. The steering overlay should match the kart's turning, A should light up during acceleration, R should light up during drifts.

## Known gotchas

- **Emulator feels slow during replay**: expected. Frame dumping is heavy. Do not panic.
- **Fewer PNGs than input frames in the `.dtm`**: expected (lag frames don't render). The pairing module trims to `min(len)`; small discrepancies are fine.
- **Way fewer PNGs (e.g., half)**: something went wrong. Probably Dolphin's dump buffer overflowed or you stopped emulation before playback completed. Re-record or let the replay run to completion.
- **PNGs named oddly**: Dolphin has used multiple naming schemes over the years. The loader sorts numerically on any sequence of digits in the filename, so `framedump_1.png`, `frame_0001.png`, and plain `1.png` all work.
