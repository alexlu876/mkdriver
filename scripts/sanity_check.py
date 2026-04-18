#!/usr/bin/env python3
"""Sanity-check visualizer — renders the first N seconds of a paired
(.dtm + frame dump) recording with controller-state overlay.

Usage:
    uv run python scripts/sanity_check.py \\
        --dtm data/raw/demos/2026-04-16.dtm \\
        --frames data/raw/frames/2026-04-16/ \\
        --output sanity_out/overlay.mp4

Watch the output. If steering tracks the kart's turn direction, A lights up
during acceleration, and R lights up during drifts — the parser + pairing
are correct, and Phase 1 is acceptable. If any of those are off, stop and
debug before Phase 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mkw_rl.dtm.pairing import pair_dtm_and_frames
from mkw_rl.dtm.viz import write_overlay_video


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtm", required=True, type=Path, help="Path to the .dtm input file.")
    p.add_argument(
        "--frames",
        required=True,
        type=Path,
        help="Path to the frame dump directory (PNG sequence).",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the output overlay MP4.",
    )
    p.add_argument(
        "--seconds",
        type=int,
        default=30,
        help="Duration in seconds to render (default: 30). Use -1 for full length.",
    )
    p.add_argument(
        "--skip-first-n",
        type=int,
        default=0,
        help="Drop this many initial input frames (for menu/countdown padding).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Playback FPS (PAL MKWii native = 50).",
    )
    args = p.parse_args()

    n_seconds = None if args.seconds < 0 else args.seconds

    print(f"[sanity_check] parsing {args.dtm}")
    print(f"[sanity_check] loading frames from {args.frames}")
    pairs = pair_dtm_and_frames(args.dtm, args.frames, skip_first_n=args.skip_first_n)
    print(f"[sanity_check] paired {len(pairs)} samples")
    if not pairs:
        print("[sanity_check] no paired samples — aborting")
        return 1

    print(f"[sanity_check] rendering overlay to {args.output}")
    out = write_overlay_video(pairs, args.output, fps=args.fps, n_seconds=n_seconds)
    print(f"[sanity_check] wrote {out}")
    print(
        "\n[sanity_check] WATCH THIS VIDEO. "
        "Acceptance criterion for Phase 1: steering / A / R overlays "
        "visibly match kart behavior. If not, stop and debug."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
