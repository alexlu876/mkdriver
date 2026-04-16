#!/usr/bin/env python3
"""End-to-end: .dtm + frame dir → pickled sequence dataset.

Usage:
    uv run python scripts/parse_demo.py \\
        --dtm data/raw/demos/2026-04-16.dtm \\
        --frames data/raw/frames/2026-04-16/ \\
        --output data/processed/user_demos/2026-04-16.pkl \\
        --skip-first-n 300

Produces a pickled dict {"samples_by_demo": {...}, "meta": {...}} that
mkw_rl.dtm.dataset.MkwBCDataset can consume directly.

To aggregate multiple demos into one dataset, pass --append <existing.pkl>
and the new demo is merged into that file's samples_by_demo.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from mkw_rl.dtm.dataset import demo_id_from_path
from mkw_rl.dtm.pairing import pair_dtm_and_frames


def _skip_first_n_from_savestate(savestate_json: Path) -> int:
    """Read skip_first_n from a savestate JSON sidecar (docs/SAVESTATE_PROTOCOL.md)."""
    with savestate_json.open("r") as f:
        payload = json.load(f)
    if "skip_first_n" not in payload:
        raise KeyError(f"{savestate_json} has no 'skip_first_n' key")
    return int(payload["skip_first_n"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtm", required=True, type=Path)
    p.add_argument("--frames", required=True, type=Path, help="Frame dump directory (PNG sequence).")
    p.add_argument("--output", required=True, type=Path, help="Output .pkl path.")
    p.add_argument(
        "--savestate-json",
        type=Path,
        default=None,
        help="Path to the savestate's JSON sidecar (data/savestates/<slug>.json). "
        "If set, skip_first_n is read from the sidecar unless --skip-first-n is also given.",
    )
    p.add_argument(
        "--skip-first-n",
        type=int,
        default=None,
        help="Override skip_first_n. If both this and --savestate-json are given, this wins.",
    )
    p.add_argument("--tail-margin", type=int, default=10)
    p.add_argument("--demo-id", type=str, default=None, help="Override demo id (default: .dtm stem).")
    p.add_argument(
        "--append",
        type=Path,
        default=None,
        help="Existing .pkl to merge the new demo into. If omitted, a fresh file is written.",
    )
    args = p.parse_args()

    # Resolve skip_first_n (M-4 audit fix): CLI override > sidecar > 0.
    if args.skip_first_n is not None:
        skip_first_n = args.skip_first_n
        print(f"[parse_demo] skip_first_n={skip_first_n} (from --skip-first-n)")
    elif args.savestate_json is not None:
        skip_first_n = _skip_first_n_from_savestate(args.savestate_json)
        print(f"[parse_demo] skip_first_n={skip_first_n} (from {args.savestate_json})")
    else:
        skip_first_n = 0
        print("[parse_demo] skip_first_n=0 (default — no sidecar, no override)")

    demo_id = args.demo_id or demo_id_from_path(args.dtm)

    print(f"[parse_demo] parsing {args.dtm}, loading frames from {args.frames}")
    pairs = pair_dtm_and_frames(
        args.dtm,
        args.frames,
        skip_first_n=skip_first_n,
        tail_margin=args.tail_margin,
    )
    print(f"[parse_demo] demo_id={demo_id}, {len(pairs)} paired samples")

    if args.append and args.append.exists():
        with args.append.open("rb") as f:
            payload = pickle.load(f)
        samples_by_demo = payload["samples_by_demo"]
        if demo_id in samples_by_demo:
            print(
                f"[parse_demo] warning: {demo_id} already in {args.append}; "
                "overwriting. Use a unique --demo-id if you don't want this."
            )
        samples_by_demo[demo_id] = pairs
        payload["samples_by_demo"] = samples_by_demo
    else:
        payload = {
            "samples_by_demo": {demo_id: pairs},
            "meta": {
                "skip_first_n": skip_first_n,
                "tail_margin": args.tail_margin,
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(payload, f)
    print(f"[parse_demo] wrote {args.output} ({len(payload['samples_by_demo'])} demos total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
