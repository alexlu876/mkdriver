"""Clone dolphin0/ into dolphin1, dolphin2, ... for multi-env training.

VIPTankz's multi-env pattern requires one Dolphin install per parallel env
instance, because Felk's Dolphin fork ships a ``portable.txt`` marker in the
binary directory that forces the emulator to use ``./User/`` (shader cache,
config, savestates) relative to the binary. Two concurrent Dolphins pointing
at the same ``User/`` directory will fight over the cache files — even if
they never crash outright, they spend CPU rebuilding shaders every session.

The ``--user`` CLI flag does NOT override portable-mode cleanly (the marker
wins), so the only reliable isolation is: each env_id gets its own
``dolphin{i}/`` directory, a full copy of ``dolphin0/`` (~278 MB each).

Usage
-----
::

    # On Vast.ai, parent dir is /root/mkwii/third_party/Wii-RL.
    .venv/bin/python scripts/setup_dolphin_instances.py \\
        --parent /root/mkwii/third_party/Wii-RL --num-envs 4

    # Health-check existing clones (does not create or modify anything).
    # Catches corruption from interrupted copies, SIGSEGV damage, or manual
    # edits that broke the binary.
    .venv/bin/python scripts/setup_dolphin_instances.py \\
        --parent /root/mkwii/third_party/Wii-RL --num-envs 4 --verify

Idempotent — skips directories that already exist. Safe to re-run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


def _verify_install(path: Path) -> list[str]:
    """Return a list of human-readable problems with the given dolphin{i}/
    install. Empty list = healthy. Checks are cheap: exist + executable +
    non-empty binary + portable marker."""
    problems: list[str] = []
    if not path.exists():
        return [f"{path} does not exist"]
    if not path.is_dir():
        problems.append(f"{path} is not a directory")
        return problems
    binary = path / "dolphin-emu"
    if not binary.exists():
        problems.append(f"{binary} missing (dolphin-emu binary not found)")
    elif not os.access(binary, os.X_OK):
        problems.append(f"{binary} not executable (chmod +x it, or re-clone)")
    elif binary.stat().st_size < 1024:
        problems.append(f"{binary} is suspiciously small ({binary.stat().st_size} bytes)")
    portable = path / "portable.txt"
    if not portable.exists():
        problems.append(f"{portable} missing (Dolphin needs the portable marker)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="Directory containing the canonical dolphin0/. "
        "Sibling dolphin1/, dolphin2/, ... will be created here.",
    )
    ap.add_argument(
        "--num-envs",
        type=int,
        required=True,
        help="Total env count (including env_id=0). For num-envs=4 we create "
        "dolphin1/ through dolphin3/; dolphin0/ is expected to already exist.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Remove existing dolphin{i}/ directories before copying. "
        "Default is to skip existing dirs (idempotent). "
        "DESTRUCTIVE: wipes User/ (saved config, shader cache, any local "
        "mods). Use after a confirmed SIGSEGV where state is corrupt.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Health-check existing dolphin{i}/ dirs instead of cloning. "
        "Reports corruption (missing binary, non-executable, missing "
        "portable marker) and exits non-zero if any env is broken.",
    )
    args = ap.parse_args()

    src = args.parent / "dolphin0"
    if not src.exists():
        print(f"error: canonical source {src} not found", file=sys.stderr)
        return 1

    # Sanity-check the source looks like a Dolphin install (saves confusion if
    # --parent points somewhere weird and we'd silently copy 0 bytes).
    if not (src / "dolphin-emu").exists():
        print(f"error: {src} doesn't contain a dolphin-emu binary", file=sys.stderr)
        return 1

    if args.verify:
        total_problems = 0
        for i in range(args.num_envs):
            dst = args.parent / f"dolphin{i}"
            problems = _verify_install(dst)
            if problems:
                total_problems += len(problems)
                print(f"[{i}/{args.num_envs - 1}] {dst}: UNHEALTHY")
                for p in problems:
                    print(f"  - {p}")
            else:
                print(f"[{i}/{args.num_envs - 1}] {dst}: ok")
        if total_problems:
            print(
                f"\n{total_problems} problem(s) across {args.num_envs} envs. "
                f"Re-clone broken envs with --force.",
                file=sys.stderr,
            )
            return 2
        print(f"\nall {args.num_envs} Dolphin instances look healthy")
        return 0

    for i in range(1, args.num_envs):
        dst = args.parent / f"dolphin{i}"
        if dst.exists():
            if args.force:
                print(f"[{i}/{args.num_envs - 1}] removing existing {dst}")
                shutil.rmtree(dst)
            else:
                # Quick health check — if the existing dir is broken, warn
                # loudly so the user runs --force to re-clone.
                problems = _verify_install(dst)
                if problems:
                    print(
                        f"[{i}/{args.num_envs - 1}] WARNING: {dst} exists but "
                        f"looks broken. Run with --force to re-clone. Problems:",
                        file=sys.stderr,
                    )
                    for p in problems:
                        print(f"  - {p}", file=sys.stderr)
                else:
                    print(f"[{i}/{args.num_envs - 1}] skipping {dst} (exists, looks healthy)")
                continue
        t0 = time.time()
        print(f"[{i}/{args.num_envs - 1}] copying {src} → {dst} ...", flush=True)
        # copytree preserves permissions (the dolphin-emu binary needs +x).
        # symlinks=False: we want real copies so each env's User/ is independent.
        shutil.copytree(src, dst, symlinks=False)
        dt = time.time() - t0
        print(f"[{i}/{args.num_envs - 1}] done in {dt:.1f}s", flush=True)

    print(f"ready: {args.num_envs} Dolphin instances at {args.parent}/dolphin{{0..{args.num_envs - 1}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
