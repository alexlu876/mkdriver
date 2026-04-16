#!/usr/bin/env python3
"""Preflight check runner for the mkw-rl project.

Runs the mechanical parts of the P-1 checklist from MKW_RL_SPEC.md §P-1.
The human-driven steps (booting the game, recording savestates, verifying
on-screen behavior) are in docs/PREFLIGHT.md — this script is the thin
automated wrapper.

Usage:
    python scripts/preflight.py [--dolphin PATH] [--frames-dir PATH]

All checks are optional and independently reportable. Missing args skip
the relevant check rather than erroring. At the end, the script prints
a compact summary and exits nonzero iff any attempted check failed.

This script is intentionally dependency-free (stdlib only) so it can run
before `uv init` and before `uv add torch …`.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Default macOS Dolphin frame-dump location.
DEFAULT_FRAMES_DIR = Path.home() / "Library" / "Application Support" / "Dolphin" / "Dump" / "Frames"

# Minimum Python we care about.
MIN_PYTHON = (3, 13)


@dataclass
class CheckResult:
    name: str
    status: str  # "pass", "fail", "skip", "warn"
    detail: str = ""
    data: dict = field(default_factory=dict)


def check_python_version() -> CheckResult:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    detail = f"running {v.major}.{v.minor}.{v.micro} ({sys.executable})"
    return CheckResult(
        name="python-version",
        status="pass" if ok else "fail",
        detail=detail,
        data={"executable": sys.executable, "version": f"{v.major}.{v.minor}.{v.micro}"},
    )


def check_platform() -> CheckResult:
    system = platform.system()
    machine = platform.machine()
    is_mac_arm = system == "Darwin" and machine == "arm64"
    detail = f"{system} {machine}"
    if is_mac_arm:
        return CheckResult("platform", "pass", detail)
    # Not fatal — spec targets M4 Mac Mini but lets the user decide.
    return CheckResult("platform", "warn", detail + " (spec targets Darwin arm64)")


def check_dolphin_binary(dolphin_path: str | None) -> CheckResult:
    if not dolphin_path:
        return CheckResult("dolphin-binary", "skip", "no --dolphin provided")
    resolved = shutil.which(dolphin_path) or (dolphin_path if Path(dolphin_path).exists() else None)
    if not resolved:
        return CheckResult("dolphin-binary", "fail", f"not found: {dolphin_path}")
    try:
        out = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("dolphin-binary", "fail", f"{resolved} --version failed: {exc}")
    combined = (out.stdout + out.stderr).strip()
    # VIPTankz's scripting fork advertises the scripting capability somewhere
    # in its --version string. If the fork's README says otherwise, adjust.
    has_scripting_marker = "scripting" in combined.lower()
    if not combined:
        return CheckResult(
            "dolphin-binary",
            "fail",
            f"{resolved} --version produced no output",
            data={"binary": resolved},
        )
    return CheckResult(
        name="dolphin-binary",
        status="pass" if has_scripting_marker else "warn",
        detail=combined.splitlines()[0] if combined else "",
        data={"binary": resolved, "scripting_marker_found": has_scripting_marker, "full": combined},
    )


def check_frame_dump_dir(frames_dir: Path) -> CheckResult:
    if not frames_dir.exists():
        return CheckResult(
            "frame-dump-dir",
            "warn",
            f"{frames_dir} does not exist yet (expected after first recording)",
        )
    pngs = sorted(frames_dir.rglob("*.png"))
    if not pngs:
        return CheckResult(
            "frame-dump-dir",
            "warn",
            f"{frames_dir} exists but contains no PNGs — record a short session per step 5 of PREFLIGHT.md",
        )
    return CheckResult(
        "frame-dump-dir",
        "pass",
        f"{frames_dir}: {len(pngs)} PNG(s) present (latest: {pngs[-1].name})",
        data={"count": len(pngs), "dir": str(frames_dir)},
    )


def check_savestate_dir() -> CheckResult:
    # Two locations Dolphin might use on macOS — either is acceptable.
    candidates = [
        Path.home() / "Library" / "Application Support" / "Dolphin" / "StateSaves",
        Path.home() / "Documents" / "Dolphin Emulator" / "StateSaves",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        return CheckResult(
            "savestate-dir",
            "warn",
            "no StateSaves directory found yet — expected after first savestate",
        )
    saves = []
    for p in found:
        saves.extend(p.glob("*.sav"))
        saves.extend(p.glob("*.s*"))
    return CheckResult(
        "savestate-dir",
        "pass" if saves else "warn",
        f"{len(saves)} state file(s) across {len(found)} dir(s)",
        data={"dirs": [str(p) for p in found], "count": len(saves)},
    )


def print_summary(results: list[CheckResult]) -> int:
    width = max(len(r.name) for r in results)
    print()
    print("=" * (width + 40))
    print("PREFLIGHT SUMMARY")
    print("=" * (width + 40))
    fail_count = 0
    for r in results:
        symbol = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skip": "SKIP"}[r.status]
        print(f"  [{symbol}] {r.name.ljust(width)}  {r.detail}")
        if r.status == "fail":
            fail_count += 1
    print("=" * (width + 40))
    print()
    print("Mechanical checks complete. These do NOT cover the six-step human")
    print("checklist in docs/PREFLIGHT.md — you must still work through that")
    print("by hand. In particular, this script cannot verify:")
    print("  - whether the fork builds (step 1)")
    print("  - whether NTSC-U MKWii boots (step 2)")
    print("  - whether the scripting API works (step 3)")
    print("  - which Python the fork links against (step 4)")
    print("  - savestate load determinism (step 6)")
    print()
    print("After all six human checks pass, report back to the user-facing")
    print("operator with the values listed at the end of docs/PREFLIGHT.md.")
    return 1 if fail_count else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dolphin", help="path to the VIPTankz dolphin-emu binary (optional)")
    parser.add_argument(
        "--frames-dir",
        default=str(DEFAULT_FRAMES_DIR),
        help=f"frame dump directory (default: {DEFAULT_FRAMES_DIR})",
    )
    args = parser.parse_args()

    results = [
        check_python_version(),
        check_platform(),
        check_dolphin_binary(args.dolphin),
        check_frame_dump_dir(Path(args.frames_dir).expanduser()),
        check_savestate_dir(),
    ]
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
