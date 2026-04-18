"""Frame dump reader.

Dolphin's frame dump on macOS outputs a PNG sequence to
``~/Library/Application Support/Dolphin/Dump/Frames/`` (or a subdir
named after the game). We load PNGs directly via PIL — do not run
ffmpeg over them, do not convert to JPEG, do not down-quantize. Lossy
compression on training data is a known generalization hazard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Recognize Dolphin's common frame dump naming schemes. Dolphin has used
# a few over the years: ``framedump_<N>.png``, ``frame_<N>.png``, plain
# ``<N>.png``. We match anything that sorts numerically and has a .png
# extension.
_FRAME_INDEX_RE = re.compile(r"(\d+)")


@dataclass
class FrameDump:
    """A loaded frame dump directory.

    Attributes:
        frame_dir: Directory containing the PNG sequence.
        frame_paths: Sorted list of PNG paths in emission order.
    """

    frame_dir: Path
    frame_paths: list[Path]

    def __len__(self) -> int:
        return len(self.frame_paths)


def _frame_sort_key(p: Path) -> tuple[int, str]:
    """Sort PNGs by the numeric portion of the filename (falls back to name)."""
    m = _FRAME_INDEX_RE.search(p.stem)
    if m:
        return (int(m.group(1)), p.name)
    # If no number is present, push to the end.
    return (10**12, p.name)


def load_frame_dump(frame_dir: Path | str) -> FrameDump:
    """Enumerate PNGs in ``frame_dir`` in emission order.

    Does not load pixel data — use ``load_frame`` for that. This is a
    lightweight index so downstream pipelines can decide which frames
    they actually need.

    Recursive: Dolphin commonly writes frame dumps to a game-ID subdirectory
    (e.g. ``Dump/Frames/RMCP01/framedump_0.png``), not directly in
    ``Dump/Frames/``. We use rglob so either layout works.
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        raise FileNotFoundError(f"frame directory does not exist: {frame_dir}")
    if not frame_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {frame_dir}")

    pngs = sorted(frame_dir.rglob("*.png"), key=_frame_sort_key)
    if not pngs:
        raise FileNotFoundError(f"no PNGs in {frame_dir} (searched recursively)")

    return FrameDump(frame_dir=frame_dir, frame_paths=pngs)


def load_frame(
    path: Path | str,
    size: tuple[int, int] = (140, 75),
    grayscale: bool = True,
) -> np.ndarray:
    """Load one frame and return it as a numpy array.

    Args:
        path: PNG path.
        size: Target (width, height). Dolphin dumps at native resolution
            (up to 640×528 + upscales) which we downsize here to match
            the BTR-paper 140×75 grayscale spec.
        grayscale: If True, return shape (H, W) uint8. If False, (H, W, 3).

    Returns:
        numpy array. uint8 in [0, 255]. Caller is responsible for
        normalizing to float [0, 1] and stacking.
    """
    path = Path(path)
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    # PIL Image.resize uses (width, height).
    img = img.resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    # For grayscale, arr is (H, W). For RGB, (H, W, 3). Match that.
    return arr
