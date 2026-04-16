"""Sanity visualizer — overlay controller state on rendered frames.

This is the acceptance criterion for Phase 1 per MKW_RL_SPEC.md §1.5:
if the overlay video matches on-screen kart behavior (steering arrow
tracks the turns, A lights up during acceleration, R lights up during
drifts), then the parser + pairing are working. If it doesn't, nothing
downstream can be trusted.

The overlay is drawn with PIL on the full-resolution frame (not the
downsized 140×114 network input) so the user has a clear visual
reference. Frames are composited into an MP4 via imageio.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .pairing import PairedSample

# Overlay style constants. Deliberately generous (thick, high-contrast)
# so the overlay reads on screen at normal playback speed.
_BAR_HEIGHT = 18
_BAR_PAD = 12
_DOT_RADIUS = 12
_DOT_SPACING = 10
_FONT_SIZE = 20

_COLOR_BG = (0, 0, 0, 180)  # translucent black
_COLOR_BAR = (255, 255, 255, 220)
_COLOR_BAR_FILL = (80, 200, 255, 255)  # steering fill
_COLOR_A = (70, 220, 90, 255)
_COLOR_B = (230, 70, 70, 255)
_COLOR_R = (80, 160, 255, 255)
_COLOR_L = (240, 220, 60, 255)
_COLOR_X = (200, 120, 240, 255)
_COLOR_TEXT = (255, 255, 255, 255)

_BUTTON_SPECS: list[tuple[str, str, tuple[int, int, int, int]]] = [
    # (attribute, label, color)
    ("accelerate", "A", _COLOR_A),
    ("brake", "B", _COLOR_B),
    ("drift", "R", _COLOR_R),
    ("item", "L", _COLOR_L),
    ("look_behind", "X", _COLOR_X),
]


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Best-effort font load. Falls back to PIL's default bitmap font."""
    # Try a couple of known macOS system fonts.
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, _FONT_SIZE)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def render_overlay(sample: PairedSample, font: ImageFont.ImageFont | None = None) -> Image.Image:
    """Render one frame with the controller-state overlay.

    Returns a new RGB ``PIL.Image`` at the original frame's resolution.
    """
    img = Image.open(sample.frame_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    font = font or _load_font()

    # ---------- Steering bar across the bottom ----------
    bar_y0 = h - _BAR_HEIGHT - _BAR_PAD
    bar_y1 = bar_y0 + _BAR_HEIGHT
    bar_x0 = _BAR_PAD
    bar_x1 = w - _BAR_PAD

    # Background box.
    draw.rectangle([bar_x0 - 4, bar_y0 - 4, bar_x1 + 4, bar_y1 + 4], fill=_COLOR_BG)
    # Outline.
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], outline=_COLOR_BAR, width=1)

    center_x = (bar_x0 + bar_x1) // 2
    steering = max(-1.0, min(1.0, sample.controller.steering))
    half = (bar_x1 - bar_x0) / 2
    if steering >= 0:
        fx0, fx1 = center_x, center_x + int(steering * half)
    else:
        fx0, fx1 = center_x + int(steering * half), center_x
    if fx1 - fx0 > 0:
        draw.rectangle([fx0, bar_y0 + 2, fx1, bar_y1 - 2], fill=_COLOR_BAR_FILL)
    # Center tick.
    draw.line([(center_x, bar_y0 - 4), (center_x, bar_y1 + 4)], fill=_COLOR_BAR, width=1)

    # ---------- Button dots, bottom-right row ----------
    dot_y = bar_y0 - _DOT_RADIUS * 2 - _BAR_PAD
    dot_x = w - _BAR_PAD - _DOT_RADIUS
    for attr, label, color in reversed(_BUTTON_SPECS):
        active = getattr(sample.controller, attr)
        fill = color if active else (*color[:3], 40)
        outline = color if active else (*color[:3], 80)
        draw.ellipse(
            [
                dot_x - _DOT_RADIUS,
                dot_y - _DOT_RADIUS,
                dot_x + _DOT_RADIUS,
                dot_y + _DOT_RADIUS,
            ],
            fill=fill,
            outline=outline,
            width=2,
        )
        # Label in white on active, gray otherwise. Center via text bbox.
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (dot_x - tw // 2 - bbox[0], dot_y - th // 2 - bbox[1]),
            label,
            fill=_COLOR_TEXT if active else (200, 200, 200, 160),
            font=font,
        )
        dot_x -= _DOT_RADIUS * 2 + _DOT_SPACING

    # ---------- Frame index, top-left ----------
    label = f"frame {sample.frame_idx}  input {sample.input_frame_idx}"
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 6
    tx = _BAR_PAD
    ty = _BAR_PAD
    draw.rectangle(
        [
            tx - pad,
            ty - pad + bbox[1],
            tx + (bbox[2] - bbox[0]) + pad,
            ty + (bbox[3] - bbox[1]) + pad,
        ],
        fill=_COLOR_BG,
    )
    draw.text((tx, ty), label, fill=_COLOR_TEXT, font=font)

    composed = Image.alpha_composite(img, overlay)
    return composed.convert("RGB")


def write_overlay_video(
    samples: list[PairedSample],
    output_path: Path | str,
    fps: int = 60,
    n_seconds: int | None = 30,
) -> Path:
    """Render an overlay MP4 from a list of PairedSamples.

    Args:
        samples: Samples in playback order.
        output_path: Destination .mp4 path.
        fps: Output frame rate. MKWii runs at 60 VIs per second; use 60 here
            unless you explicitly downsampled.
        n_seconds: If set, cap to the first ``n_seconds * fps`` frames. Use
            None for the full duration.

    Returns the output path (for chaining).
    """
    import imageio.v3 as iio  # deferred import — heavy

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font = _load_font()
    n_max = len(samples) if n_seconds is None else min(len(samples), fps * n_seconds)
    if n_max == 0:
        raise ValueError("no samples to render")

    frames_iter = (
        # imageio wants numpy arrays; PIL → array via .tobytes round-trip is fine here.
        _pil_to_ndarray(render_overlay(samples[i], font=font))
        for i in range(n_max)
    )
    iio.imwrite(output_path, list(frames_iter), fps=fps, codec="libx264", quality=7)
    return output_path


def _pil_to_ndarray(img: Image.Image):
    import numpy as np

    return np.asarray(img.convert("RGB"), dtype="uint8")
