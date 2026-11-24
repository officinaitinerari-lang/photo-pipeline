"""Font loading with graceful fallbacks and letter-spaced (tracked) text rendering."""

from __future__ import annotations

import contextlib
import logging
from functools import lru_cache
from pathlib import Path

from PIL import ImageDraw, ImageFont

log = logging.getLogger(__name__)

FontT = ImageFont.FreeTypeFont | ImageFont.ImageFont

SERIF_CANDIDATES = ("Cinzel[wght].ttf", "Cinzel-Regular.ttf", "PlayfairDisplay[wght].ttf")
SANS_CANDIDATES = ("Montserrat[wght].ttf", "Montserrat-Light.ttf")
SYSTEM_FALLBACKS = {
    "serif": ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "DejaVuSerif.ttf"),
    "sans": ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"),
}


@lru_cache(maxsize=64)
def load_font(fonts_dir: str, family: str, size: int, weight: int | None = None) -> FontT:
    """family: 'serif' (Cinzel/Playfair) or 'sans' (Montserrat). weight applies to variable fonts."""
    candidates = SERIF_CANDIDATES if family == "serif" else SANS_CANDIDATES
    for name in candidates:
        path = Path(fonts_dir) / name
        if path.is_file():
            try:
                font = ImageFont.truetype(str(path), size)
                if weight is not None:
                    # static font or FreeType without variation support: keep the default instance
                    with contextlib.suppress(OSError, ValueError, AttributeError):
                        font.set_variation_by_axes([weight])
                return font
            except OSError:
                log.warning("font unreadable", extra={"font": str(path)})
    for fallback in SYSTEM_FALLBACKS[family]:
        try:
            return ImageFont.truetype(fallback, size)
        except OSError:
            continue
    log.warning("no TTF font found, using Pillow default font", extra={"family": family})
    return ImageFont.load_default(size=size)


def tracked_width(draw: ImageDraw.ImageDraw, text: str, font: FontT, tracking: float) -> float:
    """Width of `text` with extra spacing of `tracking` x font-size between characters."""
    if not text:
        return 0.0
    size = getattr(font, "size", 10)
    widths = [draw.textlength(ch, font=font) for ch in text]
    return float(sum(widths) + tracking * size * (len(text) - 1))


def draw_tracked(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    top: float,
    text: str,
    font: FontT,
    fill: str | tuple[int, int, int],
    tracking: float,
) -> None:
    size = getattr(font, "size", 10)
    x = center_x - tracked_width(draw, text, font, tracking) / 2
    for ch in text:
        draw.text((x, top), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking * size


def fit_font(
    draw: ImageDraw.ImageDraw,
    fonts_dir: str,
    family: str,
    text: str,
    start_size: int,
    max_width: float,
    tracking: float,
    weight: int | None = None,
    min_size: int = 8,
) -> FontT:
    size = start_size
    font = load_font(fonts_dir, family, size, weight)
    while size > min_size and tracked_width(draw, text, font, tracking) > max_width:
        size = int(size * 0.92)
        font = load_font(fonts_dir, family, size, weight)
    return font


def text_height(font: FontT, sample: str = "AÉgy") -> int:
    bbox = font.getbbox(sample)
    return int(bbox[3] - bbox[1])
