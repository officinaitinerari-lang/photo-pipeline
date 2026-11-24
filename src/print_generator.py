"""300 DPI print files: Full-Bleed (5 frame ratios) and Gallery Poster / Fine Art variant."""

from __future__ import annotations

import gc
import io
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .image_loader import srgb_icc_bytes
from .smart_crop import Focus, crop_box_for_ratio
from .typography import draw_tracked, fit_font, text_height

log = logging.getLogger(__name__)

PAPER = "#FAF9F6"
HAIRLINE = "#E5E2DC"
INK = "#2B2A28"
INK_SOFT = "#6F6B65"


@dataclass(frozen=True)
class PrintFormat:
    key: str
    ratio_label: str
    px_short: int
    px_long: int
    sizes_in: tuple[str, ...]
    sizes_cm: tuple[str, ...]
    largest_cm: tuple[float, float]  # (short, long) used by the infographic silhouettes

    @property
    def ratio(self) -> float:
        """short/long side ratio (portrait w/h)."""
        return self.px_short / self.px_long

    def pixels(self, orientation: str) -> tuple[int, int]:
        return (self.px_short, self.px_long) if orientation == "portrait" else (self.px_long, self.px_short)

    def native_inches(self, dpi: int) -> str:
        return f"{self.px_short / dpi:g}x{self.px_long / dpi:g}in"


FORMATS: tuple[PrintFormat, ...] = (
    PrintFormat("ratio_2x3", "2:3", 4800, 7200,
                ("4x6", "8x12", "12x18", "16x24", "20x30", "24x36"),
                ("10x15", "20x30", "40x60", "50x75", "60x90"), (60.0, 90.0)),
    PrintFormat("ratio_3x4", "3:4", 5400, 7200,
                ("6x8", "9x12", "12x16", "15x20", "18x24"),
                ("15x20", "30x40", "45x60"), (45.0, 60.0)),
    PrintFormat("ratio_4x5", "4:5", 4800, 6000,
                ("4x5", "8x10", "12x15", "16x20"),
                ("20x25", "40x50"), (40.0, 50.0)),
    PrintFormat("iso_A_series", "ISO A", 4961, 7016,
                ("A5", "A4", "A3", "A2"),
                ("14.8x21", "21x29.7", "29.7x42", "42x59.4"), (42.0, 59.4)),
    PrintFormat("ratio_11x14", "11:14", 3300, 4200,
                ("11x14",),
                ("28x35.6",), (28.0, 35.6)),
)
POSTER_FORMAT_KEYS = ("ratio_2x3", "iso_A_series")
FORMATS_BY_KEY = {f.key: f for f in FORMATS}


def save_print_jpeg(
    img: Image.Image, path: Path, dpi: int, quality: int, min_quality: int, max_bytes: int
) -> int:
    """Save as sRGB JPEG with DPI tag, lowering quality only if needed to respect max_bytes."""
    icc = srgb_icc_bytes()
    q = quality
    while True:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, subsampling=0 if q >= 90 else 2, optimize=True,
                 progressive=False, dpi=(dpi, dpi), icc_profile=icc)
        size = buf.tell()
        if size <= max_bytes or q <= min_quality:
            path.write_bytes(buf.getvalue())
            buf.close()
            if size > max_bytes:
                log.warning("print above size cap even at min quality", extra={"file": path.name, "bytes": size})
            return q
        buf.close()
        q -= 3


def render_fullbleed(img: Image.Image, fmt: PrintFormat, orientation: str, focus: Focus) -> Image.Image:
    tw, th = fmt.pixels(orientation)
    box = crop_box_for_ratio(img.width, img.height, tw / th, focus)
    cropped = img.crop(box)
    out = cropped.resize((tw, th), Image.Resampling.LANCZOS, reducing_gap=3.0)
    cropped.close()
    return out


@dataclass(frozen=True)
class PosterLayout:
    photo_box: tuple[int, int, int, int]
    title_top: int
    subtitle_top: int
    title_size: int
    subtitle_size: int
    hairline: int


def compute_poster_layout(width: int, height: int) -> PosterLayout:
    """Asymmetric passe-partout: equal sides/top, a deeper bottom band for the typography."""
    short = min(width, height)
    side = round(short * 0.085)
    top = side
    bottom = round(height * (0.17 if height >= width else 0.21))
    photo_box = (side, top, width - side, height - bottom)
    band_top = photo_box[3]
    title_size = round(short * 0.050)
    subtitle_size = round(short * 0.0165)
    gap = round(title_size * 0.55)
    block = title_size + gap + subtitle_size
    title_top = band_top + (bottom - block) // 2 - round(bottom * 0.04)
    return PosterLayout(
        photo_box=photo_box,
        title_top=title_top,
        subtitle_top=title_top + title_size + gap,
        title_size=title_size,
        subtitle_size=subtitle_size,
        hairline=max(2, round(short / 1600)),
    )


def render_gallery_poster(
    img: Image.Image, fmt: PrintFormat, orientation: str, focus: Focus, title: str, subtitle: str, fonts_dir: str
) -> Image.Image:
    W, H = fmt.pixels(orientation)
    layout = compute_poster_layout(W, H)
    x0, y0, x1, y1 = layout.photo_box
    pw, ph = x1 - x0, y1 - y0
    canvas = Image.new("RGB", (W, H), PAPER)
    box = crop_box_for_ratio(img.width, img.height, pw / ph, focus)
    cropped = img.crop(box)
    photo = cropped.resize((pw, ph), Image.Resampling.LANCZOS, reducing_gap=3.0)
    cropped.close()
    canvas.paste(photo, (x0, y0))
    photo.close()

    draw = ImageDraw.Draw(canvas)
    t = layout.hairline
    draw.rectangle((x0 - t, y0 - t, x1 + t - 1, y1 + t - 1), outline=HAIRLINE, width=t)

    max_text_w = W * 0.84
    title_font = fit_font(draw, fonts_dir, "serif", title, layout.title_size, max_text_w, tracking=0.32, weight=500)
    sub_font = fit_font(draw, fonts_dir, "sans", subtitle, layout.subtitle_size, max_text_w, tracking=0.22, weight=300)
    # vertically compensate if fit_font shrank the title
    title_shift = (layout.title_size - text_height(title_font)) // 3
    draw_tracked(draw, W / 2, layout.title_top + title_shift, title, title_font, INK, tracking=0.32)
    if subtitle:
        draw_tracked(draw, W / 2, layout.subtitle_top, subtitle, sub_font, INK_SOFT, tracking=0.22)
    return canvas


@dataclass
class PrintOutputs:
    fullbleed_files: list[Path]
    poster_files: list[Path]
    fullbleed_preview: Image.Image  # ~2400px copy of ratio_2x3 for mockups
    poster_preview: Image.Image
    crop_previews: dict[str, Image.Image]  # small thumbnails per format for the infographic


def _preview(img: Image.Image, long_side: int) -> Image.Image:
    p = img.copy()
    p.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    return p


def generate_prints(
    img: Image.Image,
    *,
    orientation: str,
    focus: Focus,
    slug: str,
    out_dir: Path,
    title: str,
    subtitle: str,
    fonts_dir: str,
    dpi: int,
    quality: int,
    min_quality: int,
    max_file_bytes: int,
) -> PrintOutputs:
    full_dir, poster_dir = out_dir / "fullbleed", out_dir / "gallery_poster"
    full_dir.mkdir(parents=True, exist_ok=True)
    poster_dir.mkdir(parents=True, exist_ok=True)
    fullbleed_files: list[Path] = []
    poster_files: list[Path] = []
    crop_previews: dict[str, Image.Image] = {}
    fullbleed_preview: Image.Image | None = None
    poster_preview: Image.Image | None = None

    for fmt in FORMATS:
        rendered = render_fullbleed(img, fmt, orientation, focus)
        w, h = rendered.size
        path = full_dir / f"{slug}_{fmt.key}_{w}x{h}px_{dpi}dpi.jpg"
        q = save_print_jpeg(rendered, path, dpi, quality, min_quality, max_file_bytes)
        fullbleed_files.append(path)
        crop_previews[fmt.key] = _preview(rendered, 600)
        if fmt.key == "ratio_2x3":
            fullbleed_preview = _preview(rendered, 2000)
        log.info("full-bleed print saved", extra={"file": path.name, "quality": q, "bytes": path.stat().st_size})
        rendered.close()
        gc.collect()

    for key in POSTER_FORMAT_KEYS:
        fmt = FORMATS_BY_KEY[key]
        rendered = render_gallery_poster(img, fmt, orientation, focus, title, subtitle, fonts_dir)
        w, h = rendered.size
        path = poster_dir / f"{slug}_gallery_{fmt.key}_{w}x{h}px_{dpi}dpi.jpg"
        q = save_print_jpeg(rendered, path, dpi, quality, min_quality, max_file_bytes)
        poster_files.append(path)
        if key == "ratio_2x3":
            poster_preview = _preview(rendered, 2000)
        log.info("gallery poster saved", extra={"file": path.name, "quality": q, "bytes": path.stat().st_size})
        rendered.close()
        gc.collect()

    if fullbleed_preview is None or poster_preview is None:
        raise RuntimeError("print previews were not generated")
    return PrintOutputs(fullbleed_files, poster_files, fullbleed_preview, poster_preview, crop_previews)
