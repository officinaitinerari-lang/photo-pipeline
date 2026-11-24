"""Procedural Etsy showcase images (2400x1800, sRGB) drawn entirely with Pillow + NumPy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .image_loader import srgb_icc_bytes
from .print_generator import FORMATS, INK, INK_SOFT, PAPER, PrintFormat
from .typography import draw_tracked, fit_font, load_font, text_height

log = logging.getLogger(__name__)

MOCKUP_SIZE = (2400, 1800)


# ------------------------------------------------------------------ procedural surfaces
def _smooth_noise(shape: tuple[int, int], scale: int, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    small = rng.standard_normal((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    lo, hi = float(small.min()), float(small.max())
    img = Image.fromarray(((small - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8), "L")
    big = np.asarray(img.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return big - 0.5


def make_wall(size: tuple[int, int], base_rgb: tuple[int, int, int], seed: int) -> Image.Image:
    """Soft natural-light gradient (window at upper left) + subtle plaster texture + vignette."""
    w, h = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = xx / w, yy / h
    light = 1.06 - 0.10 * np.sqrt((nx - 0.15) ** 2 + (ny - 0.05) ** 2)
    vignette = 1.0 - 0.07 * (((nx - 0.5) * 2) ** 2 + ((ny - 0.5) * 2) ** 2) / 2
    texture = 1.0 + 0.012 * _smooth_noise((h, w), 6, rng) + 0.008 * rng.standard_normal((h, w)).astype(np.float32)
    texture += 0.010 * _smooth_noise((h, w), 120, rng)
    factor = (light * vignette * texture)[..., None]
    arr = np.clip(np.array(base_rgb, dtype=np.float32)[None, None, :] * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def cast_shadow(wall: Image.Image, rect: tuple[int, int, int, int], offset: tuple[int, int], blur: float, strength: float) -> None:
    """Darken the wall in-place with a blurred, offset silhouette of `rect`."""
    mask = Image.new("L", wall.size, 0)
    x0, y0, x1, y1 = rect
    ImageDraw.Draw(mask).rectangle((x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1]), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur))
    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0 * strength
    arr = np.asarray(wall, dtype=np.float32) * (1.0 - alpha)
    wall.paste(Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB"))


def _fit_outer(art_aspect: float, mat: float, frame: float, max_w: float, max_h: float) -> float:
    """Scale s so that art(short side = s) + mat + frame fits max_w x max_h. mat/frame relative to s."""
    if art_aspect >= 1:  # landscape: short side is height
        unit_w, unit_h = art_aspect + 2 * (mat + frame), 1 + 2 * (mat + frame)
    else:
        unit_w, unit_h = 1 + 2 * (mat + frame), 1 / art_aspect + 2 * (mat + frame)
    return min(max_w / unit_w, max_h / unit_h)


def _wood_texture(length: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Light oak: grain lines running along `length`."""
    base = np.array([205, 170, 126], dtype=np.float32)
    yy, xx = np.mgrid[0:width, 0:length].astype(np.float32)
    warp = _smooth_noise((width, length), 60, rng) * 6 + _smooth_noise((width, length), 12, rng) * 1.5
    rings = np.sin((yy + warp) * (2 * np.pi / max(8.0, width / 4.0)))
    fibres = _smooth_noise((width, length), 2, rng)
    shade = 0.93 + 0.03 * rings + 0.04 * fibres + 0.02 * np.sin(xx / length * np.pi)
    return np.clip(base[None, None, :] * shade[..., None], 0, 255).astype(np.uint8)


def draw_mitred_frame(outer: tuple[int, int], fw: int, kind: str, seed: int) -> Image.Image:
    """Frame ring as RGBA (transparent centre) with mitred corners."""
    W, H = outer
    frame = Image.new("RGBA", outer, (0, 0, 0, 0))
    rng = np.random.default_rng(seed)
    pieces = {
        "top": [(0, 0), (W, 0), (W - fw, fw), (fw, fw)],
        "bottom": [(0, H), (W, H), (W - fw, H - fw), (fw, H - fw)],
        "left": [(0, 0), (fw, fw), (fw, H - fw), (0, H)],
        "right": [(W, 0), (W - fw, fw), (W - fw, H - fw), (W, H)],
    }
    shading = {"top": 1.06, "left": 1.02, "right": 0.90, "bottom": 0.84}
    for name, poly in pieces.items():
        if kind == "oak":
            if name in ("top", "bottom"):
                tex = _wood_texture(W, fw, rng)
                layer = np.zeros((H, W, 3), dtype=np.uint8)
                y = 0 if name == "top" else H - fw
                layer[y : y + fw, :, :] = tex
            else:
                tex = np.ascontiguousarray(_wood_texture(H, fw, rng).transpose(1, 0, 2))
                layer = np.zeros((H, W, 3), dtype=np.uint8)
                x = 0 if name == "left" else W - fw
                layer[:, x : x + fw, :] = tex
            arr = np.clip(layer.astype(np.float32) * shading[name], 0, 255).astype(np.uint8)
        else:
            tone = int(30 * shading[name])
            arr = np.full((H, W, 3), (tone, tone, tone + 1), dtype=np.uint8)
        mask = Image.new("L", outer, 0)
        ImageDraw.Draw(mask).polygon(poly, fill=255)
        frame.paste(Image.fromarray(arr, "RGB"), (0, 0), mask)
    d = ImageDraw.Draw(frame)
    edge = max(1, fw // 18)
    d.rectangle((fw - edge, fw - edge, W - fw + edge - 1, H - fw + edge - 1), outline=(0, 0, 0, 90), width=edge)
    d.rectangle((0, 0, W - 1, H - 1), outline=(255, 255, 255, 40), width=max(1, edge // 2))
    return frame


def _framed_mockup(
    art: Image.Image,
    wall_rgb: tuple[int, int, int],
    kind: str,
    mat_ratio: float,
    frame_ratio: float,
    seed: int,
) -> Image.Image:
    W, H = MOCKUP_SIZE
    wall = make_wall(MOCKUP_SIZE, wall_rgb, seed)
    aspect = art.width / art.height
    s = _fit_outer(aspect, mat_ratio, frame_ratio, W * 0.60, H * 0.76)
    short = s
    art_w = round(short * aspect) if aspect >= 1 else round(short)
    art_h = round(short) if aspect >= 1 else round(short / aspect)
    mat, fw = round(short * mat_ratio), max(8, round(short * frame_ratio))
    outer_w, outer_h = art_w + 2 * (mat + fw), art_h + 2 * (mat + fw)
    ox, oy = (W - outer_w) // 2, round(H * 0.46 - outer_h / 2)
    rect = (ox, oy, ox + outer_w, oy + outer_h)

    cast_shadow(wall, rect, (round(W * 0.010), round(H * 0.022)), blur=W * 0.016, strength=0.38)
    cast_shadow(wall, rect, (3, 6), blur=6, strength=0.30)

    if mat:
        mat_img = Image.new("RGB", (outer_w - 2 * fw, outer_h - 2 * fw), (246, 245, 241))
        md = ImageDraw.Draw(mat_img)
        bevel = max(2, mat // 40)
        ax0, ay0 = mat - bevel, mat - bevel
        md.rectangle((ax0, ay0, ax0 + art_w + 2 * bevel, ay0 + art_h + 2 * bevel), fill=(255, 255, 253))
        wall.paste(mat_img, (ox + fw, oy + fw))
    else:
        wall.paste(Image.new("RGB", (art_w, art_h), PAPER), (ox + fw, oy + fw))

    art_resized = art.resize((art_w, art_h), Image.Resampling.LANCZOS)
    ax, ay = ox + fw + mat, oy + fw + mat
    wall.paste(art_resized, (ax, ay))
    art_resized.close()

    # inner shadow cast by the frame/mat on the artwork (top-left light)
    inner = Image.new("L", (outer_w - 2 * fw, outer_h - 2 * fw), 0)
    ImageDraw.Draw(inner).rectangle((0, 0, inner.width, max(3, fw // 3)), fill=120)
    ImageDraw.Draw(inner).rectangle((0, 0, max(3, fw // 4), inner.height), fill=90)
    inner = inner.filter(ImageFilter.GaussianBlur(max(2, fw // 4)))
    region = wall.crop((ox + fw, oy + fw, ox + outer_w - fw, oy + outer_h - fw))
    darkened = Image.composite(Image.new("RGB", region.size, (40, 38, 35)), region, inner.point(lambda v: v * 0.35))
    wall.paste(darkened, (ox + fw, oy + fw))

    # soft glass reflection
    glare = Image.new("L", (outer_w - 2 * fw, outer_h - 2 * fw), 0)
    gd = ImageDraw.Draw(glare)
    gw, gh = glare.size
    gd.polygon([(int(gw * 0.55), 0), (int(gw * 0.80), 0), (int(gw * 0.35), gh), (int(gw * 0.10), gh)], fill=14)
    glare = glare.filter(ImageFilter.GaussianBlur(gw // 12 or 1))
    region = wall.crop((ox + fw, oy + fw, ox + outer_w - fw, oy + outer_h - fw))
    wall.paste(Image.composite(Image.new("RGB", region.size, (255, 255, 255)), region, glare), (ox + fw, oy + fw))

    frame = draw_mitred_frame((outer_w, outer_h), fw, kind, seed)
    wall.paste(frame, (ox, oy), frame)
    return wall


def mockup_black_frame(fullbleed_art: Image.Image, seed: int = 7) -> Image.Image:
    return _framed_mockup(fullbleed_art, (226, 224, 219), "black", mat_ratio=0.14, frame_ratio=0.035, seed=seed)


def mockup_oak_frame(poster_art: Image.Image, seed: int = 11) -> Image.Image:
    return _framed_mockup(poster_art, (236, 229, 216), "oak", mat_ratio=0.0, frame_ratio=0.055, seed=seed)


# ------------------------------------------------------------------ infographic
def mockup_sizes_infographic(
    crop_previews: dict[str, Image.Image], orientation: str, fonts_dir: str, dpi: int = 300
) -> Image.Image:
    W, H = MOCKUP_SIZE
    canvas = Image.new("RGB", MOCKUP_SIZE, PAPER)
    d = ImageDraw.Draw(canvas)

    title_font = fit_font(d, fonts_dir, "serif", "PRINT SIZES INCLUDED", 84, W * 0.8, tracking=0.25, weight=500)
    sub_font = load_font(fonts_dir, "sans", 34, 300)
    draw_tracked(d, W / 2, 90, "PRINT SIZES INCLUDED", title_font, INK, tracking=0.25)
    draw_tracked(d, W / 2, 90 + text_height(title_font) + 36, "FORMATI DI STAMPA INCLUSI", sub_font, INK_SOFT, tracking=0.3)

    # proportional silhouettes, bottom aligned
    zone_top, zone_bottom = 330, 1060
    gap = 70
    dims: list[tuple[PrintFormat, float, float]] = []
    for fmt in FORMATS:
        sw, lg = fmt.largest_cm
        w_cm, h_cm = (sw, lg) if orientation == "portrait" else (lg, sw)
        dims.append((fmt, w_cm, h_cm))
    total_w_cm = sum(w for _, w, _ in dims)
    max_h_cm = max(h for _, _, h in dims)
    avail_w = W * 0.86 - gap * (len(dims) - 1)
    scale = min(avail_w / total_w_cm, (zone_bottom - zone_top) / max_h_cm)
    row_w = total_w_cm * scale + gap * (len(dims) - 1)
    x = (W - row_w) / 2
    # centre the silhouette row vertically when the zone is not fully used (landscape)
    zone_bottom -= round(((zone_bottom - zone_top) - max_h_cm * scale) / 2)
    label_font = load_font(fonts_dir, "sans", 38, 500)
    small_font = load_font(fonts_dir, "sans", 26, 300)
    for fmt, w_cm, h_cm in dims:
        rw, rh = round(w_cm * scale), round(h_cm * scale)
        rx, ry = round(x), zone_bottom - rh
        prev = crop_previews.get(fmt.key)
        if prev is not None:
            thumb = prev.resize((rw, rh), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (rx, ry))
            thumb.close()
        else:
            d.rectangle((rx, ry, rx + rw, ry + rh), fill="#ECE9E3")
        d.rectangle((rx, ry, rx + rw - 1, ry + rh - 1), outline=INK, width=3)
        cx = rx + rw / 2
        draw_tracked(d, cx, zone_bottom + 28, fmt.ratio_label, label_font, INK, tracking=0.08)
        max_label = f"max {w_cm:g}×{h_cm:g} cm"
        draw_tracked(d, cx, zone_bottom + 80, max_label, small_font, INK_SOFT, tracking=0.02)
        x += rw + gap

    # bilingual table
    table_top = 1230
    col_x = (W * 0.08, W * 0.26, W * 0.66)
    head_font = load_font(fonts_dir, "sans", 28, 600)
    cell_font = load_font(fonts_dir, "sans", 30, 400)
    for cx_, txt in zip(col_x, ("RATIO / FORMATO", "INCHES / POLLICI", "CM"), strict=True):
        d.text((cx_, table_top), txt, font=head_font, fill=INK_SOFT)
    d.line((W * 0.08, table_top + 50, W * 0.92, table_top + 50), fill="#D9D5CE", width=2)
    row_h = 62
    for i, fmt in enumerate(FORMATS):
        y = table_top + 72 + i * row_h
        unit_in = "" if fmt.key == "iso_A_series" else " in"
        d.text((col_x[0], y), fmt.ratio_label, font=cell_font, fill=INK)
        d.text((col_x[1], y), " · ".join(s.replace("x", "×") for s in fmt.sizes_in) + unit_in, font=cell_font, fill=INK)
        d.text((col_x[2], y), " · ".join(s.replace("x", "×") for s in fmt.sizes_cm) + " cm", font=cell_font, fill=INK)

    foot_font = load_font(fonts_dir, "sans", 26, 300)
    footer_en = f"{dpi} DPI · High-resolution JPEG · sRGB · Instant digital download"
    footer_it = f"{dpi} DPI · JPEG ad alta risoluzione · sRGB · Download digitale immediato"
    note = "2:3 native 300 DPI up to 16×24 in; larger sizes ≥ 200 DPI  ·  2:3 nativo 300 DPI fino a 40×60 cm; formati maggiori ≥ 200 DPI"
    draw_tracked(d, W / 2, H - 150, footer_en, foot_font, INK_SOFT, tracking=0.04)
    draw_tracked(d, W / 2, H - 108, footer_it, foot_font, INK_SOFT, tracking=0.04)
    note_font = fit_font(d, fonts_dir, "sans", note, 22, W * 0.9, tracking=0.0, weight=300)
    draw_tracked(d, W / 2, H - 62, note, note_font, INK_SOFT, tracking=0.0)
    return canvas


def save_mockup(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=90, optimize=True, subsampling=0, icc_profile=srgb_icc_bytes(), dpi=(72, 72))


def generate_mockups(
    fullbleed_preview: Image.Image,
    poster_preview: Image.Image,
    crop_previews: dict[str, Image.Image],
    orientation: str,
    out_dir: Path,
    fonts_dir: str,
    dpi: int,
) -> list[Path]:
    jobs = (
        ("mockup_01_fullbleed_black_frame.jpg", lambda: mockup_black_frame(fullbleed_preview)),
        ("mockup_02_gallery_poster_oak_frame.jpg", lambda: mockup_oak_frame(poster_preview)),
        ("mockup_03_sizes_infographic.jpg", lambda: mockup_sizes_infographic(crop_previews, orientation, fonts_dir, dpi)),
    )
    paths: list[Path] = []
    for name, render in jobs:
        img = render()
        path = out_dir / name
        save_mockup(img, path)
        img.close()
        paths.append(path)
        log.info("mockup saved", extra={"file": name})
    return paths
