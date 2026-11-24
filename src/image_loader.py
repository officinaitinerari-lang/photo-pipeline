"""Multi-format loading (JPEG/TIFF/PNG/RAW), colour management, orientation, EXIF and path context."""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .config import RAW_EXTENSIONS, SUPPORTED_EXTENSIONS

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 400_000_000  # large scans/panoramas are legitimate input

_CAMERA_STEM = re.compile(
    r"^(_?ds[cf][fn]?|img|pxl|dji|gopr|p\d{3}|_mg|r\d{3}|scan|cimg|sam|mvimg|photo)[\s_\-]*\d+([\s_\-]\d+)*$",
    re.IGNORECASE,
)
_SEPARATORS = re.compile(r"[_\-.\s]+")
_TOKEN_SPLIT = re.compile(r"[^0-9a-zA-Z]+")


class ImageLoadError(Exception):
    """The file could not be decoded as an image."""


@dataclass
class ExifInfo:
    latitude: float | None = None
    longitude: float | None = None
    datetime_original: str | None = None
    make: str | None = None
    model: str | None = None
    lens: str | None = None
    focal_length_mm: float | None = None
    f_number: float | None = None
    iso: int | None = None
    exposure_time: float | None = None

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def camera(self) -> str | None:
        parts = [p for p in (self.make, self.model) if p]
        if len(parts) == 2 and parts[1].lower().startswith(parts[0].lower()):
            return parts[1]
        return " ".join(parts) or None

    def as_prompt_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.has_gps:
            d["gps"] = f"{self.latitude:.5f}, {self.longitude:.5f}"
        for key, value in (
            ("date", self.datetime_original),
            ("camera", self.camera),
            ("lens", self.lens),
            ("focal_length_mm", self.focal_length_mm),
            ("aperture", f"f/{self.f_number:g}" if self.f_number else None),
            ("iso", self.iso),
        ):
            if value:
                d[key] = value
        return d


@dataclass
class LoadedImage:
    image: Image.Image  # 8-bit sRGB "RGB"
    source_path: Path
    orientation: str  # "landscape" | "portrait"
    is_analog_film: bool
    context_text: str
    exif: ExifInfo = field(default_factory=ExifInfo)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def close(self) -> None:
        self.image.close()


# ----------------------------------------------------------------------------- context
def _humanise(text: str) -> str:
    return _SEPARATORS.sub(" ", text).strip()


def build_context(path: Path, input_root: Path) -> str:
    """'/in/Giappone_Nagano_Inverno/tempio_neve.jpg' -> 'Giappone Nagano Inverno - tempio neve'."""
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = Path(path.name)
    folders = [_humanise(p) for p in rel.parts[:-1]]
    folders = [f for f in folders if f]
    stem = path.stem
    stem_text = "" if _CAMERA_STEM.match(stem.strip()) else _humanise(stem)
    folder_text = " / ".join(folders)
    if folder_text and stem_text:
        return f"{folder_text} - {stem_text}"
    return folder_text or stem_text


def detect_analog(path: Path, input_root: Path, keywords: list[str]) -> bool:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = Path(path.name)
    tokens: list[str] = []
    for part in (*rel.parts[:-1], rel.stem):
        tokens.extend(t.lower() for t in _TOKEN_SPLIT.split(part) if t)
    return any(kw in tok for tok in tokens for kw in keywords)


def orientation_of(width: int, height: int) -> str:
    return "portrait" if height > width else "landscape"


# ----------------------------------------------------------------------------- exif
def _float(v: Any) -> float | None:
    try:
        if isinstance(v, tuple) and len(v) == 2:
            return float(v[0]) / float(v[1]) if v[1] else None
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _read_exif_exiftool(path: Path) -> ExifInfo | None:
    exe = shutil.which("exiftool")
    if not exe:
        return None
    cmd = [
        exe, "-j", "-n", "-charset", "filename=utf8",
        "-GPSLatitude", "-GPSLatitudeRef", "-GPSLongitude", "-GPSLongitudeRef",
        "-DateTimeOriginal", "-CreateDate", "-Make", "-Model", "-LensModel", "-Lens",
        "-FocalLength", "-FNumber", "-ISO", "-ExposureTime", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
        data = json.loads(out.stdout.decode("utf-8") or "[]")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        log.warning("exiftool read failed", extra={"path": str(path), "error": str(exc)})
        return None
    if not data:
        return None
    d = data[0]
    lat, lon = _float(d.get("GPSLatitude")), _float(d.get("GPSLongitude"))
    if lat is not None and str(d.get("GPSLatitudeRef", "")).upper().startswith("S") and lat > 0:
        lat = -lat
    if lon is not None and str(d.get("GPSLongitudeRef", "")).upper().startswith("W") and lon > 0:
        lon = -lon
    if lat == 0 and lon == 0:
        lat = lon = None
    iso = _float(d.get("ISO"))
    return ExifInfo(
        latitude=lat,
        longitude=lon,
        datetime_original=str(d.get("DateTimeOriginal") or d.get("CreateDate") or "") or None,
        make=(str(d["Make"]).strip() if d.get("Make") else None),
        model=(str(d["Model"]).strip() if d.get("Model") else None),
        lens=(str(d.get("LensModel") or d.get("Lens") or "").strip() or None),
        focal_length_mm=_float(d.get("FocalLength")),
        f_number=_float(d.get("FNumber")),
        iso=int(iso) if iso else None,
        exposure_time=_float(d.get("ExposureTime")),
    )


def _dms_to_deg(dms: Any, ref: Any) -> float | None:
    try:
        deg = _float(dms[0]) or 0.0
        minutes = _float(dms[1]) or 0.0
        seconds = _float(dms[2]) or 0.0
    except (TypeError, IndexError):
        return None
    value = deg + minutes / 60.0 + seconds / 3600.0
    if str(ref).upper().startswith(("S", "W")):
        value = -value
    return value


def _read_exif_pillow(img: Image.Image) -> ExifInfo:
    info = ExifInfo()
    try:
        exif = img.getexif()
    except Exception:
        return info
    info.make = (str(exif.get(271)).strip() or None) if exif.get(271) else None
    info.model = (str(exif.get(272)).strip() or None) if exif.get(272) else None
    try:
        sub = exif.get_ifd(0x8769)
        info.datetime_original = str(sub.get(36867)) if sub.get(36867) else None
        info.focal_length_mm = _float(sub.get(37386))
        info.f_number = _float(sub.get(33437))
        iso = sub.get(34855)
        info.iso = int(iso[0] if isinstance(iso, tuple) else iso) if iso else None
        info.exposure_time = _float(sub.get(33434))
        info.lens = str(sub.get(42036)).strip() if sub.get(42036) else None
        gps = exif.get_ifd(0x8825)
        if gps and 2 in gps and 4 in gps:
            info.latitude = _dms_to_deg(gps[2], gps.get(1, "N"))
            info.longitude = _dms_to_deg(gps[4], gps.get(3, "E"))
            if info.latitude == 0 and info.longitude == 0:
                info.latitude = info.longitude = None
    except Exception:
        log.debug("partial EXIF parse failure", exc_info=True)
    return info


# ----------------------------------------------------------------------------- decoding
_SRGB_PROFILE = ImageCms.createProfile("sRGB")


def srgb_icc_bytes() -> bytes:
    return ImageCms.ImageCmsProfile(_SRGB_PROFILE).tobytes()


def _to_srgb_rgb8(img: Image.Image) -> Image.Image:
    """Normalise any Pillow image to 8-bit sRGB RGB, honouring embedded ICC profiles."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.uint32)
        maxv = 65535 if arr.max() > 255 else 255
        arr8 = (arr.astype(np.float32) * (255.0 / maxv)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr8, "L")
    elif img.mode == "F":
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
        img = Image.fromarray((arr * 255).astype(np.uint8), "L")

    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        icc = img.info.get("icc_profile")
        img = bg
        if icc:
            img.info["icc_profile"] = icc

    icc = img.info.get("icc_profile")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if img.mode not in ("RGB", "CMYK", "L"):
                img = img.convert("RGB")
            converted = ImageCms.profileToProfile(
                img, src, _SRGB_PROFILE, renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB"
            )
            if converted is not None:
                return converted
        except (ImageCms.PyCMSError, OSError, ValueError) as exc:
            log.warning("ICC conversion failed, assuming sRGB", extra={"error": str(exc)})
    return img.convert("RGB") if img.mode != "RGB" else img


def _decode_raw(path: Path) -> Image.Image:
    import rawpy  # local import: heavy native module

    try:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=8,
                no_auto_bright=False,
                auto_bright_thr=0.001,
            )
    except (rawpy.LibRawError, OSError) as exc:
        raise ImageLoadError(f"RAW decode failed: {exc}") from exc
    img = Image.fromarray(rgb, "RGB")
    del rgb
    return img


def _decode_with_opencv(path: Path) -> Image.Image:
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if arr is None:
        raise ImageLoadError("OpenCV could not decode the file")
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr, "RGB")


def _decode_raster(path: Path) -> tuple[Image.Image, ExifInfo]:
    try:
        with Image.open(path) as src:
            src.load()
            exif_info = _read_exif_pillow(src)
            oriented = ImageOps.exif_transpose(src)
            if oriented is None:
                oriented = src.copy()
            if "icc_profile" in src.info and "icc_profile" not in oriented.info:
                oriented.info["icc_profile"] = src.info["icc_profile"]
            img = _to_srgb_rgb8(oriented)
        return img, exif_info
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        if path.suffix.lower() in (".tif", ".tiff"):
            log.info("Pillow cannot decode TIFF, falling back to OpenCV", extra={"error": str(exc)})
            # IMREAD_UNCHANGED ignores EXIF orientation: apply it explicitly.
            return _apply_orientation(_decode_with_opencv(path), _read_orientation(path)), ExifInfo()
        raise ImageLoadError(f"Cannot decode image: {exc}") from exc


def _read_orientation(path: Path) -> int:
    exe = shutil.which("exiftool")
    if not exe:
        return 1
    try:
        out = subprocess.run([exe, "-s3", "-n", "-Orientation", str(path)], capture_output=True, timeout=30, check=False)
        return int(out.stdout.decode().strip() or 1)
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return 1


def _apply_orientation(img: Image.Image, orientation: int) -> Image.Image:
    ops = {
        2: [Image.Transpose.FLIP_LEFT_RIGHT],
        3: [Image.Transpose.ROTATE_180],
        4: [Image.Transpose.FLIP_TOP_BOTTOM],
        5: [Image.Transpose.TRANSPOSE],
        6: [Image.Transpose.ROTATE_270],
        7: [Image.Transpose.TRANSVERSE],
        8: [Image.Transpose.ROTATE_90],
    }.get(orientation, [])
    for op in ops:
        img = img.transpose(op)
    return img


def load_image(path: Path, input_root: Path, analog_keywords: list[str]) -> LoadedImage:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"Unsupported extension: {ext}")
    if ext in RAW_EXTENSIONS:
        img, pil_exif = _decode_raw(path), ExifInfo()
    else:
        img, pil_exif = _decode_raster(path)

    exif = _read_exif_exiftool(path) or pil_exif
    if not exif.has_gps and pil_exif.has_gps:
        exif.latitude, exif.longitude = pil_exif.latitude, pil_exif.longitude

    w, h = img.size
    loaded = LoadedImage(
        image=img,
        source_path=path,
        orientation=orientation_of(w, h),
        is_analog_film=detect_analog(path, input_root, analog_keywords),
        context_text=build_context(path, input_root),
        exif=exif,
    )
    log.info(
        "image loaded",
        extra={
            "path": str(path), "width": w, "height": h, "orientation": loaded.orientation,
            "analog": loaded.is_analog_film, "context": loaded.context_text, "gps": exif.has_gps,
        },
    )
    return loaded


def make_preview_jpeg(img: Image.Image, long_side: int = 1024, quality: int = 80) -> bytes:
    """Lightweight in-memory JPEG used for the Vision API call."""
    preview = img.copy()
    preview.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    try:
        preview.save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    finally:
        preview.close()
        buf.close()
