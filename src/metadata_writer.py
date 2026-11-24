"""Microstock JPEG export and IPTC/XMP metadata embedding via ExifTool."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .image_loader import srgb_icc_bytes

log = logging.getLogger(__name__)

IPTC_OBJECTNAME_MAX = 64
IPTC_HEADLINE_MAX = 256
IPTC_CAPTION_MAX = 2000
IPTC_KEYWORD_MAX = 64

# Camera tags copied from the original (Orientation deliberately excluded: pixels are already upright).
_COPY_TAGS = (
    "EXIF:Make", "EXIF:Model", "EXIF:LensModel", "EXIF:FocalLength", "EXIF:FNumber",
    "EXIF:ExposureTime", "EXIF:ISO", "EXIF:DateTimeOriginal", "EXIF:CreateDate",
)
_GPS_TAGS = ("EXIF:GPSLatitude", "EXIF:GPSLatitudeRef", "EXIF:GPSLongitude", "EXIF:GPSLongitudeRef")


class MetadataError(Exception):
    pass


def export_microstock_jpeg(img: Image.Image, path: Path, quality: int, min_megapixels: float) -> Path:
    """sRGB JPEG q95; images below the agencies' minimum size are enlarged with Lanczos."""
    out = img
    mp = img.width * img.height / 1e6
    if mp < min_megapixels:
        factor = (min_megapixels / mp) ** 0.5 * 1.01
        out = img.resize((round(img.width * factor), round(img.height * factor)), Image.Resampling.LANCZOS)
        log.info("microstock image enlarged to minimum size", extra={"from_mp": round(mp, 2)})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, "JPEG", quality=quality, subsampling=0, optimize=True, icc_profile=srgb_icc_bytes())
    if out is not img:
        out.close()
    return path


def _clip(text: str, limit: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    return data[:limit].decode("utf-8", errors="ignore").rsplit(" ", 1)[0]


def build_exiftool_args(
    title: str,
    description: str,
    keywords: list[str],
    *,
    original: Path | None,
    keep_gps: bool,
    author: str = "",
    copyright_notice: str = "",
) -> list[str]:
    args = ["-overwrite_original", "-m", "-charset", "iptc=UTF8", "-codedcharacterset=utf8"]
    if original is not None and original.exists():
        args += ["-TagsFromFile", str(original)]
        args += [f"-{t}" for t in _COPY_TAGS]
        if keep_gps:
            args += [f"-{t}" for t in _GPS_TAGS]
    args += [
        "-XMP-dc:Title=" + title,
        "-XMP-dc:Description=" + description,
        "-XMP-photoshop:Headline=" + _clip(title, IPTC_HEADLINE_MAX),
        "-IPTC:ObjectName=" + _clip(title, IPTC_OBJECTNAME_MAX),
        "-IPTC:Headline=" + _clip(title, IPTC_HEADLINE_MAX),
        "-IPTC:Caption-Abstract=" + _clip(description, IPTC_CAPTION_MAX),
        "-EXIF:ImageDescription=" + _clip(description, IPTC_CAPTION_MAX),
        "-XMP-dc:Subject=",
        "-IPTC:Keywords=",
    ]
    for kw in keywords:
        k = _clip(kw, IPTC_KEYWORD_MAX)
        args += ["-XMP-dc:Subject+=" + k, "-IPTC:Keywords+=" + k]
    if author:
        args += ["-XMP-dc:Creator=" + author, "-IPTC:By-line=" + _clip(author, 32), "-EXIF:Artist=" + author]
    if copyright_notice:
        args += [
            "-XMP-dc:Rights=" + copyright_notice,
            "-IPTC:CopyrightNotice=" + _clip(copyright_notice, 128),
            "-EXIF:Copyright=" + copyright_notice,
        ]
    return args


def write_metadata(
    jpeg_path: Path,
    title: str,
    description: str,
    keywords: list[str],
    *,
    original: Path | None = None,
    keep_gps: bool = False,
    author: str = "",
    copyright_notice: str = "",
) -> None:
    exe = shutil.which("exiftool")
    if not exe:
        raise MetadataError("exiftool not found in PATH (install libimage-exiftool-perl)")
    args = build_exiftool_args(
        title, description, keywords, original=original, keep_gps=keep_gps,
        author=author, copyright_notice=copyright_notice,
    )
    # An argfile avoids shell quoting and command-line length issues; one argument per line.
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(a.replace("\n", " ") for a in args) + "\n")
        argfile = Path(fh.name)
    try:
        proc = subprocess.run(
            [exe, "-charset", "filename=utf8", "-@", str(argfile), str(jpeg_path)],
            capture_output=True, timeout=120, check=False,
        )
    finally:
        argfile.unlink(missing_ok=True)
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0 or "1 image files updated" not in stdout:
        raise MetadataError(f"exiftool failed (rc={proc.returncode}): {stderr.strip() or stdout.strip()}")
    if stderr.strip():
        log.debug("exiftool warnings", extra={"stderr": stderr.strip()[:500]})
    log.info("IPTC/XMP metadata written", extra={"file": jpeg_path.name, "keywords": len(keywords)})


def read_back_keywords(jpeg_path: Path) -> list[str]:
    """Verification helper (used by tests/diagnostics)."""
    exe = shutil.which("exiftool")
    if not exe:
        return []
    proc = subprocess.run(
        [exe, "-s3", "-sep", "||", "-XMP-dc:Subject", str(jpeg_path)], capture_output=True, timeout=60, check=False
    )
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    return [k for k in out.split("||") if k]
