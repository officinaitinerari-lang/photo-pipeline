"""Saliency-driven smart crop: find the visual centre of mass and crop around it."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_ANALYSIS_LONG_SIDE = 512


@dataclass(frozen=True)
class Focus:
    """Normalised (0..1) focus point plus the salient bounding box to keep in frame."""

    cx: float
    cy: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 normalised
    method: str


def _to_small_gray(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    small = img.convert("RGB").copy()
    small.thumbnail((_ANALYSIS_LONG_SIDE, _ANALYSIS_LONG_SIDE), Image.Resampling.BILINEAR)
    rgb = np.asarray(small)
    small.close()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def _spectral_residual(bgr: np.ndarray) -> np.ndarray | None:
    saliency_mod = getattr(cv2, "saliency", None)
    if saliency_mod is None:
        return None
    try:
        detector = saliency_mod.StaticSaliencySpectralResidual_create()
        ok, sal = detector.computeSaliency(bgr)
    except cv2.error:
        return None
    if not ok or sal is None:
        return None
    sal = sal.astype(np.float32)
    return sal if float(sal.max()) > 0 else None


def _edge_density(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edges = cv2.Canny(gray, 60, 160).astype(np.float32)
    combined = mag / (float(mag.max()) + 1e-6) + edges / 255.0
    k = max(3, (min(gray.shape) // 12) | 1)
    return cv2.GaussianBlur(combined, (k, k), 0)


def _faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return []
        found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24))
        return [tuple(int(v) for v in f) for f in found]  # type: ignore[misc]
    except (AttributeError, cv2.error):
        return []


def find_focus(img: Image.Image) -> Focus:
    bgr, gray = _to_small_gray(img)
    h, w = gray.shape
    sal = _spectral_residual(bgr)
    method = "spectral_residual"
    if sal is None:
        sal = _edge_density(gray)
        method = "edge_density"
    sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=max(1.0, w / 100))
    sal = sal / (float(sal.max()) + 1e-6)

    faces = _faces(gray)
    for x, y, fw, fh in faces:
        sal[y : y + fh, x : x + fw] += 2.0  # faces dominate the composition
    if faces:
        method += "+faces"

    # Keep only the most salient part to avoid the background dragging the centre to the middle.
    thr = float(np.percentile(sal, 85))
    weights = np.where(sal >= thr, sal, 0.0).astype(np.float64)
    total = float(weights.sum())
    if total <= 0:
        return Focus(0.5, 0.5, (0.25, 0.25, 0.75, 0.75), "center")
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((weights * xs).sum() / total) / w
    cy = float((weights * ys).sum() / total) / h

    mask = weights > 0
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    box = (cols[0] / w, rows[0] / h, (cols[-1] + 1) / w, (rows[-1] + 1) / h)
    return Focus(cx, cy, box, method)


def crop_box_for_ratio(
    width: int, height: int, ratio: float, focus: Focus | None = None
) -> tuple[int, int, int, int]:
    """Largest crop with aspect `ratio` (= w/h) positioned around the focus, clamped to the image.

    When the salient box fits inside the crop window the window is shifted so the box is
    fully contained, otherwise the window is centred on the focus point.
    """
    if width <= 0 or height <= 0 or ratio <= 0:
        raise ValueError("invalid dimensions or ratio")
    if width / height > ratio:
        ch, cw = height, max(1, round(height * ratio))
    else:
        cw, ch = width, max(1, round(width / ratio))
    cw, ch = min(cw, width), min(ch, height)
    f = focus or Focus(0.5, 0.5, (0.5, 0.5, 0.5, 0.5), "center")

    def place(center: float, size: int, total: int, b0: float, b1: float) -> int:
        start = round(center * total - size / 2)
        bx0, bx1 = b0 * total, b1 * total
        if bx1 - bx0 <= size:  # salient box fits: ensure it is fully inside
            start = min(start, int(bx0))
            start = max(start, int(np.ceil(bx1)) - size)
        return int(min(max(0, start), total - size))

    left = place(f.cx, cw, width, f.box[0], f.box[2])
    top = place(f.cy, ch, height, f.box[1], f.box[3])
    return left, top, left + cw, top + ch


def smart_crop(img: Image.Image, ratio: float, focus: Focus | None = None) -> Image.Image:
    box = crop_box_for_ratio(img.width, img.height, ratio, focus or find_focus(img))
    return img.crop(box)
