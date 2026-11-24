"""CPU-safe conditional upscaling: tiled ONNX Runtime (Real-ESRGAN compact) or Lanczos fallback.

Film scans never go through the AI model (it would smear the grain): they use a
conservative multi-step Lanczos resize with gentle micro-contrast.
"""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger(__name__)


def needs_upscale(width: int, height: int, threshold: int) -> bool:
    return max(width, height) < threshold


def target_size(width: int, height: int, target_long_side: int) -> tuple[int, int]:
    scale = target_long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def lanczos_stepwise(img: Image.Image, size: tuple[int, int], max_step: float = 1.5) -> Image.Image:
    """Upscale in steps of at most `max_step`x: fewer ringing artefacts than one big jump."""
    current = img
    while True:
        cw, ch = current.size
        if cw >= size[0] and ch >= size[1]:
            break
        factor = min(max_step, size[0] / cw, size[1] / ch)
        if factor <= 1.0001:
            break
        nxt = current.resize((math.ceil(cw * factor), math.ceil(ch * factor)), Image.Resampling.LANCZOS)
        if current is not img:
            current.close()
        current = nxt
    if current.size != size:
        final = current.resize(size, Image.Resampling.LANCZOS)
        if current is not img:
            current.close()
        current = final
    return current if current is not img else img.copy()


def upscale_lanczos_digital(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = lanczos_stepwise(img, size)
    sharpened = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))
    out.close()
    return sharpened


def upscale_film(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Grain-preserving resize: no denoise, only Lanczos steps + mild large-radius local contrast."""
    out = lanczos_stepwise(img, size, max_step=1.25)
    local_contrast = out.filter(ImageFilter.UnsharpMask(radius=max(4, round(max(size) / 900)), percent=12, threshold=0))
    out.close()
    return local_contrast


class OnnxUpscaler:
    """Tiled inference so peak RAM stays bounded (<~1 GB for 512px tiles on the compact model)."""

    def __init__(self, model_path: Path, tile_size: int = 512, overlap: int = 24, threads: int = 2) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = False  # release memory between tiles
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # [N, C, H, W], dims may be symbolic
        fixed_h = shape[2] if isinstance(shape[2], int) else None
        fixed_w = shape[3] if isinstance(shape[3], int) else None
        self.fixed_tile = (fixed_h, fixed_w) if fixed_h and fixed_w else None
        self.tile = fixed_h if self.fixed_tile else tile_size
        self.overlap = overlap
        self.scale: int | None = None
        log.info("ONNX upscaler ready", extra={"model": model_path.name, "tile": self.tile, "threads": threads})

    def _infer(self, tile_rgb: np.ndarray) -> np.ndarray:
        x = tile_rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        y = self.session.run(None, {self.input_name: x})[0][0]
        out = (y.transpose(1, 2, 0).clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        del x, y
        return out

    def upscale(self, img: Image.Image, size: tuple[int, int]) -> Image.Image:
        """Run the model tile by tile, resampling each tile straight to the final target scale.

        The output canvas is allocated only at the final target size, never at model scale,
        so a 4x model on a 4000px input does not allocate a 16000px intermediate image.
        """
        src = np.asarray(img.convert("RGB"))
        h, w = src.shape[:2]
        fx, fy = size[0] / w, size[1] / h
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        step = self.tile - 2 * self.overlap
        if step <= 0:
            raise ValueError("tile size must exceed twice the overlap")

        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                # core region (no overlap) in source coordinates
                cx1, cy1 = min(x0 + step, w), min(y0 + step, h)
                # padded region giving the network context on each side
                px0, py0 = max(0, x0 - self.overlap), max(0, y0 - self.overlap)
                px1, py1 = min(w, cx1 + self.overlap), min(h, cy1 + self.overlap)
                patch = src[py0:py1, px0:px1]
                if self.fixed_tile:
                    th, tw = self.fixed_tile
                    pad = np.zeros((th, tw, 3), dtype=np.uint8)
                    pad[: patch.shape[0], : patch.shape[1]] = patch
                    out = self._infer(pad)
                else:
                    out = self._infer(patch)
                if self.scale is None:
                    self.scale = round(out.shape[0] / (pad.shape[0] if self.fixed_tile else patch.shape[0]))
                s = self.scale
                core = out[(y0 - py0) * s : (cy1 - py0) * s, (x0 - px0) * s : (cx1 - px0) * s]
                # map core to target canvas coordinates
                tx0, ty0 = round(x0 * fx), round(y0 * fy)
                tx1, ty1 = round(cx1 * fx), round(cy1 * fy)
                if tx1 > tx0 and ty1 > ty0:
                    resized = Image.fromarray(core).resize((tx1 - tx0, ty1 - ty0), Image.Resampling.LANCZOS)
                    canvas[ty0:ty1, tx0:tx1] = np.asarray(resized)
                    resized.close()
                del out, core, patch
        del src
        gc.collect()
        return Image.fromarray(canvas, "RGB")


_onnx_cache: dict[str, OnnxUpscaler] = {}


def upscale_if_needed(
    img: Image.Image,
    *,
    is_analog: bool,
    threshold: int,
    target_long_side: int,
    engine: str,
    model_path: Path,
    tile_size: int,
    overlap: int,
    threads: int,
) -> tuple[Image.Image, str]:
    """Return (image, method). The returned image is a new object unless method == 'none'."""
    w, h = img.size
    if not needs_upscale(w, h, threshold):
        return img, "none"
    size = target_size(w, h, max(target_long_side, threshold))
    if is_analog:
        log.info("upscaling film scan with grain-preserving Lanczos", extra={"from": f"{w}x{h}", "to": f"{size[0]}x{size[1]}"})
        return upscale_film(img, size), "lanczos_film"
    if engine == "onnx":
        if model_path.is_file():
            try:
                key = str(model_path)
                if key not in _onnx_cache:
                    _onnx_cache[key] = OnnxUpscaler(model_path, tile_size, overlap, threads)
                log.info("upscaling with ONNX", extra={"from": f"{w}x{h}", "to": f"{size[0]}x{size[1]}"})
                return _onnx_cache[key].upscale(img, size), "onnx"
            except Exception:
                log.exception("ONNX upscaling failed, falling back to Lanczos")
                gc.collect()
        else:
            log.warning("ONNX model not found, using Lanczos fallback", extra={"model": str(model_path)})
    return upscale_lanczos_digital(img, size), "lanczos"
