import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.smart_crop import Focus, crop_box_for_ratio, find_focus, smart_crop


def _scene(w=1500, h=1000, subject_box=(1150, 380, 1350, 620)):
    rng = np.random.default_rng(0)
    base = (np.full((h, w, 3), 120, np.float32) + rng.normal(0, 3, (h, w, 3))).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base, "RGB")
    d = ImageDraw.Draw(img)
    d.rectangle(subject_box, fill=(250, 40, 30))
    d.ellipse((subject_box[0] + 40, subject_box[1] + 40, subject_box[2] - 40, subject_box[3] - 40), fill=(10, 10, 200))
    return img


@pytest.mark.parametrize("ratio", [2 / 3, 3 / 4, 4 / 5, 1 / 1.414, 11 / 14, 1.5])
def test_crop_box_has_exact_ratio_and_fits(ratio):
    left, top, right, bottom = crop_box_for_ratio(1500, 1000, ratio)
    w, h = right - left, bottom - top
    assert left >= 0 and top >= 0 and right <= 1500 and bottom <= 1000
    assert abs(w / h - ratio) < 0.01
    assert w == 1500 or h == 1000  # largest possible crop


def test_crop_is_clamped_at_border():
    focus = Focus(0.99, 0.5, (0.98, 0.4, 1.0, 0.6), "test")
    left, _, right, _ = crop_box_for_ratio(1500, 1000, 2 / 3, focus)
    assert right == 1500 and left == 1500 - round(1000 * 2 / 3)


def test_salient_box_kept_inside_crop():
    focus = Focus(0.5, 0.5, (0.70, 0.3, 0.92, 0.7), "test")
    left, _, right, _ = crop_box_for_ratio(1500, 1000, 2 / 3, focus)
    assert left <= 0.70 * 1500 and right >= 0.92 * 1500


def test_focus_finds_off_centre_subject():
    img = _scene()
    focus = find_focus(img)
    assert focus.cx > 0.6, focus
    assert 0.25 < focus.cy < 0.75


def test_portrait_crop_contains_subject():
    subject = (1150, 380, 1350, 620)
    img = _scene(subject_box=subject)
    focus = find_focus(img)
    left, top, right, bottom = crop_box_for_ratio(img.width, img.height, 2 / 3, focus)
    assert left <= subject[0] and right >= subject[2]
    assert top <= subject[1] and bottom >= subject[3]


def test_smart_crop_returns_requested_ratio():
    out = smart_crop(_scene(), 4 / 5)
    assert abs(out.width / out.height - 0.8) < 0.01


def test_edge_density_fallback(monkeypatch):
    import src.smart_crop as sc

    monkeypatch.setattr(sc, "_spectral_residual", lambda _bgr: None)
    focus = sc.find_focus(_scene())
    assert focus.method.startswith("edge_density")
    assert focus.cx > 0.6
