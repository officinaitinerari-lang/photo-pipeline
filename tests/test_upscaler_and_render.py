import numpy as np
import pytest
from PIL import Image

from src.upscaler import OnnxUpscaler, needs_upscale, target_size, upscale_film, upscale_if_needed
from src.watcher import StabilityTracker, is_candidate


def test_threshold():
    assert needs_upscale(4000, 3000, 6000)
    assert not needs_upscale(6000, 4000, 6000)
    assert target_size(4000, 3000, 7200) == (7200, 5400)


def test_film_upscale_size_and_no_model(tmp_path):
    img = Image.new("RGB", (400, 300), (120, 110, 100))
    out, method = upscale_if_needed(
        img, is_analog=True, threshold=1000, target_long_side=1200, engine="onnx",
        model_path=tmp_path / "missing.onnx", tile_size=128, overlap=8, threads=1,
    )
    assert method == "lanczos_film" and out.size == (1200, 900)
    assert upscale_film(img, (800, 600)).size == (800, 600)


def test_missing_model_falls_back_to_lanczos(tmp_path):
    img = Image.new("RGB", (300, 200))
    out, method = upscale_if_needed(
        img, is_analog=False, threshold=1000, target_long_side=1200, engine="onnx",
        model_path=tmp_path / "missing.onnx", tile_size=128, overlap=8, threads=1,
    )
    assert method == "lanczos" and out.size == (1200, 800)


def _nearest_x4_model(path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "h4", "w4"])
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 4.0, 4.0])
    node = helper.make_node("Resize", ["input", "", "scales"], ["output"], mode="nearest")
    graph = helper.make_graph([node], "x4", [inp], [out], initializer=[scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_onnx_tiled_upscale_is_seamless(tmp_path):
    model_path = tmp_path / "x4.onnx"
    _nearest_x4_model(model_path)
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, (150, 230, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    up = OnnxUpscaler(model_path, tile_size=64, overlap=8, threads=1)
    out = up.upscale(img, (920, 600))  # exactly x4 -> must equal nearest-neighbour x4
    assert out.size == (920, 600)
    expected = np.repeat(np.repeat(arr, 4, axis=0), 4, axis=1)
    diff = np.abs(np.asarray(out).astype(int) - expected.astype(int))
    assert diff.max() <= 1  # no seams or offsets between tiles


def test_stability_tracker(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x" * 10)
    tr = StabilityTracker(stable_seconds=5)
    tr.touch(f)
    assert tr.poll(now=0) == []  # first observation
    assert tr.poll(now=3) == []
    with f.open("ab") as fh:
        fh.write(b"more")  # still being copied
    assert tr.poll(now=4) == []
    assert tr.poll(now=8) == []
    assert tr.poll(now=9.1) == [f]
    assert len(tr) == 0


def test_candidate_filter(tmp_path):
    root = tmp_path
    assert is_candidate(root / "a" / "x.NEF", root, set())
    assert not is_candidate(root / "a" / "x.txt", root, set())
    assert not is_candidate(root / "@eaDir" / "x.jpg", root, {"@eaDir"})
    assert not is_candidate(root / "._x.jpg", root, set())
    assert not is_candidate(root / "x.jpg.part", root, set())


def test_poster_layout_and_mockups(tmp_path):
    from src.mockup_generator import MOCKUP_SIZE, mockup_black_frame, mockup_oak_frame, mockup_sizes_infographic
    from src.print_generator import FORMATS, compute_poster_layout

    lay = compute_poster_layout(4800, 7200)
    x0, y0, x1, y1 = lay.photo_box
    assert x0 == y0 and 4800 - x1 == x0
    assert 7200 - y1 > y0  # deeper bottom margin for typography
    assert y1 < lay.title_top < lay.subtitle_top < 7200

    art = Image.new("RGB", (400, 600), (90, 120, 160))
    previews = {f.key: Image.new("RGB", (60, 90), (200, 100, 50)) for f in FORMATS}
    for im in (mockup_black_frame(art), mockup_oak_frame(art), mockup_sizes_infographic(previews, "portrait", str(tmp_path))):
        assert im.size == MOCKUP_SIZE and im.mode == "RGB"
