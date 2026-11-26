import os
import zipfile

import pytest

from src.packaging import PackagingError, TooManyPartsError, package_files, plan_parts

MB = 1024 * 1024


def _make_files(tmp_path, sizes_mb):
    files = []
    for i, s in enumerate(sizes_mb):
        p = tmp_path / f"print_{i}.jpg"
        p.write_bytes(os.urandom(int(s * MB)))  # incompressible, like a JPEG
        files.append(p)
    return files


def test_single_zip_when_under_limit(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    files = _make_files(src, [2, 3, 1])
    out = package_files(files, tmp_path / "out", "prints_fullbleed", int(19.5 * MB))
    assert [p.name for p in out] == ["prints_fullbleed.zip"]
    assert out[0].stat().st_size <= 19.5 * MB


def test_split_into_parts_under_limit(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    sizes = [15, 14, 12, 9, 6, 4]  # 60 MB total
    files = _make_files(src, sizes)
    limit = int(19.5 * MB)
    out = package_files(files, tmp_path / "out", "prints_fullbleed", limit)
    assert len(out) >= 4
    assert [p.name for p in out] == [f"prints_fullbleed_part{i}.zip" for i in range(1, len(out) + 1)]
    seen = []
    for z in out:
        assert z.stat().st_size <= limit, z
        with zipfile.ZipFile(z) as zf:
            assert zf.testzip() is None
            seen += zf.namelist()
    assert sorted(seen) == sorted(f.name for f in files)  # each file exactly once


def test_boundary_close_to_limit(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    limit = 1 * MB
    files = _make_files(src, [0.49, 0.49, 0.49])
    out = package_files(files, tmp_path / "out", "p", limit)
    assert all(z.stat().st_size <= limit for z in out)
    assert len(out) == 2


def test_file_bigger_than_limit_raises(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    files = _make_files(src, [2])
    with pytest.raises(PackagingError):
        plan_parts(files, 1 * MB)


def test_stale_parts_removed(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "prints_fullbleed_part9.zip").write_bytes(b"old")
    package_files(_make_files(src, [1]), out_dir, "prints_fullbleed", 5 * MB)
    assert sorted(p.name for p in out_dir.iterdir()) == ["prints_fullbleed.zip"]


def test_max_parts_respected(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    files = _make_files(src, [0.6, 0.6, 0.6])
    out = package_files(files, tmp_path / "out", "p", 1 * MB, max_parts=3)
    assert len(out) == 3


def test_too_many_parts_raises_and_writes_nothing(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    files = _make_files(src, [0.6, 0.6, 0.6])
    out_dir = tmp_path / "out"
    with pytest.raises(TooManyPartsError):
        package_files(files, out_dir, "p", 1 * MB, max_parts=2)
    assert not out_dir.exists() or not list(out_dir.glob("*.zip"))
    assert issubclass(TooManyPartsError, PackagingError)
