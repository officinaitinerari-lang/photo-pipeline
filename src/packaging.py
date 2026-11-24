"""ZIP packaging with automatic split so every archive stays under Etsy's per-file limit."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

# Conservative per-entry overhead: local header (30) + central dir (46) + name twice + data
# descriptor/zip64 extras; plus end-of-central-directory record.
_ENTRY_OVERHEAD = 30 + 46 + 64
_ARCHIVE_OVERHEAD = 22 + 64


class PackagingError(Exception):
    pass


def _entry_cost(path: Path) -> int:
    return path.stat().st_size + _ENTRY_OVERHEAD + 2 * len(path.name.encode("utf-8"))


def plan_parts(files: list[Path], max_bytes: int) -> list[list[Path]]:
    """First-fit-decreasing bin packing of files into parts whose STORED zip fits max_bytes."""
    budget = max_bytes - _ARCHIVE_OVERHEAD
    bins: list[tuple[int, list[Path]]] = []
    for f in sorted(files, key=_entry_cost, reverse=True):
        cost = _entry_cost(f)
        if cost > budget:
            raise PackagingError(f"{f.name} ({f.stat().st_size} bytes) alone exceeds the {max_bytes} byte limit")
        for i, (used, items) in enumerate(bins):
            if used + cost <= budget:
                items.append(f)
                bins[i] = (used + cost, items)
                break
        else:
            bins.append((cost, [f]))
    # stable, human friendly order inside each part
    return [sorted(items, key=lambda p: p.name) for _, items in bins]


def package_files(files: list[Path], out_dir: Path, base_name: str, max_bytes: int) -> list[Path]:
    """Write `<base>.zip` or `<base>_part1.zip`, `<base>_part2.zip`... each <= max_bytes.

    JPEGs are already compressed, so entries are STORED (fast on low-power CPUs, size predictable).
    """
    if not files:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{base_name}*.zip"):
        stale.unlink()
    parts = plan_parts(files, max_bytes)
    written: list[Path] = []
    for idx, items in enumerate(parts, start=1):
        name = f"{base_name}.zip" if len(parts) == 1 else f"{base_name}_part{idx}.zip"
        target = out_dir / name
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for f in items:
                zf.write(f, arcname=f.name)
        size = target.stat().st_size
        if size > max_bytes:
            target.unlink()
            raise PackagingError(f"{name} is {size} bytes, above limit {max_bytes}")
        written.append(target)
        log.info("zip written", extra={"zip": name, "files": len(items), "bytes": size})
    return written
