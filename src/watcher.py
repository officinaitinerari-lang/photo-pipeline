"""Recursive folder watcher with size-stability debounce before enqueueing into SQLite."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from .config import SUPPORTED_EXTENSIONS, Settings
from .database import Database, sha256_file

log = logging.getLogger(__name__)

_TEMP_SUFFIXES = (".part", ".tmp", ".crdownload", ".partial", ".download", "~")


def is_candidate(path: Path, input_root: Path, ignore_dirs: set[str]) -> bool:
    """True for supported image files that are not hidden/temp/system artefacts."""
    name = path.name
    if name.startswith((".", "~$", "._")) or name.lower().endswith(_TEMP_SUFFIXES):
        return False
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    try:
        rel_parts = path.relative_to(input_root).parts[:-1]
    except ValueError:
        return False
    return not any(p in ignore_dirs or p.startswith(".") for p in rel_parts)


@dataclass
class _Pending:
    size: int
    mtime: float
    stable_since: float


class StabilityTracker:
    """Tracks files until their size and mtime stay unchanged for `stable_seconds`."""

    def __init__(self, stable_seconds: float) -> None:
        self.stable_seconds = stable_seconds
        self._pending: dict[Path, _Pending] = {}
        self._lock = threading.Lock()

    def touch(self, path: Path) -> None:
        with self._lock:
            self._pending.setdefault(path, _Pending(-1, -1.0, time.monotonic()))

    def discard(self, path: Path) -> None:
        with self._lock:
            self._pending.pop(path, None)

    def __len__(self) -> int:
        return len(self._pending)

    def poll(self, now: float | None = None) -> list[Path]:
        """Return files that became stable; they are removed from tracking."""
        now = time.monotonic() if now is None else now
        ready: list[Path] = []
        with self._lock:
            for path, state in list(self._pending.items()):
                try:
                    st = path.stat()
                except FileNotFoundError:
                    del self._pending[path]
                    continue
                except OSError as exc:
                    log.warning("stat failed", extra={"path": str(path), "error": str(exc)})
                    continue
                if st.st_size != state.size or st.st_mtime != state.mtime:
                    state.size, state.mtime, state.stable_since = st.st_size, st.st_mtime, now
                    continue
                if st.st_size > 0 and now - state.stable_since >= self.stable_seconds:
                    ready.append(path)
                    del self._pending[path]
        return ready


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: FolderWatcher) -> None:
        self.watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.watcher.observe(Path(str(event.src_path)))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.watcher.observe(Path(str(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            self.watcher.scan_directory(Path(str(event.dest_path)))
        else:
            self.watcher.tracker.discard(Path(str(event.src_path)))
            self.watcher.observe(Path(str(event.dest_path)))

    def on_deleted(self, event: FileSystemEvent) -> None:
        self.watcher.tracker.discard(Path(str(event.src_path)))


class FolderWatcher:
    """watchdog observer + background stabiliser thread that enqueues stable files."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.root = settings.input_dir
        self.ignore_dirs = settings.ignore_dir_set
        self.tracker = StabilityTracker(settings.stable_seconds)
        self._stop = threading.Event()
        self._observer = (
            PollingObserver(timeout=settings.poll_interval_seconds) if settings.watch_use_polling else Observer()
        )
        self._thread = threading.Thread(target=self._stabilise_loop, name="stabiliser", daemon=True)

    def observe(self, path: Path) -> None:
        if is_candidate(path, self.root, self.ignore_dirs):
            self.tracker.touch(path)

    def scan_directory(self, directory: Path) -> int:
        count = 0
        for p in directory.rglob("*"):
            if p.is_file() and is_candidate(p, self.root, self.ignore_dirs):
                self.tracker.touch(p)
                count += 1
        return count

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._observer.schedule(_Handler(self), str(self.root), recursive=True)
        self._observer.start()
        self._thread.start()
        found = self.scan_directory(self.root)
        log.info(
            "watcher started",
            extra={"root": str(self.root), "polling": self.settings.watch_use_polling, "initial_files": found},
        )

    def stop(self) -> None:
        self._stop.set()
        self._observer.stop()
        self._observer.join(timeout=10)
        self._thread.join(timeout=10)

    def _stabilise_loop(self) -> None:
        while not self._stop.wait(1.0):
            for path in self.tracker.poll():
                try:
                    self._enqueue(path)
                except Exception:
                    log.exception("enqueue failed", extra={"path": str(path)})

    def _enqueue(self, path: Path) -> None:
        if not path.exists():
            return
        digest = sha256_file(path)
        outcome, job_id = self.db.enqueue(path, digest)
        log.info("file stable", extra={"path": str(path), "sha256": digest[:12], "outcome": outcome, "job_id": job_id})
        if outcome == "duplicate":
            dup_dir = self.settings.processed_dir / "_duplicates"
            dup_dir.mkdir(parents=True, exist_ok=True)
            target = dup_dir / path.name
            if target.exists():
                target = dup_dir / f"{path.stem}_{digest[:8]}{path.suffix}"
            shutil.move(str(path), target)
            log.warning("duplicate of an already completed photo moved aside", extra={"target": str(target)})
