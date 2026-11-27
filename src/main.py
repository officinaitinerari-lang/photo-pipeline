"""Entry point: watcher + sequential worker loop, plus maintenance sub-commands.

Usage:
    python -m src.main run              # default: watch /data/input and process forever
    python -m src.main process FILE     # process a single file now (it is moved like any other)
    python -m src.main upload-pending   # push uploads left in DRY_RUN/failed state (needs DRY_RUN=false)
    python -m src.main retry-failed     # re-queue every failed job
    python -m src.main status           # print job counts
    python -m src.main healthcheck      # exit 0 if the worker heartbeat is fresh
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from types import FrameType

from .config import Settings, get_settings
from .database import Database, sha256_file
from .logging_setup import setup_logging
from .pipeline import Pipeline
from .watcher import FolderWatcher

log = logging.getLogger("pipeline.main")

UPLOAD_RETRY_INTERVAL_S = 3600
HEARTBEAT_MAX_AGE_S = 300


def _write_heartbeat(path: Path) -> None:
    try:
        path.write_text(str(int(time.time())), encoding="utf-8")
    except OSError:
        log.warning("cannot write heartbeat", extra={"path": str(path)})


def cmd_run(settings: Settings) -> int:
    problems = settings.validate_runtime()
    if problems:
        for p in problems:
            log.critical("configuration error", extra={"problem": p})
        return 2
    db = Database(settings.db_path)
    reset = db.reset_stale_processing()
    if reset:
        log.warning("recovered interrupted jobs after restart", extra={"count": reset})
    pipeline = Pipeline(settings, db)
    watcher = FolderWatcher(settings, db)
    stop = threading.Event()

    def _on_signal(signum: int, _frame: FrameType | None) -> None:
        log.info("shutdown requested", extra={"signal": signum})
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    watcher.start()
    log.info(
        "pipeline running",
        extra={"dry_run": settings.dry_run, "targets": [t.name for t in settings.upload_targets()],
               "model": settings.gemini_model, "upscale_engine": settings.upscale_engine},
    )
    last_upload_retry = time.monotonic() - UPLOAD_RETRY_INTERVAL_S - 1  # first retry right after start-up
    etsy_interval = settings.etsy_sync_minutes * 60
    last_etsy_sync = time.monotonic() - etsy_interval - 1  # first Etsy sync right after start-up
    try:
        while not stop.is_set():
            _write_heartbeat(settings.heartbeat_path)
            job = db.claim_next()
            if job is None:
                if not settings.dry_run and time.monotonic() - last_upload_retry > UPLOAD_RETRY_INTERVAL_S:
                    last_upload_retry = time.monotonic()
                    pipeline.retry_pending_uploads()
                if settings.etsy_enabled and not settings.dry_run and time.monotonic() - last_etsy_sync > etsy_interval:
                    last_etsy_sync = time.monotonic()
                    pipeline.sync_etsy_drafts()
                stop.wait(settings.worker_idle_seconds)
                continue
            pipeline.run_job(job)
            gc.collect()
    finally:
        watcher.stop()
        db.close()
        log.info("pipeline stopped")
    return 0


def cmd_process(settings: Settings, file: Path) -> int:
    if not file.is_file():
        log.error("file not found", extra={"path": str(file)})
        return 1
    db = Database(settings.db_path)
    outcome, job_id = db.enqueue(file.resolve(), sha256_file(file))
    if outcome == "duplicate":
        log.warning("already processed (same SHA-256)", extra={"job_id": job_id})
        return 0
    db.update_job(job_id, status="processing", next_attempt_at=0)
    job = db.get_job(job_id)
    if job is None:
        return 1
    Pipeline(settings, db).run_job(job)
    final = db.get_job(job_id)
    print(json.dumps({"job_id": job_id, "status": final.status if final else None, "error": final.error if final else None}))
    return 0 if final and final.status == "completed" else 1


def cmd_retry_failed(settings: Settings, db: Database) -> int:
    """Move originals back from /data/failed to /data/input and re-queue their jobs."""
    for job in db.failed_jobs():
        src = Path(job.src_path)
        try:
            rel = src.relative_to(settings.failed_dir)
        except ValueError:
            continue
        if not src.exists():
            continue
        dest = settings.input_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dest)
        src.with_name(src.name + ".error.log").unlink(missing_ok=True)
        db.update_job(job.id, src_path=str(dest))
    return db.requeue_failed()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="photo-pipeline")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run")
    p_proc = sub.add_parser("process")
    p_proc.add_argument("file", type=Path)
    sub.add_parser("upload-pending")
    sub.add_parser("retry-failed")
    sub.add_parser("status")
    sub.add_parser("healthcheck")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    command = args.command or "run"

    if command == "healthcheck":
        try:
            age = time.time() - int(settings.heartbeat_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 1
        return 0 if age < HEARTBEAT_MAX_AGE_S else 1

    settings.ensure_directories()
    if command == "run":
        return cmd_run(settings)
    if command == "process":
        return cmd_process(settings, args.file)
    db = Database(settings.db_path)
    if command == "upload-pending":
        n = Pipeline(settings, db).retry_pending_uploads()
        print(json.dumps({"uploaded": n, "dry_run": settings.dry_run}))
        return 0
    if command == "retry-failed":
        print(json.dumps({"requeued": cmd_retry_failed(settings, db)}))
        return 0
    if command == "status":
        print(json.dumps(db.status_counts(), indent=2))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
