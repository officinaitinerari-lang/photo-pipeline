"""SQLite state store: job queue, idempotency by SHA-256, AI result cache, upload tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

UPLOAD_PENDING = "pending"
UPLOAD_DRY_RUN = "dry_run"
UPLOAD_UPLOADED = "uploaded"
UPLOAD_FAILED = "failed"
UPLOAD_SKIPPED = "skipped"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256          TEXT NOT NULL UNIQUE,
    src_path        TEXT NOT NULL,
    status          TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    error           TEXT,
    slug            TEXT,
    is_analog       INTEGER,
    commercial_safe INTEGER,
    safety_notes    TEXT,
    ai_json         TEXT,
    microstock_path TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, next_attempt_at);

CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    platform    TEXT NOT NULL,
    local_path  TEXT NOT NULL,
    remote_name TEXT NOT NULL,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    uploaded_at REAL,
    updated_at  REAL NOT NULL,
    UNIQUE(job_id, platform)
);
"""


@dataclass(frozen=True)
class Job:
    id: int
    sha256: str
    src_path: str
    status: str
    attempts: int
    slug: str | None
    ai_json: str | None
    error: str | None

    @property
    def ai_data(self) -> dict[str, Any] | None:
        return json.loads(self.ai_json) if self.ai_json else None


@dataclass(frozen=True)
class Upload:
    id: int
    job_id: int
    platform: str
    local_path: str
    remote_name: str
    status: str
    attempts: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class Database:
    """Thread-safe wrapper (one connection per thread, WAL journal, serialized writes)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self._write_lock:
            # executescript manages its own transaction (it COMMITs first), so no BEGIN here
            self._connection().executescript(_SCHEMA)

    def _connection(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._connection()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------ jobs
    def enqueue(self, src_path: Path, sha256: str) -> tuple[str, int]:
        """Register a stable input file. Returns (outcome, job_id).

        outcome: "new" | "requeued" | "duplicate" | "already_queued"
        """
        now = time.time()
        with self._conn() as c:
            row = c.execute("SELECT id, status FROM jobs WHERE sha256=?", (sha256,)).fetchone()
            if row is None:
                cur = c.execute(
                    "INSERT INTO jobs(sha256, src_path, status, created_at, updated_at) VALUES (?,?,?,?,?)",
                    (sha256, str(src_path), STATUS_QUEUED, now, now),
                )
                return "new", int(cur.lastrowid or 0)
            job_id, status = int(row["id"]), row["status"]
            if status == STATUS_COMPLETED:
                return "duplicate", job_id
            if status == STATUS_FAILED:
                c.execute(
                    "UPDATE jobs SET status=?, src_path=?, attempts=0, next_attempt_at=0, error=NULL, updated_at=? "
                    "WHERE id=?",
                    (STATUS_QUEUED, str(src_path), now, job_id),
                )
                return "requeued", job_id
            c.execute("UPDATE jobs SET src_path=?, updated_at=? WHERE id=?", (str(src_path), now, job_id))
            return "already_queued", job_id

    def reset_stale_processing(self) -> int:
        """After a crash/restart, put interrupted jobs back in the queue."""
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET status=?, next_attempt_at=0, updated_at=? WHERE status=?",
                (STATUS_QUEUED, time.time(), STATUS_PROCESSING),
            )
            return cur.rowcount

    def claim_next(self) -> Job | None:
        now = time.time()
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM jobs WHERE status=? AND next_attempt_at<=? ORDER BY next_attempt_at, id LIMIT 1",
                (STATUS_QUEUED, now),
            ).fetchone()
            if row is None:
                return None
            c.execute(
                "UPDATE jobs SET status=?, attempts=attempts+1, updated_at=? WHERE id=?",
                (STATUS_PROCESSING, now, row["id"]),
            )
        return self.get_job(int(row["id"]))

    def get_job(self, job_id: int) -> Job | None:
        row = self._connection().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return Job(
            id=int(row["id"]),
            sha256=row["sha256"],
            src_path=row["src_path"],
            status=row["status"],
            attempts=int(row["attempts"]),
            slug=row["slug"],
            ai_json=row["ai_json"],
            error=row["error"],
        )

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "src_path", "status", "error", "slug", "is_analog", "commercial_safe",
            "safety_notes", "ai_json", "microstock_path", "next_attempt_at", "attempts",
        }
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Unknown job fields: {bad}")
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def defer_job(self, job_id: int, delay_seconds: float, error: str, refund_attempt: bool = False) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET status=?, next_attempt_at=?, error=?, "
                "attempts=CASE WHEN ? THEN MAX(attempts-1,0) ELSE attempts END, updated_at=? WHERE id=?",
                (STATUS_QUEUED, time.time() + delay_seconds, error, int(refund_attempt), time.time(), job_id),
            )

    def failed_jobs(self) -> list[Job]:
        rows = self._connection().execute("SELECT id FROM jobs WHERE status=?", (STATUS_FAILED,)).fetchall()
        return [j for r in rows if (j := self.get_job(int(r["id"]))) is not None]

    def requeue_failed(self) -> int:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE jobs SET status=?, attempts=0, next_attempt_at=0, updated_at=? WHERE status=?",
                (STATUS_QUEUED, time.time(), STATUS_FAILED),
            )
            return cur.rowcount

    def get_microstock_path(self, job_id: int) -> str | None:
        row = self._connection().execute("SELECT microstock_path FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row["microstock_path"] if row else None

    def status_counts(self) -> dict[str, int]:
        rows = self._connection().execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    def known_paths_in_flight(self) -> set[str]:
        rows = self._connection().execute(
            "SELECT src_path FROM jobs WHERE status IN (?,?)", (STATUS_QUEUED, STATUS_PROCESSING)
        ).fetchall()
        return {r["src_path"] for r in rows}

    # --------------------------------------------------------------- uploads
    def ensure_upload(self, job_id: int, platform: str, local_path: Path, remote_name: str) -> Upload:
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO uploads(job_id, platform, local_path, remote_name, status, updated_at) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(job_id, platform) DO UPDATE SET "
                "local_path=excluded.local_path, remote_name=excluded.remote_name, updated_at=excluded.updated_at",
                (job_id, platform, str(local_path), remote_name, UPLOAD_PENDING, now),
            )
        row = self._connection().execute(
            "SELECT * FROM uploads WHERE job_id=? AND platform=?", (job_id, platform)
        ).fetchone()
        return self._row_to_upload(row)

    def jobs_missing_upload(self, platform: str) -> list[tuple[int, str]]:
        """Completed, commercially safe jobs that have a microstock file but no upload row for `platform`.

        Happens when a photo was processed while that platform was still disabled.
        """
        rows = self._connection().execute(
            "SELECT id, microstock_path FROM jobs j WHERE status=? AND commercial_safe=1 "
            "AND microstock_path IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM uploads u WHERE u.job_id=j.id AND u.platform=?) ORDER BY id",
            (STATUS_COMPLETED, platform),
        ).fetchall()
        return [(int(r["id"]), r["microstock_path"]) for r in rows]

    def set_upload_status(self, upload_id: int, status: str, error: str | None = None) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                "UPDATE uploads SET status=?, error=?, attempts=attempts+1, updated_at=?, "
                "uploaded_at=CASE WHEN ?='uploaded' THEN ? ELSE uploaded_at END WHERE id=?",
                (status, error, now, status, now, upload_id),
            )

    def update_upload_path(self, upload_id: int, local_path: Path) -> None:
        with self._conn() as c:
            c.execute("UPDATE uploads SET local_path=?, updated_at=? WHERE id=?", (str(local_path), time.time(), upload_id))

    def uploads_for_job(self, job_id: int) -> list[Upload]:
        rows = self._connection().execute("SELECT * FROM uploads WHERE job_id=?", (job_id,)).fetchall()
        return [self._row_to_upload(r) for r in rows]

    def uploads_needing_transfer(self) -> list[Upload]:
        rows = self._connection().execute(
            "SELECT * FROM uploads WHERE status IN (?,?,?) ORDER BY id",
            (UPLOAD_PENDING, UPLOAD_DRY_RUN, UPLOAD_FAILED),
        ).fetchall()
        return [self._row_to_upload(r) for r in rows]

    @staticmethod
    def _row_to_upload(row: sqlite3.Row) -> Upload:
        return Upload(
            id=int(row["id"]),
            job_id=int(row["job_id"]),
            platform=row["platform"],
            local_path=row["local_path"],
            remote_name=row["remote_name"],
            status=row["status"],
            attempts=int(row["attempts"]),
        )
