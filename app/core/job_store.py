"""
SQLite-backed job persistence for PatchForge.

Jobs are serialized as JSON in a single-table SQLite database.
A fast in-memory cache is maintained for read performance, with
writes going through to disk. On startup, the cache is populated
from the database so jobs survive restarts.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from app.models.job import Job

logger = logging.getLogger("patchforge.job_store")

_DB_PATH = Path("patchforge_jobs.db")
_cache: dict[str, Job] = {}
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute(
        "CREATE TABLE IF NOT EXISTS jobs ("
        "  id TEXT PRIMARY KEY,"
        "  data TEXT NOT NULL"
        ")"
    )
    _conn.commit()
    return _conn


def _init_cache() -> None:
    """Load all jobs from disk into the memory cache."""
    if _cache:
        return
    conn = _get_conn()
    rows = conn.execute("SELECT id, data FROM jobs").fetchall()
    for row_id, row_data in rows:
        try:
            _cache[row_id] = Job.model_validate_json(row_data)
        except Exception:
            logger.warning("Failed to deserialize job %s; skipping.", row_id)
    if rows:
        logger.info("Loaded %d jobs from database.", len(_cache))


def get_job(job_id: str) -> Optional[Job]:
    _init_cache()
    return _cache.get(job_id)


def store_job(job: Job) -> None:
    _init_cache()
    job.serialize_contours()
    _cache[job.id] = job
    data = job.model_dump_json(exclude={"contours"})
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO jobs (id, data) VALUES (?, ?)",
        (job.id, data),
    )
    conn.commit()


def all_jobs() -> list[Job]:
    _init_cache()
    return list(_cache.values())


def delete_job(job_id: str) -> None:
    _init_cache()
    _cache.pop(job_id, None)
    conn = _get_conn()
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()
