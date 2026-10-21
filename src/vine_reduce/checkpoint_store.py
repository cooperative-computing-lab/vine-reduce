"""Sqlite-backed record of checkpoints and final results, so an interrupted run can restart
without redoing work: on startup the checkpoint rows for each (processor, dataset) are read and
restart doesn't reprocess events they already cover. See "Temporary Results, Checkpoints, and
Restart" in PLAN.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Sequence

from .event_ranges import Coverage

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2


def checksum_dataset(dataset: dict[str, Any]) -> str:
    """Stable hash of a dataset's contents, to detect definition changes between runs."""
    encoded = json.dumps(dataset, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointRecord:
    """One row of `checkpoints`, joined with its `checkpoint_files`.

    covers: event ranges, per dataset file URL, that this checkpoint's result represents
        (possibly part of a file).
    num_events / wall_time_s / memory_mb: totals accumulated into the result.
    is_final: a final result (results_dir) rather than an intermediate checkpoint (the
        distributor's durable storage).
    path: the serialized result file, or a pointer file (locator JSON) if `offloaded`.
    offloaded: a final result produced with result_postprocess_offload=True; always False for
        non-final checkpoints. Introspection-only.
    """

    id: int
    processor: str
    dataset: str
    covers: Coverage
    num_events: int
    wall_time_s: float
    memory_mb: float
    is_final: bool
    path: str
    offloaded: bool = False


class CheckpointStore:
    """Sqlite-backed checkpoint/final-result records for every (processor, dataset) pair."""

    def __init__(self, db_path: str):
        """Opens (creating if needed) the sqlite database at db_path."""
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        # Resumption aid only: if a crash loses a record, its work is just redone, so skip the
        # fsync for write throughput.
        self._conn.execute("PRAGMA synchronous = OFF")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        user_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if user_version == _SCHEMA_VERSION:
            return

        # Brand-new db (nothing to discard, stay silent) or an incompatible schema version
        # (discard everything and warn, rather than reconcile).
        discarded = self._count_existing_checkpoints()
        self._conn.execute("DROP TABLE IF EXISTS checkpoint_files")
        self._conn.execute("DROP TABLE IF EXISTS checkpoints")
        self._conn.execute("DROP TABLE IF EXISTS dataset_checksums")
        self._create_schema()
        self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        self._conn.commit()

        if discarded:
            logger.warning(
                "checkpoint schema changed; discarding %d existing checkpoint(s), "
                "run will restart from scratch",
                discarded,
            )

    def _count_existing_checkpoints(self) -> int:
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'checkpoints'"
        ).fetchone()
        if exists is None:
            return 0
        return self._conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE checkpoints (
                id INTEGER PRIMARY KEY,
                processor   TEXT    NOT NULL,
                dataset     TEXT    NOT NULL,
                num_events  INTEGER NOT NULL,
                wall_time_s REAL    NOT NULL,
                memory_mb   REAL    NOT NULL,
                is_final    INTEGER NOT NULL,
                path        TEXT    NOT NULL,
                offloaded   INTEGER NOT NULL DEFAULT 0
            )
            """)
        self._conn.execute("CREATE INDEX checkpoints_by_pair ON checkpoints(processor, dataset)")
        self._conn.execute("""
            CREATE TABLE checkpoint_files (
                checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
                file_url      TEXT    NOT NULL,
                start         INTEGER NOT NULL,
                stop          INTEGER NOT NULL,
                PRIMARY KEY (checkpoint_id, file_url, start)
            )
            """)
        self._conn.execute("""
            CREATE TABLE dataset_checksums (
                dataset  TEXT PRIMARY KEY,
                checksum TEXT NOT NULL
            )
            """)

    def reset_if_dataset_changed(self, dataset: str, checksum: str) -> list[str]:
        """If checksum differs from the one on record for dataset (or none exists), record it and
        delete the dataset's checkpoints. Returns the discarded rows' result file paths, which
        the caller must unlink (only the database is touched); empty if the checksum matched."""
        row = self._conn.execute(
            "SELECT checksum FROM dataset_checksums WHERE dataset = ?", (dataset,)
        ).fetchone()
        if row is not None and row["checksum"] == checksum:
            return []

        discarded_paths = [
            r["path"]
            for r in self._conn.execute(
                "SELECT path FROM checkpoints WHERE dataset = ?", (dataset,)
            ).fetchall()
        ]

        with self._conn:
            # ON DELETE CASCADE clears the matching checkpoint_files rows too.
            self._conn.execute("DELETE FROM checkpoints WHERE dataset = ?", (dataset,))
            self._conn.execute(
                "INSERT INTO dataset_checksums(dataset, checksum) VALUES (?, ?) "
                "ON CONFLICT(dataset) DO UPDATE SET checksum = excluded.checksum",
                (dataset, checksum),
            )
        return discarded_paths

    def record(
        self,
        *,
        processor: str,
        dataset: str,
        covers: Coverage,
        num_events: int,
        wall_time_s: float,
        memory_mb: float,
        is_final: bool,
        path: str,
        supersedes: Sequence[int] = (),
        offloaded: bool = False,
    ) -> int:
        """Insert a checkpoint and delete the rows it supersedes, in one transaction. Returns the
        new row id."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO checkpoints"
                " (processor, dataset, num_events, wall_time_s, memory_mb, is_final, path,"
                "  offloaded)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    processor,
                    dataset,
                    num_events,
                    wall_time_s,
                    memory_mb,
                    int(is_final),
                    path,
                    int(offloaded),
                ),
            )
            row_id = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO checkpoint_files (checkpoint_id, file_url, start, stop)"
                " VALUES (?, ?, ?, ?)",
                [
                    (row_id, file_url, start, stop)
                    for file_url, ranges in covers.items()
                    for start, stop in ranges
                ],
            )
            for superseded_id in supersedes:
                # ON DELETE CASCADE clears the matching checkpoint_files rows too.
                self._conn.execute("DELETE FROM checkpoints WHERE id = ?", (superseded_id,))
        return row_id

    def checkpoints_for(self, processor: str, dataset: str) -> list[CheckpointRecord]:
        """All checkpoint rows (final and intermediate) for one (processor, dataset)."""
        rows = self._conn.execute(
            "SELECT id, processor, dataset, num_events, wall_time_s, memory_mb, is_final, path,"
            "  offloaded"
            " FROM checkpoints WHERE processor = ? AND dataset = ?",
            (processor, dataset),
        ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def _record_from_row(self, row: sqlite3.Row) -> CheckpointRecord:
        ranges = self._conn.execute(
            "SELECT file_url, start, stop FROM checkpoint_files"
            " WHERE checkpoint_id = ? ORDER BY file_url, start",
            (row["id"],),
        ).fetchall()
        covers: dict[str, list[tuple[int, int]]] = {}
        for r in ranges:
            covers.setdefault(r["file_url"], []).append((r["start"], r["stop"]))
        return CheckpointRecord(
            id=row["id"],
            processor=row["processor"],
            dataset=row["dataset"],
            covers={url: tuple(pairs) for url, pairs in covers.items()},
            num_events=row["num_events"],
            wall_time_s=row["wall_time_s"],
            memory_mb=row["memory_mb"],
            is_final=bool(row["is_final"]),
            path=row["path"],
            offloaded=bool(row["offloaded"]),
        )

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        self._conn.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
