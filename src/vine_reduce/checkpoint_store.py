"""Sqlite-backed record of checkpoints and final results.

Used so an interrupted run can be restarted without redoing work: on
startup, vine_reduce reads the checkpoint rows for each (processor, dataset)
and skips any files they already cover. See "Temporary Results, Checkpoints,
and Restart" in PLAN.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def checksum_dataset(dataset: dict[str, Any]) -> str:
    """A stable hash of a dataset's contents, used to detect when a
    dataset's definition has changed since the last run (see
    CheckpointStore.dataset_changed)."""
    encoded = json.dumps(dataset, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointRecord:
    """One row of the `checkpoints` table, joined with its `checkpoint_files`.

    id: row id (sqlite primary key).
    processor / dataset: the (processor, dataset) pair this checkpoint
        belongs to.
    covers_files: dataset file URLs whose data this checkpoint's result
        represents.
    num_events / wall_time_s / memory_mb: totals accumulated into this
        checkpoint's result.
    is_final: whether this is a final result (results_dir) or an
        intermediate checkpoint (wherever the distributor durably stores
        one - e.g. TaskVineDistributor's own checkpoint_dir).
    path: where the checkpoint's serialized result file lives on disk.
    """

    id: int
    processor: str
    dataset: str
    covers_files: frozenset[str]
    num_events: int
    wall_time_s: float
    memory_mb: float
    is_final: bool
    path: str


class CheckpointStore:
    """Sqlite-backed record of checkpoints and final results for every
    (processor, dataset) pair, used to resume an interrupted run without
    redoing finished work. See the module docstring."""

    def __init__(self, db_path: str):
        """Opens (creating if needed) the sqlite database at db_path."""
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        # This DB is a work-resumption aid, not a record that must survive a
        # power loss: on crash, a checkpoint just fails to record and the
        # work it covered is redone. Skipping the fsync makes that trade for
        # write throughput, since correctness doesn't depend on durability.
        self._conn.execute("PRAGMA synchronous = OFF")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        user_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if user_version == _SCHEMA_VERSION:
            return

        # Either a brand-new db (nothing to discard, so stay silent) or one
        # written by an incompatible/old schema (discard it and warn - the
        # same "discard, don't try to reconcile" trade dataset_changed
        # already makes for a single dataset, extended here to the whole db).
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
        tables = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'checkpoints'"
            )
        }
        if "checkpoints" not in tables:
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
                path        TEXT    NOT NULL
            )
            """)
        self._conn.execute("CREATE INDEX checkpoints_by_pair ON checkpoints(processor, dataset)")
        self._conn.execute("""
            CREATE TABLE checkpoint_files (
                checkpoint_id INTEGER NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
                file_url      TEXT    NOT NULL,
                PRIMARY KEY (checkpoint_id, file_url)
            )
            """)
        self._conn.execute("""
            CREATE TABLE dataset_checksums (
                dataset  TEXT PRIMARY KEY,
                checksum TEXT NOT NULL
            )
            """)

    def dataset_changed(self, dataset: str, checksum: str) -> bool:
        """Compares checksum to what's on record for dataset. If different (or
        not on record yet), records it, discards any checkpoints on file for
        that dataset (they no longer apply), and returns True."""
        row = self._conn.execute(
            "SELECT checksum FROM dataset_checksums WHERE dataset = ?", (dataset,)
        ).fetchone()
        if row is not None and row["checksum"] == checksum:
            return False

        with self._conn:
            # ON DELETE CASCADE clears the matching checkpoint_files rows too.
            self._conn.execute("DELETE FROM checkpoints WHERE dataset = ?", (dataset,))
            self._conn.execute(
                "INSERT INTO dataset_checksums(dataset, checksum) VALUES (?, ?) "
                "ON CONFLICT(dataset) DO UPDATE SET checksum = excluded.checksum",
                (dataset, checksum),
            )
        return True

    def record(
        self,
        *,
        processor: str,
        dataset: str,
        covers_files: Iterable[str],
        num_events: int,
        wall_time_s: float,
        memory_mb: float,
        is_final: bool,
        path: str,
        supersedes: Sequence[int] = (),
    ) -> int:
        """Insert this checkpoint and delete the rows it supersedes, in ONE
        transaction. Returns the new row id. This is the only write path for
        checkpoints - there is no commit= parameter and no public commit()."""
        covers_files = sorted(covers_files)
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO checkpoints"
                " (processor, dataset, num_events, wall_time_s, memory_mb, is_final, path)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (processor, dataset, num_events, wall_time_s, memory_mb, int(is_final), path),
            )
            row_id = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO checkpoint_files (checkpoint_id, file_url) VALUES (?, ?)",
                [(row_id, file_url) for file_url in covers_files],
            )
            for superseded_id in supersedes:
                # ON DELETE CASCADE clears the matching checkpoint_files rows too.
                self._conn.execute("DELETE FROM checkpoints WHERE id = ?", (superseded_id,))
        return row_id

    def checkpoints_for(self, processor: str, dataset: str) -> list[CheckpointRecord]:
        """All checkpoint rows (final and intermediate) on record for one
        (processor, dataset) pair."""
        rows = self._conn.execute(
            "SELECT id, processor, dataset, num_events, wall_time_s, memory_mb, is_final, path"
            " FROM checkpoints WHERE processor = ? AND dataset = ?",
            (processor, dataset),
        ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def _record_from_row(self, row: sqlite3.Row) -> CheckpointRecord:
        files = self._conn.execute(
            "SELECT file_url FROM checkpoint_files WHERE checkpoint_id = ?", (row["id"],)
        ).fetchall()
        return CheckpointRecord(
            id=row["id"],
            processor=row["processor"],
            dataset=row["dataset"],
            covers_files=frozenset(f["file_url"] for f in files),
            num_events=row["num_events"],
            wall_time_s=row["wall_time_s"],
            memory_mb=row["memory_mb"],
            is_final=bool(row["is_final"]),
            path=row["path"],
        )

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        self._conn.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
