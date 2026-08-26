"""Sqlite-backed record of checkpoints and final results.

Used so an interrupted run can be restarted without redoing work: on
startup, vine_reduce reads the checkpoint rows for each (processor, dataset)
and skips any files they already cover. See "Implementation Clarifications"
in PLAN.md.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any


def checksum_dataset(dataset: dict[str, Any]) -> str:
    """A stable hash of a dataset's contents, used to detect when a
    dataset's definition has changed since the last run (see
    CheckpointDB.dataset_changed)."""
    encoded = json.dumps(dataset, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CheckpointRow:
    """One row of the `checkpoints` table.

    id: row id (sqlite AUTOINCREMENT primary key).
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
    covers_files: list[str]
    num_events: int
    wall_time_s: float
    memory_mb: float
    is_final: bool
    path: str


class CheckpointDB:
    """Sqlite-backed record of checkpoints and final results for every
    (processor, dataset) pair, used to resume an interrupted run without
    redoing finished work. See the module docstring."""

    def __init__(self, db_path: str):
        """Opens (creating if needed) the sqlite database at db_path."""
        self._conn = sqlite3.connect(db_path)
        # This DB is a work-resumption aid, not a record that must survive a
        # power loss: on crash, a checkpoint just fails to record and the
        # work it covered is redone. Skipping the fsync makes that trade for
        # write throughput, since correctness doesn't depend on durability.
        self._conn.execute("PRAGMA synchronous = OFF")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                processor TEXT NOT NULL,
                dataset TEXT NOT NULL,
                covers_files TEXT NOT NULL,
                num_events INTEGER NOT NULL,
                wall_time_s REAL NOT NULL,
                memory_mb REAL NOT NULL,
                is_final INTEGER NOT NULL,
                path TEXT NOT NULL
            )
            """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_checksums (
                dataset TEXT PRIMARY KEY,
                checksum TEXT NOT NULL
            )
            """)
        self._conn.commit()

    def dataset_changed(self, dataset: str, checksum: str) -> bool:
        """Compares checksum to what's on record for dataset. If different (or
        not on record yet), records it, discards any checkpoints on file for
        that dataset (they no longer apply), and returns True."""
        row = self._conn.execute(
            "SELECT checksum FROM dataset_checksums WHERE dataset = ?", (dataset,)
        ).fetchone()
        if row is not None and row[0] == checksum:
            return False

        self._conn.execute("DELETE FROM checkpoints WHERE dataset = ?", (dataset,))
        self._conn.execute(
            "INSERT INTO dataset_checksums(dataset, checksum) VALUES (?, ?) "
            "ON CONFLICT(dataset) DO UPDATE SET checksum = excluded.checksum",
            (dataset, checksum),
        )
        self._conn.commit()
        return True

    def add_checkpoint(
        self,
        processor: str,
        dataset: str,
        covers_files: list[str],
        num_events: int,
        wall_time_s: float,
        memory_mb: float,
        is_final: bool,
        path: str,
        commit: bool = True,
    ) -> int:
        """Insert a new checkpoint row and return its id. commit=False lets
        a caller that also has delete_checkpoint calls for the same event
        (e.g. replacing checkpoints on a reduction) batch them into one
        transaction via a single trailing self.commit()."""
        cur = self._conn.execute(
            "INSERT INTO checkpoints"
            " (processor, dataset, covers_files, num_events, wall_time_s, memory_mb,"
            " is_final, path)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                processor,
                dataset,
                json.dumps(covers_files),
                num_events,
                wall_time_s,
                memory_mb,
                int(is_final),
                path,
            ),
        )
        if commit:
            self._conn.commit()
        return cur.lastrowid

    def delete_checkpoint(self, row_id: int, commit: bool = True) -> None:
        """Remove a checkpoint row by id. commit=False defers the write to a
        later self.commit(), as with add_checkpoint."""
        self._conn.execute("DELETE FROM checkpoints WHERE id = ?", (row_id,))
        if commit:
            self._conn.commit()

    def commit(self) -> None:
        """Commit any pending writes made with commit=False."""
        self._conn.commit()

    def checkpoints_for(self, processor: str, dataset: str) -> list[CheckpointRow]:
        """All checkpoint rows (final and intermediate) on record for one
        (processor, dataset) pair."""
        rows = self._conn.execute(
            "SELECT id, processor, dataset, covers_files, num_events, wall_time_s,"
            " memory_mb, is_final, path FROM checkpoints WHERE processor = ? AND dataset = ?",
            (processor, dataset),
        ).fetchall()
        return [
            CheckpointRow(
                id=r[0],
                processor=r[1],
                dataset=r[2],
                covers_files=json.loads(r[3]),
                num_events=r[4],
                wall_time_s=r[5],
                memory_mb=r[6],
                is_final=bool(r[7]),
                path=r[8],
            )
            for r in rows
        ]

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        self._conn.close()

    def __enter__(self) -> "CheckpointDB":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
