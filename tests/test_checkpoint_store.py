from __future__ import annotations

import json
import logging
import sqlite3

import pytest

from vine_reduce.checkpoint_store import CheckpointStore, checksum_dataset


def test_record_and_checkpoints_for_round_trip(tmp_path):
    store = CheckpointStore(str(tmp_path / "db.sqlite"))
    row_id = store.record(
        processor="proc",
        dataset="ds",
        covers_files=["a.root"],
        num_events=10,
        wall_time_s=1.5,
        memory_mb=2.5,
        is_final=False,
        path="/tmp/x.pkl",
    )

    rows = store.checkpoints_for("proc", "ds")
    assert len(rows) == 1
    row = rows[0]
    assert row.id == row_id
    assert row.processor == "proc"
    assert row.dataset == "ds"
    assert row.covers_files == frozenset({"a.root"})
    assert row.num_events == 10
    assert row.wall_time_s == 1.5
    assert row.memory_mb == 2.5
    assert row.is_final is False
    assert row.path == "/tmp/x.pkl"

    assert store.checkpoints_for("proc", "other-ds") == []
    store.close()


def test_record_supersedes_deletes_old_rows_and_cascades_files(tmp_path):
    store = CheckpointStore(str(tmp_path / "db.sqlite"))
    old_id = store.record(
        processor="proc",
        dataset="ds",
        covers_files=["a.root"],
        num_events=5,
        wall_time_s=1.0,
        memory_mb=1.0,
        is_final=False,
        path="/tmp/old.pkl",
    )

    new_id = store.record(
        processor="proc",
        dataset="ds",
        covers_files=["a.root", "b.root"],
        num_events=10,
        wall_time_s=2.0,
        memory_mb=2.0,
        is_final=False,
        path="/tmp/new.pkl",
        supersedes=[old_id],
    )

    rows = store.checkpoints_for("proc", "ds")
    assert [row.id for row in rows] == [new_id]

    # the superseded row's checkpoint_files rows must be gone too (cascade).
    remaining_files = store._conn.execute(
        "SELECT * FROM checkpoint_files WHERE checkpoint_id = ?", (old_id,)
    ).fetchall()
    assert remaining_files == []
    store.close()


def test_record_is_atomic_when_insert_fails(tmp_path):
    store = CheckpointStore(str(tmp_path / "db.sqlite"))
    old_id = store.record(
        processor="proc",
        dataset="ds",
        covers_files=["a.root"],
        num_events=5,
        wall_time_s=1.0,
        memory_mb=1.0,
        is_final=False,
        path="/tmp/old.pkl",
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.record(
            processor="proc",
            dataset="ds",
            covers_files=["a.root", "b.root"],
            num_events=10,
            wall_time_s=2.0,
            memory_mb=2.0,
            is_final=False,
            path=None,  # violates the NOT NULL constraint on `path`
            supersedes=[old_id],
        )

    # the failed insert must not have taken the supersede-delete with it.
    rows = store.checkpoints_for("proc", "ds")
    assert [row.id for row in rows] == [old_id]
    store.close()


def test_dataset_changed_first_time_is_true(tmp_path):
    store = CheckpointStore(str(tmp_path / "db.sqlite"))
    assert store.dataset_changed("ds", "checksum-1") is True
    store.close()


def test_dataset_changed_stable_checksum_is_false(tmp_path):
    store = CheckpointStore(str(tmp_path / "db.sqlite"))
    store.dataset_changed("ds", "checksum-1")
    assert store.dataset_changed("ds", "checksum-1") is False
    store.close()


def test_dataset_changed_wipes_checkpoints_for_that_dataset_only(tmp_path):
    store = CheckpointStore(str(tmp_path / "db.sqlite"))
    store.dataset_changed("ds1", "checksum-1")
    store.dataset_changed("ds2", "checksum-1")
    store.record(
        processor="proc",
        dataset="ds1",
        covers_files=["a.root"],
        num_events=10,
        wall_time_s=1.0,
        memory_mb=1.0,
        is_final=False,
        path="/tmp/a.pkl",
    )
    store.record(
        processor="proc",
        dataset="ds2",
        covers_files=["b.root"],
        num_events=10,
        wall_time_s=1.0,
        memory_mb=1.0,
        is_final=False,
        path="/tmp/b.pkl",
    )

    assert store.dataset_changed("ds1", "checksum-2") is True

    assert store.checkpoints_for("proc", "ds1") == []
    assert len(store.checkpoints_for("proc", "ds2")) == 1
    store.close()


def test_checksum_dataset_stable_and_sensitive_to_content():
    d1 = {"metadata": {"x": 1}, "files": {"a.root": 10}}
    d2 = {"files": {"a.root": 10}, "metadata": {"x": 1}}  # different key order
    d3 = {"metadata": {"x": 2}, "files": {"a.root": 10}}

    assert checksum_dataset(d1) == checksum_dataset(d2)
    assert checksum_dataset(d1) != checksum_dataset(d3)


def _write_old_schema_db(db_path: str) -> None:
    """Build a db in the shape today's (pre-rewrite) CheckpointDB produces:
    a single `checkpoints` table with a JSON `covers_files` TEXT column, and
    no user_version set (defaults to 0)."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE checkpoints (
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
    conn.execute("""
        CREATE TABLE dataset_checksums (
            dataset TEXT PRIMARY KEY,
            checksum TEXT NOT NULL
        )
        """)
    conn.execute(
        "INSERT INTO checkpoints"
        " (processor, dataset, covers_files, num_events, wall_time_s, memory_mb,"
        " is_final, path)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("proc", "ds", json.dumps(["a.root"]), 5, 1.0, 1.0, 0, "/tmp/old.pkl"),
    )
    conn.commit()
    conn.close()


def test_opening_an_old_schema_db_discards_its_rows_and_stamps_user_version(tmp_path, caplog):
    db_path = str(tmp_path / "db.sqlite")
    _write_old_schema_db(db_path)

    with caplog.at_level(logging.WARNING):
        store = CheckpointStore(db_path)

    assert store.checkpoints_for("proc", "ds") == []
    user_version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert user_version == 1
    assert "discarding 1 existing checkpoint" in caplog.text
    store.close()


def test_opening_a_brand_new_db_does_not_warn(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        store = CheckpointStore(str(tmp_path / "db.sqlite"))

    assert caplog.text == ""
    store.close()
