from __future__ import annotations

from vine_reduce.checkpoint_db import CheckpointDB, checksum_dataset


def test_add_and_query_checkpoint(tmp_path):
    db = CheckpointDB(str(tmp_path / "db.sqlite"))
    row_id = db.add_checkpoint("proc", "ds", ["a.root"], 10, 1.5, 2.5, False, "/tmp/x.pkl")

    rows = db.checkpoints_for("proc", "ds")
    assert len(rows) == 1
    assert rows[0].id == row_id
    assert rows[0].covers_files == ["a.root"]
    assert rows[0].num_events == 10
    assert rows[0].is_final is False

    assert db.checkpoints_for("proc", "other-ds") == []
    db.close()


def test_delete_checkpoint(tmp_path):
    db = CheckpointDB(str(tmp_path / "db.sqlite"))
    row_id = db.add_checkpoint("proc", "ds", ["a.root"], 10, 1.5, 2.5, False, "/tmp/x.pkl")
    db.delete_checkpoint(row_id)
    assert db.checkpoints_for("proc", "ds") == []
    db.close()


def test_dataset_changed_first_time_is_true(tmp_path):
    db = CheckpointDB(str(tmp_path / "db.sqlite"))
    assert db.dataset_changed("ds", "checksum-1") is True
    db.close()


def test_dataset_changed_stable_checksum_is_false(tmp_path):
    db = CheckpointDB(str(tmp_path / "db.sqlite"))
    db.dataset_changed("ds", "checksum-1")
    assert db.dataset_changed("ds", "checksum-1") is False
    db.close()


def test_dataset_changed_wipes_checkpoints_for_that_dataset_only(tmp_path):
    db = CheckpointDB(str(tmp_path / "db.sqlite"))
    db.dataset_changed("ds1", "checksum-1")
    db.dataset_changed("ds2", "checksum-1")
    db.add_checkpoint("proc", "ds1", ["a.root"], 10, 1.0, 1.0, False, "/tmp/a.pkl")
    db.add_checkpoint("proc", "ds2", ["b.root"], 10, 1.0, 1.0, False, "/tmp/b.pkl")

    assert db.dataset_changed("ds1", "checksum-2") is True

    assert db.checkpoints_for("proc", "ds1") == []
    assert len(db.checkpoints_for("proc", "ds2")) == 1
    db.close()


def test_checksum_dataset_stable_and_sensitive_to_content():
    d1 = {"metadata": {"x": 1}, "files": {"a.root": 10}}
    d2 = {"files": {"a.root": 10}, "metadata": {"x": 1}}  # different key order
    d3 = {"metadata": {"x": 2}, "files": {"a.root": 10}}

    assert checksum_dataset(d1) == checksum_dataset(d2)
    assert checksum_dataset(d1) != checksum_dataset(d3)
