from __future__ import annotations

import json
import os

from vine_reduce.size_log import SizeLog, SizeRecord


def _record(**overrides):
    defaults = dict(
        dataset_name="ds",
        processor_name="proc",
        chunk_size=1000,
        reduction_size=10,
        processing={"cores": 1.0, "memory": 512.0},
        reduction={"cores": 1.0, "memory": 256.0},
    )
    defaults.update(overrides)
    return SizeRecord(**defaults)


def test_size_log_creates_file_lazily(tmp_path):
    path = str(tmp_path / "size.jsonl")
    log = SizeLog(path)
    assert not os.path.exists(path)

    log.log(_record())
    assert os.path.exists(path)


def test_size_log_appends_one_json_line_per_record_in_order(tmp_path):
    path = str(tmp_path / "size.jsonl")
    log = SizeLog(path)

    log.log(_record(dataset_name="ds1"))
    log.log(_record(dataset_name="ds2", processing=None, reduction=None))

    lines = open(path).read().splitlines()
    assert len(lines) == 2

    row1 = json.loads(lines[0])
    assert row1 == {
        "dataset_name": "ds1",
        "processor_name": "proc",
        "chunk_size": 1000,
        "reduction_size": 10,
        "processing": {"cores": 1.0, "memory": 512.0},
        "reduction": {"cores": 1.0, "memory": 256.0},
    }

    row2 = json.loads(lines[1])
    assert row2["dataset_name"] == "ds2"
    assert row2["processing"] is None
    assert row2["reduction"] is None


def test_size_log_handles_null_chunk_size(tmp_path):
    """chunksize=None (one chunk per file) must round-trip as JSON null, not
    be dropped or coerced."""
    path = str(tmp_path / "size.jsonl")
    log = SizeLog(path)

    log.log(_record(chunk_size=None))

    row = json.loads(open(path).read())
    assert row["chunk_size"] is None
