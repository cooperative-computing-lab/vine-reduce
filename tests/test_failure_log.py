from __future__ import annotations

import os

from vine_reduce.failure_log import FailureLog, FailureRecord
from vine_reduce.types import ResourceUsage


def test_failure_log_creates_file_lazily(tmp_path):
    path = str(tmp_path / "failed_files.log")
    log = FailureLog(path)
    assert not os.path.exists(path)

    log.log(
        FailureRecord(
            dataset_name="ds",
            filename="a.root",
            ranges=((0, 5),),
            kind="processor",
            attempts=1,
            resources_allocated=None,
            resources_measured=None,
            traceback=None,
        )
    )
    assert os.path.exists(path)


def test_failure_log_appends_readable_blocks_in_order(tmp_path):
    path = str(tmp_path / "failed_files.log")
    log = FailureLog(path)

    log.log(
        FailureRecord(
            dataset_name="ds1",
            filename="a.root",
            ranges=((0, 5),),
            kind="processor",
            attempts=3,
            resources_allocated={"cores": 1},
            resources_measured=ResourceUsage(cores=1, memory_mb=512.0, wall_time_s=1.5),
            traceback="Traceback (most recent call last):\nValueError: boom",
        )
    )
    log.log(
        FailureRecord(
            dataset_name="ds1",
            filename="b.root",
            ranges=((0, 2), (4, 6)),
            kind="reducer",
            attempts=1,
            resources_allocated=None,
            resources_measured=None,
            traceback=None,
        )
    )

    contents = open(path).read()
    for expected in (
        "ds1",
        "a.root",
        "b.root",
        "[0:5]",
        "[0:2] [4:6]",
        "processor",
        "reducer",
        "ValueError: boom",
    ):
        assert expected in contents
    # each record is its own block, appended in the order logged
    assert contents.index("a.root") < contents.index("b.root")
