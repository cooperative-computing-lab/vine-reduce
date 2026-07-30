from __future__ import annotations

import pytest

from vine_reduce import serialization
from vine_reduce.defaults import default_chunk_to_args, default_executor, executor_wrapper
from vine_reduce.local_distributor import LocalDistributor
from vine_reduce.types import Chunk, Success

from helpers import count_events


@pytest.fixture
def distributor(tmp_path):
    dist = LocalDistributor(max_workers=2, work_dir=str(tmp_path / "cluster"))
    yield dist
    dist.shutdown()


def _submit_chunk(distributor, priority, chunk):
    return distributor.submit(
        priority,
        "test:process",
        "processor",
        executor_wrapper,
        count_events,
        chunk,
        {},
        None,
        None,
        default_chunk_to_args,
        default_executor,
    )


def test_submit_and_wait_round_trip(distributor):
    result_id = _submit_chunk(distributor, 1, Chunk("a.root", 0, 5))

    outcome = distributor.wait(timeout=30)

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id
    assert serialization.load(outcome.file) == 5


def test_wait_returns_none_when_nothing_pending(distributor):
    assert distributor.wait(timeout=0.1) is None


def test_retrieve_copies_file(distributor, tmp_path):
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 3))
    outcome = distributor.wait(timeout=30)

    dest = tmp_path / "copy.pkl.zst"
    distributor.retrieve(outcome.result_id, str(dest))

    assert serialization.load(str(dest)) == 3


def test_free_result_removes_file(distributor):
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 3))
    outcome = distributor.wait(timeout=30)

    distributor.free_result(outcome.result_id)

    import os

    assert not os.path.exists(outcome.file)


def test_hungry_reports_available_capacity(distributor):
    # 2 workers -> target queue depth of 4, nothing in flight yet
    assert distributor.hungry() == 4
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 100000))
    assert distributor.hungry() == 3
