from __future__ import annotations

import pytest

from vine_reduce import serialization
from vine_reduce.defaults import default_chunk_to_args, executor_wrapper
from vine_reduce.executor import simple_executor
from vine_reduce.local_distributor import LocalDistributor
from vine_reduce.types import Chunk, Success

from helpers import count_events, read_env_var


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
        simple_executor,
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


def test_release_result_removes_file(distributor):
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 3))
    outcome = distributor.wait(timeout=30)

    distributor.release_result(outcome.result_id)

    import os

    assert not os.path.exists(outcome.file)


def test_capacity_reports_available_capacity(distributor):
    # 2 workers -> target queue depth of 4, nothing in flight yet
    assert distributor.capacity() == 4
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 100000))
    assert distributor.capacity() == 3


def test_add_file_is_a_no_op_that_does_not_raise(distributor, tmp_path):
    # Worker subprocesses already share vine_reduce's filesystem, so there's
    # nothing to ship - this only guards against add_file raising.
    shipped = tmp_path / "shipped.txt"
    shipped.write_text("hi")
    distributor.add_file(str(shipped))


def test_set_env_var_is_visible_to_submitted_calls(distributor):
    distributor.set_env_var("VINE_REDUCE_TEST_VAR", "xyz")

    result_id = distributor.submit(
        1,
        "test:process",
        "processor",
        executor_wrapper,
        read_env_var,
        Chunk("a.root", 0, 1),
        {},
        None,
        None,
        default_chunk_to_args,
        simple_executor,
    )
    outcome = distributor.wait(timeout=30)

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id
    assert serialization.load(outcome.file) == "xyz"
