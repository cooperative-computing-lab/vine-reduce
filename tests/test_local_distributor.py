from __future__ import annotations

import os

import pytest

from vine_reduce import serialization
from vine_reduce.defaults import default_chunk_to_args, executor_wrapper
from vine_reduce.executor import simple_executor
from vine_reduce.local_distributor import LocalDistributor
from vine_reduce.types import Chunk, ResultHandle, Success

from helpers import count_events, read_env_var


@pytest.fixture
def distributor(tmp_path):
    dist = LocalDistributor(
        max_workers=2,
        work_dir=str(tmp_path / "cluster"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
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


def test_checkpoint_result_is_written_to_checkpoint_dir_not_work_dir(distributor, tmp_path):
    result_id = _submit_chunk(distributor, 1, Chunk("a.root", 0, 5))
    checkpoint_id = distributor.submit(
        1,
        "test:process",
        "processor",
        executor_wrapper,
        count_events,
        Chunk("a.root", 0, 5),
        {},
        None,
        None,
        default_chunk_to_args,
        simple_executor,
        is_checkpoint=True,
    )

    outcomes = {}
    for _ in range(2):
        outcome = distributor.wait(timeout=30)
        outcomes[outcome.result_id] = outcome

    assert outcomes[result_id].file.startswith(str(tmp_path / "cluster") + os.sep)
    assert outcomes[checkpoint_id].file.startswith(str(tmp_path / "checkpoints") + os.sep)


def test_shutdown_leaves_checkpoint_dir_in_place(tmp_path):
    """A checkpoint has to survive this process ending, so it can be read back
    on restart - unlike an owned work_dir, which is disposable scratch space
    shutdown() removes (see test_shutdown_removes_owned_work_dir)."""
    dist = LocalDistributor(max_workers=2, checkpoint_dir=str(tmp_path / "checkpoints"))
    checkpoint_id = dist.submit(
        1,
        "test:process",
        "processor",
        executor_wrapper,
        count_events,
        Chunk("a.root", 0, 5),
        {},
        None,
        None,
        default_chunk_to_args,
        simple_executor,
        is_checkpoint=True,
    )
    outcome = dist.wait(timeout=30)
    checkpoint_path = dist.checkpoint_path(checkpoint_id)

    dist.shutdown()

    assert os.path.exists(checkpoint_path)
    assert outcome.file == checkpoint_path


def test_shutdown_removes_owned_work_dir(tmp_path):
    dist = LocalDistributor(max_workers=2, checkpoint_dir=str(tmp_path / "checkpoints"))
    work_dir = dist._work_dir
    _submit_chunk(dist, 1, Chunk("a.root", 0, 5))
    dist.wait(timeout=30)

    dist.shutdown()

    assert not os.path.exists(work_dir)


def test_adopt_checkpoint_returns_a_usable_handle(distributor, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    seeded_path = str(checkpoint_dir / "seeded.pkl.zst")
    serialization.dump(42, seeded_path)  # stands in for a prior run's checkpoint

    handle = distributor.adopt_checkpoint(seeded_path)

    assert isinstance(handle, ResultHandle)
    assert handle.file == seeded_path
    assert distributor.checkpoint_path(handle.result_id) == seeded_path

    dest = tmp_path / "copy.pkl.zst"
    distributor.retrieve(handle.result_id, str(dest))
    assert serialization.load(str(dest)) == 42


def test_adopt_checkpoint_release_result_removes_the_adopted_file(distributor, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    seeded_path = str(checkpoint_dir / "seeded.pkl.zst")
    serialization.dump(42, seeded_path)

    handle = distributor.adopt_checkpoint(seeded_path)
    distributor.release_result(handle.result_id)

    assert not os.path.exists(seeded_path)


def test_checkpoint_filenames_never_collide_with_a_leftover_from_a_prior_run(tmp_path):
    """§2.7: the on-disk filename for a non-final checkpoint must be unique
    for the lifetime of checkpoint_dir, not just for this process's
    result_id counter - otherwise a fresh distributor's counter restarting
    at 1 could mint a filename that collides with (and silently overwrites)
    a still-live checkpoint from an earlier run at that same path."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    # A pre-fix, result_id-based naming scheme would have named the first
    # checkpoint of a fresh process "1.pkl.zst" - plant exactly that,
    # holding content distinguishable from anything this test submits.
    leftover_path = checkpoint_dir / "1.pkl.zst"
    serialization.dump("leftover-from-a-prior-run", str(leftover_path))

    dist = LocalDistributor(max_workers=2, checkpoint_dir=str(checkpoint_dir))
    try:
        checkpoint_paths = []
        for _ in range(2):
            dist.submit(
                1,
                "test:process",
                "processor",
                executor_wrapper,
                count_events,
                Chunk("a.root", 0, 5),
                {},
                None,
                None,
                default_chunk_to_args,
                simple_executor,
                is_checkpoint=True,
            )
            outcome = dist.wait(timeout=30)
            checkpoint_paths.append(outcome.file)
    finally:
        dist.shutdown()

    assert str(leftover_path) not in checkpoint_paths
    assert len(set(checkpoint_paths)) == 2
    assert serialization.load(str(leftover_path)) == "leftover-from-a-prior-run"


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
