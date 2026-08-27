from __future__ import annotations

import os
import shutil

import ndcctools.taskvine as vine
import pytest

from vine_reduce import VineReduce, serialization
from vine_reduce.defaults import (
    default_chunk_to_args,
    executor_wrapper,
    reducer_wrapper,
)
from vine_reduce.executor import simple_executor
from vine_reduce.taskvine_distributor import TaskVineDistributor, _result_token
from vine_reduce.types import Chunk, RuntimeFailure, Success

from helpers import count_events, failing_processor, read_env_var, read_shipped_file, sum_reducer

pytestmark = pytest.mark.skipif(
    shutil.which("vine_factory") is None, reason="vine_factory not on PATH"
)

WAIT_TIMEOUT = 30  # generous, to absorb the worker's first-connect latency


@pytest.fixture
def distributor(monkeypatch, tmp_path):
    # The worker is a real separate process, unlike LocalDistributor's forked
    # ProcessPoolExecutor workers, which inherit the test process's already-
    # imported modules for free. cloudpickle pickles tests/helpers.py's
    # functions by reference, so the worker needs tests/ on its own
    # PYTHONPATH to import them when unpickling. vine.Factory launches
    # vine_factory (and, transitively, vine_worker and the per-task Python
    # subprocess it forks) by inheriting this process's environment as-is,
    # so setting PYTHONPATH here is enough - no need to route it through
    # Factory's own --env option.
    monkeypatch.setenv("PYTHONPATH", os.path.dirname(__file__))

    dist = TaskVineDistributor(
        port=0,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1},
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    workers = vine.Factory(manager=dist._manager)
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1
    workers.timeout = WAIT_TIMEOUT
    with workers:
        yield dist
    dist.shutdown()


def _submit_chunk(distributor, priority, chunk, is_checkpoint=False):
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
        is_checkpoint=is_checkpoint,
    )


def test_submit_and_wait_round_trip(distributor, tmp_path):
    result_id = _submit_chunk(distributor, 1, Chunk("a.root", 0, 5))

    outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id
    # outcome.file is an opaque token, not a readable path (see
    # taskvine_distributor.py's docstring) - retrieve() is how it's read.
    dest = tmp_path / "copy.pkl.zst"
    distributor.retrieve(outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == 5


def test_wait_returns_none_when_nothing_pending(distributor):
    assert distributor.wait(timeout=0.1) is None


def test_retrieve_copies_file(distributor, tmp_path):
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 3))
    outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    dest = tmp_path / "copy.pkl.zst"
    distributor.retrieve(outcome.result_id, str(dest))

    assert serialization.load(str(dest)) == 3


def test_release_result_allows_reuse(distributor):
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 3))
    outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    distributor.release_result(outcome.result_id)
    # release_result is fire-and-forget cleanup; the main guarantee is that
    # it doesn't raise, and that the distributor's own bookkeeping is cleared.
    assert _result_token(outcome.result_id) not in distributor._files_by_key


def test_ordinary_result_is_not_written_to_checkpoint_dir(distributor):
    """A result submitted without is_checkpoint=True must be an ordinary
    vine_temp() - it should never appear under checkpoint_dir, and
    checkpoint_path() (only meaningful for is_checkpoint=True results) must
    not know about it."""
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 3))
    outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    assert _result_token(outcome.result_id) not in distributor._checkpoint_paths_by_token
    assert os.listdir(distributor._checkpoint_dir) == []


def test_checkpoint_result_is_durably_written_to_checkpoint_dir(distributor):
    """A result submitted with is_checkpoint=True must be a
    vine_file(cache=True) written under checkpoint_dir - readable straight
    off disk via checkpoint_path(), with no retrieve() call needed, since
    TaskVine already wrote it there as part of completing the task (see the
    module docstring)."""
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 5), is_checkpoint=True)
    outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    path = distributor.checkpoint_path(outcome.result_id)
    assert path.startswith(distributor._checkpoint_dir + os.sep)
    assert serialization.load(path) == 5


def test_release_result_removes_checkpoint_file_from_disk(distributor):
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 5), is_checkpoint=True)
    outcome = distributor.wait(timeout=WAIT_TIMEOUT)
    path = distributor.checkpoint_path(outcome.result_id)
    assert os.path.exists(path)

    distributor.release_result(outcome.result_id)

    assert not os.path.exists(path)
    assert _result_token(outcome.result_id) not in distributor._checkpoint_paths_by_token


def test_failed_task_reports_real_traceback_not_output_missing(distributor):
    """A processor that raises must come back as the wrapper's own
    RuntimeFailure, with the real traceback - not as a generic "output
    missing" RuntimeFailure with no traceback, which is what happens if the
    declared dest_file output isn't produced on failure (see defaults.py's
    _run_and_wrap)."""
    result_id = distributor.submit(
        1,
        "test:process",
        "processor",
        executor_wrapper,
        failing_processor,
        Chunk("a.root", 0, 5),
        {},
        None,
        None,
        default_chunk_to_args,
        simple_executor,
    )

    outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    assert isinstance(outcome, RuntimeFailure)
    assert outcome.result_id == result_id
    assert "ValueError: boom" in outcome.traceback
    # The placeholder dest_file must not leak once its outcome is consumed.
    assert _result_token(result_id) not in distributor._files_by_key


def test_capacity_reports_a_non_negative_capacity(distributor):
    assert distributor.capacity() >= 0


def test_constructor_reuses_a_pre_built_manager(monkeypatch, tmp_path):
    # A caller-built manager (e.g. vine.DaskVine, so coffea's own
    # dataset_tools.preprocess() and this distributor share one manager/port/
    # worker pool - see PLAN.md) must be used as-is, not replaced by a second
    # one built from port/name.
    monkeypatch.setenv("PYTHONPATH", os.path.dirname(__file__))
    manager = vine.Manager(port=0)

    dist = TaskVineDistributor(
        manager=manager,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1},
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    assert dist._manager is manager

    workers = vine.Factory(manager=manager)
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1
    workers.timeout = WAIT_TIMEOUT
    with workers:
        result_id = _submit_chunk(dist, 1, Chunk("a.root", 0, 4))
        outcome = dist.wait(timeout=WAIT_TIMEOUT)
    dist.shutdown()

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id


def test_add_file_ships_file_to_every_task_sandbox(distributor, tmp_path):
    # add_file places the file under its basename in the task's own sandbox,
    # so the processor must open it by that relative name (see
    # helpers.read_shipped_file), not by its local path.
    shipped = tmp_path / "shipped.txt"
    shipped.write_text("hello from add_file")
    distributor.add_file(str(shipped))

    result_id = distributor.submit(
        1,
        "test:process",
        "processor",
        executor_wrapper,
        read_shipped_file,
        Chunk("a.root", 0, 1),
        {},
        None,
        None,
        default_chunk_to_args,
        simple_executor,
    )
    outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id
    dest = tmp_path / "copy.pkl.zst"
    distributor.retrieve(outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == "hello from add_file"


def test_set_env_var_is_visible_to_every_task(distributor, tmp_path):
    distributor.set_env_var("VINE_REDUCE_TEST_VAR", "abc123")

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
    outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id
    dest = tmp_path / "copy.pkl.zst"
    distributor.retrieve(outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == "abc123"


def test_reduction_chains_across_two_tasks(distributor, tmp_path):
    """The core file-passing bridge: a reduction task's input_files list
    contains tokens minted by earlier Success outcomes, not real paths -
    _remap_files must turn those into real task inputs."""
    id_a = _submit_chunk(distributor, 1, Chunk("a.root", 0, 3))
    id_b = _submit_chunk(distributor, 1, Chunk("a.root", 3, 8))

    outcomes = {}
    for _ in range(2):
        outcome = distributor.wait(timeout=WAIT_TIMEOUT)
        outcomes[outcome.result_id] = outcome

    file_a, file_b = outcomes[id_a].file, outcomes[id_b].file

    reduce_id = distributor.submit(
        10,
        "test:reduce",
        "reducer",
        reducer_wrapper,
        sum_reducer,
        [file_a, file_b],
        True,
        None,
    )
    reduce_outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    assert isinstance(reduce_outcome, Success)
    assert reduce_outcome.result_id == reduce_id
    dest = tmp_path / "reduced.pkl.zst"
    distributor.retrieve(reduce_outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == 3 + 5


def test_restart_seeded_checkpoint_path_is_declared_as_task_input(distributor, tmp_path):
    """On restart, Pipeline._pool_item_from_checkpoint seeds a pooled item's
    `file` with the checkpoint's real on-disk path, not a token this class
    ever minted (there was no live run to mint one in) - _remap_files must
    recognize that case too, not just a known token."""
    seeded_path = str(tmp_path / "checkpoints" / "seeded.p")
    os.makedirs(os.path.dirname(seeded_path), exist_ok=True)
    serialization.dump(100, seeded_path)  # stands in for a prior run's checkpoint

    _submit_chunk(distributor, 1, Chunk("b.root", 0, 3))
    outcome_b = distributor.wait(timeout=WAIT_TIMEOUT)

    reduce_id = distributor.submit(
        10,
        "test:reduce",
        "reducer",
        reducer_wrapper,
        sum_reducer,
        [seeded_path, outcome_b.file],
        True,
        None,
    )
    reduce_outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    assert isinstance(reduce_outcome, Success)
    assert reduce_outcome.result_id == reduce_id
    dest = tmp_path / "reduced.pkl.zst"
    distributor.retrieve(reduce_outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == 100 + 3


def test_restart_seeded_checkpoint_path_is_cached_and_released(distributor, tmp_path):
    """A restart-seeded path is declared once and cached under its own path
    in _files_by_key - the same dict dest_tokens live in - rather than
    re-declared every time it's used, so a retried resubmission of the same
    group (e.g. after a ResourceExhaustion retry) doesn't leak a fresh
    vine.File each time. release_path() drops the cache entry once Pipeline
    has removed the checkpoint from disk (see pipeline.py's _checkpoint)."""
    seeded_path = str(tmp_path / "checkpoints" / "seeded.p")
    os.makedirs(os.path.dirname(seeded_path), exist_ok=True)
    serialization.dump(100, seeded_path)  # stands in for a prior run's checkpoint

    def _reduce_with_seeded_path():
        _submit_chunk(distributor, 1, Chunk("b.root", 0, 3))
        outcome_b = distributor.wait(timeout=WAIT_TIMEOUT)
        distributor.submit(
            10,
            "test:reduce",
            "reducer",
            reducer_wrapper,
            sum_reducer,
            [seeded_path, outcome_b.file],
            True,
            None,
        )
        distributor.wait(timeout=WAIT_TIMEOUT)

    _reduce_with_seeded_path()
    assert seeded_path in distributor._files_by_key
    cached_file = distributor._files_by_key[seeded_path]

    _reduce_with_seeded_path()
    assert distributor._files_by_key[seeded_path] is cached_file

    distributor.release_path(seeded_path)
    assert seeded_path not in distributor._files_by_key


def test_shutdown_frees_a_self_built_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", os.path.dirname(__file__))
    dist = TaskVineDistributor(port=0, checkpoint_dir=str(tmp_path / "checkpoints"))

    dist.shutdown()

    assert dist._manager is None


def test_shutdown_leaves_a_caller_supplied_manager_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", os.path.dirname(__file__))
    manager = vine.Manager(port=0)
    dist = TaskVineDistributor(manager=manager, checkpoint_dir=str(tmp_path / "checkpoints"))

    dist.shutdown()

    assert dist._manager is manager


def test_engine_end_to_end_via_taskvine(tmp_path, dataset_input, distributor):
    """The distributor in isolation only proves submit/wait/retrieve work;
    this drives it through the real VineReduce pipeline (chunking, pooled
    reduction across two files, checkpointing) the way a user actually would."""
    input_path = dataset_input({"numbers": {"metadata": {}, "files": {"a.root": 7, "b.root": 3}}})

    vr = VineReduce(
        processors={"count": count_events},
        input=input_path,
        reducer=sum_reducer,
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    vr.compute()

    dataset_dir = os.path.join(vr.results_dir, "numbers", "count")
    files = os.listdir(dataset_dir)
    assert len(files) == 1
    assert serialization.load(os.path.join(dataset_dir, files[0])) == 10
