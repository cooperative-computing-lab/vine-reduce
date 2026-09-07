from __future__ import annotations

import os
import shutil
import time
from uuid import uuid4

import pytest

vine = pytest.importorskip("ndcctools.taskvine")

from vine_reduce import VineReduce, serialization  # noqa: E402
from vine_reduce.defaults import (  # noqa: E402
    default_chunk_to_args,
    executor_wrapper,
    reducer_wrapper,
)
from vine_reduce.executor import SimpleExecutor  # noqa: E402
from vine_reduce.taskvine_distributor import (  # noqa: E402
    TaskVineDistributor,
    _InFlight,
    _result_token,
)
from vine_reduce.types import Chunk, RuntimeFailure, Success  # noqa: E402

from helpers import (  # noqa: E402
    count_events,
    failing_processor,
    read_env_var,
    read_shipped_file,
    submit_chunk,
    sum_reducer,
)

pytestmark = pytest.mark.skipif(
    shutil.which("vine_factory") is None, reason="vine_factory not on PATH"
)

WAIT_TIMEOUT = 30  # generous, to absorb the worker's first-connect latency


def _wait(distributor, timeout=WAIT_TIMEOUT):
    """distributor.wait(t) can return None well before t elapses even
    though a submitted task is still legitimately in flight - dispatch/
    completion isn't synchronous with the call. So, mirroring TaskVine's
    own idiom for draining a manager (`while not m.empty(): t =
    m.wait()`), keep polling as long as there's something outstanding,
    instead of treating one None as "nothing happened"."""
    deadline = time.monotonic() + timeout
    while not distributor._manager.empty():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        outcome = distributor.wait(timeout=remaining)
        if outcome is not None:
            return outcome
    return None


@pytest.fixture(autouse=True)
def _pythonpath(monkeypatch):
    """cloudpickle pickles tests/helpers.py's functions by reference, so a
    worker subprocess needs tests/ on its own PYTHONPATH to import them when
    unpickling - see the comment this used to carry on every test."""
    monkeypatch.setenv("PYTHONPATH", os.path.dirname(__file__))


@pytest.fixture
def dist(tmp_path):
    """A TaskVineDistributor with the standard single-core processor/reducer
    resources and a fresh checkpoint_dir - the config every test uses unless
    it specifically needs something different (a caller-supplied manager, a
    non-default resource config, or to test shutdown() itself outside a
    `with` block)."""
    with TaskVineDistributor(
        port=0,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1},
        checkpoint_dir=str(tmp_path / "checkpoints"),
    ) as distributor:
        yield distributor


@pytest.fixture
def dist_with_workers(dist):
    """`dist`, plus one running vine_factory worker (cores=2) - for tests
    that actually submit and run a task, rather than just exercising
    dist's own bookkeeping."""
    workers = vine.Factory(manager=dist._manager)
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1
    workers.timeout = WAIT_TIMEOUT
    with workers:
        yield dist


def test_submit_and_wait_round_trip(dist_with_workers, tmp_path):
    # The worker is a real separate process, unlike LocalDistributor's forked
    # ProcessPoolExecutor workers, which inherit the test process's already-
    # imported modules for free. vine.Factory launches vine_factory (and,
    # transitively, vine_worker and the per-task Python subprocess it forks)
    # by inheriting this process's environment as-is, so PYTHONPATH set via
    # the _pythonpath fixture is enough - no need to route it through
    # Factory's own --env option.
    dist = dist_with_workers
    result_id = submit_chunk(dist, 1, Chunk("a.root", 0, 5))

    outcome = _wait(dist)

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id
    # outcome.file is an opaque token, not a readable path (see
    # taskvine_distributor.py's docstring) - retrieve() is how it's read.
    dest = tmp_path / "copy.pkl.zst"
    dist.retrieve(outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == 5


def test_wait_returns_none_when_nothing_pending(dist):
    # No task is ever submitted here, so no worker is needed - skip the
    # Factory entirely. cctools has a bug (see CCTOOLS_BUG.txt) where, for
    # batch_type="local", a worker can get stuck mid-SSL-handshake and then
    # Factory.stop() hangs forever trying to reap it; not spinning up a
    # worker at all is the workaround until that's fixed upstream.
    assert dist.wait(timeout=0.1) is None


def test_wait_survives_a_corrupted_task_output(monkeypatch, dist):
    """PythonTask.output hands back the exception object itself, not an
    Outcome, if cloudpickle.load of the task's output fails on the
    manager side (verified in cctools' task.py) - wait() must turn that
    into a RuntimeFailure rather than let dataclasses.replace(raw, ...) raise
    TypeError. A stub task via a monkeypatched
    wait_for_tag, no real worker needed - see test_wait_returns_none_when_
    nothing_pending for why that's safe here."""
    result_id = uuid4().hex
    dist._in_flight_by_taskvine_id[123] = _InFlight(result_id=result_id, kind="processor")

    class _FakeTask:
        id = 123
        result = "success"
        std_output = ""
        output = ValueError("simulated cloudpickle.load failure")
        resources_measured = None
        resources_allocated = None

        def successful(self):
            return True

    monkeypatch.setattr(dist._manager, "wait_for_tag", lambda tag, timeout: _FakeTask())

    outcome = dist.wait(timeout=1)

    assert isinstance(outcome, RuntimeFailure)
    assert outcome.result_id == result_id


def test_retrieve_copies_file(dist_with_workers, tmp_path):
    dist = dist_with_workers
    submit_chunk(dist, 1, Chunk("a.root", 0, 3))
    outcome = _wait(dist)

    dest = tmp_path / "copy.pkl.zst"
    dist.retrieve(outcome.result_id, str(dest))

    assert serialization.load(str(dest)) == 3


def test_release_result_allows_reuse(dist_with_workers):
    dist = dist_with_workers
    submit_chunk(dist, 1, Chunk("a.root", 0, 3))
    outcome = _wait(dist)

    dist.release_result(outcome.result_id)
    # release_result is fire-and-forget cleanup; the main guarantee is
    # that it doesn't raise, and that the distributor's own bookkeeping
    # is cleared.
    assert _result_token(outcome.result_id) not in dist._files_by_key


def test_ordinary_result_is_not_written_to_checkpoint_dir(dist_with_workers):
    """A result submitted without is_checkpoint=True must be an ordinary
    vine_temp() - it should never appear under checkpoint_dir, and
    checkpoint_path() (only meaningful for is_checkpoint=True results) must
    not know about it."""
    dist = dist_with_workers
    submit_chunk(dist, 1, Chunk("a.root", 0, 3))
    outcome = _wait(dist)

    assert _result_token(outcome.result_id) not in dist._checkpoint_paths_by_token
    assert os.listdir(dist._checkpoint_dir) == []


def test_checkpoint_result_is_durably_written_to_checkpoint_dir(dist_with_workers):
    """A result submitted with is_checkpoint=True must be a
    vine_file(cache=True) written under checkpoint_dir - readable straight
    off disk via checkpoint_path(), with no retrieve() call needed, since
    TaskVine already wrote it there as part of completing the task (see the
    module docstring)."""
    dist = dist_with_workers
    submit_chunk(dist, 1, Chunk("a.root", 0, 5), is_checkpoint=True)
    outcome = _wait(dist)

    path = dist.checkpoint_path(outcome.result_id)
    assert path.startswith(dist._checkpoint_dir + os.sep)
    assert serialization.load(path) == 5


def test_release_result_removes_checkpoint_file_from_disk(dist_with_workers):
    dist = dist_with_workers
    submit_chunk(dist, 1, Chunk("a.root", 0, 5), is_checkpoint=True)
    outcome = _wait(dist)
    path = dist.checkpoint_path(outcome.result_id)
    assert os.path.exists(path)

    dist.release_result(outcome.result_id)

    assert not os.path.exists(path)
    assert _result_token(outcome.result_id) not in dist._checkpoint_paths_by_token


def test_failed_task_reports_real_traceback_not_output_missing(dist_with_workers):
    """A processor that raises must come back as the wrapper's own
    RuntimeFailure, with the real traceback - not as a generic "output
    missing" RuntimeFailure with no traceback, which is what happens if the
    declared dest_file output isn't produced on failure (see defaults.py's
    _run_and_wrap)."""
    dist = dist_with_workers
    result_id = uuid4().hex
    dist.submit(
        result_id,
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
        SimpleExecutor(),
    )

    outcome = _wait(dist)

    assert isinstance(outcome, RuntimeFailure)
    assert outcome.result_id == result_id
    assert "ValueError: boom" in outcome.traceback
    # The placeholder dest_file must not leak once its outcome is consumed.
    assert _result_token(result_id) not in dist._files_by_key


def test_capacity_reports_a_non_negative_capacity(dist):
    # No task is ever submitted here, so no worker is needed - skip the
    # Factory entirely (see test_wait_returns_none_when_nothing_pending and
    # CCTOOLS_BUG.txt).
    assert dist.capacity() >= 0


def test_constructor_reuses_a_pre_built_manager(tmp_path):
    # A caller-built manager (e.g. vine.DaskVine, so coffea's own
    # dataset_tools.preprocess() and this distributor share one manager/port/
    # worker pool - see PLAN.md) must be used as-is, not replaced by a second
    # one built from port/name.
    manager = vine.Manager(port=0, ssl=True)

    with TaskVineDistributor(
        manager=manager,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1},
        checkpoint_dir=str(tmp_path / "checkpoints"),
    ) as dist:
        assert dist._manager is manager

        workers = vine.Factory(manager=manager)
        workers.cores = 2
        workers.min_workers = 1
        workers.max_workers = 1
        workers.timeout = WAIT_TIMEOUT
        with workers:
            result_id = submit_chunk(dist, 1, Chunk("a.root", 0, 4))
            outcome = _wait(dist)

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id


def test_wait_ignores_tasks_submitted_directly_to_a_shared_manager(tmp_path):
    # On a caller-supplied manager= (see test_constructor_reuses_a_pre_built_manager),
    # the caller may submit its own tasks straight to that manager (e.g.
    # coffea's own dataset_tools.preprocess()). wait() must only ever
    # surface tasks *this* distributor submitted - via the tag set on every
    # task in submit() - or a foreign task finishing first would make
    # wait() pop an unknown taskvine id from _in_flight_by_taskvine_id and
    # KeyError.
    manager = vine.Manager(port=0, ssl=True)

    with TaskVineDistributor(
        manager=manager,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1},
        checkpoint_dir=str(tmp_path / "checkpoints"),
    ) as dist:
        workers = vine.Factory(manager=manager)
        workers.cores = 2
        workers.min_workers = 1
        workers.max_workers = 1
        workers.timeout = WAIT_TIMEOUT
        with workers:
            foreign_task = vine.PythonTask(count_events, Chunk("a.root", 0, 1))
            manager.submit(foreign_task)

            result_id = submit_chunk(dist, 1, Chunk("a.root", 0, 4))
            outcome = _wait(dist)

            assert isinstance(outcome, Success)
            assert outcome.result_id == result_id

            # The foreign task is still the caller's to reap - drain it
            # directly off the manager so the `with workers` block above
            # doesn't tear down while it's still in flight.
            manager.wait_for_task_id(foreign_task.id, WAIT_TIMEOUT)


def test_add_file_ships_file_to_every_task_sandbox(dist_with_workers, tmp_path):
    # add_file places the file under its basename in the task's own sandbox,
    # so the processor must open it by that relative name (see
    # helpers.read_shipped_file), not by its local path.
    dist = dist_with_workers
    shipped = tmp_path / "shipped.txt"
    shipped.write_text("hello from add_file")
    dist.add_file(str(shipped))

    result_id = uuid4().hex
    dist.submit(
        result_id,
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
        SimpleExecutor(),
    )
    outcome = _wait(dist)

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id
    dest = tmp_path / "copy.pkl.zst"
    dist.retrieve(outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == "hello from add_file"


def test_set_env_var_is_visible_to_every_task(dist_with_workers, tmp_path):
    dist = dist_with_workers
    dist.set_env_var("VINE_REDUCE_TEST_VAR", "abc123")

    result_id = uuid4().hex
    dist.submit(
        result_id,
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
        SimpleExecutor(),
    )
    outcome = _wait(dist)

    assert isinstance(outcome, Success)
    assert outcome.result_id == result_id
    dest = tmp_path / "copy.pkl.zst"
    dist.retrieve(outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == "abc123"


def test_reduction_chains_across_two_tasks(dist_with_workers, tmp_path):
    """The core file-passing bridge: a reduction task's input_files list
    contains tokens minted by earlier Success outcomes, not real paths -
    _remap_files must turn those into real task inputs."""
    dist = dist_with_workers
    id_a = submit_chunk(dist, 1, Chunk("a.root", 0, 3))
    id_b = submit_chunk(dist, 1, Chunk("a.root", 3, 8))

    outcomes = {}
    for _ in range(2):
        outcome = _wait(dist)
        outcomes[outcome.result_id] = outcome

    file_a, file_b = outcomes[id_a].file, outcomes[id_b].file

    reduce_id = uuid4().hex
    dist.submit(
        reduce_id,
        10,
        "test:reduce",
        "reducer",
        reducer_wrapper,
        sum_reducer,
        [file_a, file_b],
        True,
        None,
    )
    reduce_outcome = _wait(dist)

    assert isinstance(reduce_outcome, Success)
    assert reduce_outcome.result_id == reduce_id
    dest = tmp_path / "reduced.pkl.zst"
    dist.retrieve(reduce_outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == 3 + 5


def test_adopt_checkpoint_flows_through_remap_release_and_retrieve(dist_with_workers, tmp_path):
    """A seeded checkpoint adopted via adopt_checkpoint must behave exactly
    like a this-run checkpoint from then on: its file token remaps/declares
    as a task input, the reduction using it succeeds, and release_result
    undeclares it and removes it from disk - no special-casing needed."""
    dist = dist_with_workers
    seeded_path = str(tmp_path / "checkpoints" / "seeded.p")
    os.makedirs(os.path.dirname(seeded_path), exist_ok=True)
    serialization.dump(100, seeded_path)  # stands in for a prior run's checkpoint

    adopted_id = uuid4().hex
    file = dist.adopt_checkpoint(adopted_id, seeded_path)
    assert file in dist._files_by_key

    submit_chunk(dist, 1, Chunk("b.root", 0, 3))
    outcome_b = _wait(dist)

    reduce_id = uuid4().hex
    dist.submit(
        reduce_id,
        10,
        "test:reduce",
        "reducer",
        reducer_wrapper,
        sum_reducer,
        [file, outcome_b.file],
        True,
        None,
    )
    reduce_outcome = _wait(dist)

    assert isinstance(reduce_outcome, Success)
    assert reduce_outcome.result_id == reduce_id
    dest = tmp_path / "reduced.pkl.zst"
    dist.retrieve(reduce_outcome.result_id, str(dest))
    assert serialization.load(str(dest)) == 100 + 3

    dist.release_result(adopted_id)
    assert file not in dist._files_by_key
    assert not os.path.exists(seeded_path)


def test_checkpoint_filenames_never_collide_across_restarts(tmp_path):
    """§2.7: a fresh distributor instance must not be able to mint an
    on-disk checkpoint filename that collides with one still-live from an
    earlier instance/run against the same checkpoint_dir."""
    checkpoint_dir = str(tmp_path / "checkpoints")

    def _write_one_checkpoint():
        with TaskVineDistributor(
            port=0,
            resources_processor={"cores": 1},
            checkpoint_dir=checkpoint_dir,
        ) as dist:
            workers = vine.Factory(manager=dist._manager)
            workers.cores = 2
            workers.min_workers = 1
            workers.max_workers = 1
            workers.timeout = WAIT_TIMEOUT
            with workers:
                submit_chunk(dist, 1, Chunk("a.root", 0, 5), is_checkpoint=True)
                outcome = _wait(dist)
                path = dist.checkpoint_path(outcome.result_id)
        return path

    path_1 = _write_one_checkpoint()
    path_2 = _write_one_checkpoint()

    assert path_1 != path_2
    assert os.path.exists(path_1)
    assert os.path.exists(path_2)


def test_shutdown_frees_a_self_built_manager(tmp_path):
    dist = TaskVineDistributor(port=0, checkpoint_dir=str(tmp_path / "checkpoints"))

    dist.shutdown()

    assert dist._manager is None


def test_shutdown_leaves_a_caller_supplied_manager_alone(tmp_path):
    manager = vine.Manager(port=0)
    dist = TaskVineDistributor(manager=manager, checkpoint_dir=str(tmp_path / "checkpoints"))

    dist.shutdown()

    assert dist._manager is manager


def test_engine_end_to_end_via_taskvine(dist_with_workers, tmp_path, dataset_input):
    """The distributor in isolation only proves submit/wait/retrieve work;
    this drives it through the real VineReduce pipeline (chunking, pooled
    reduction across two files, checkpointing) the way a user actually would."""
    dist = dist_with_workers
    input_path = dataset_input(
        {"numbers": {"metadata": {}, "files": {"a.root": 7, "b.root": 3}}}
    )

    vr = VineReduce(
        processors={"count": count_events},
        input=input_path,
        reducer=sum_reducer,
        results_dir=str(tmp_path / "results"),
        distributor=dist,
    )
    vr.compute()

    dataset_dir = os.path.join(vr.results_dir, "numbers", "count")
    files = os.listdir(dataset_dir)
    assert len(files) == 1
    assert serialization.load(os.path.join(dataset_dir, files[0])) == 10
