from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from vine_reduce import VineReduce, serialization
from vine_reduce.defaults import (
    default_chunk_to_args,
    default_executor,
    executor_wrapper,
    reducer_wrapper,
)
from vine_reduce.taskvine_distributor import TaskVineDistributor
from vine_reduce.types import Chunk, Success

from helpers import count_events, sum_reducer

pytestmark = pytest.mark.skipif(
    shutil.which("vine_worker") is None, reason="vine_worker not on PATH"
)

WAIT_TIMEOUT = 30  # generous, to absorb the worker's first-connect latency


@pytest.fixture
def distributor():
    dist = TaskVineDistributor(port=0, resources_processor={"cores": 1}, resources_reducer={"cores": 1})
    # The worker is a real separate process, unlike LocalDistributor's forked
    # ProcessPoolExecutor workers, which inherit the test process's already-
    # imported modules for free. cloudpickle pickles tests/helpers.py's
    # functions by reference, so the worker needs tests/ on its own
    # PYTHONPATH to import them when unpickling.
    env = dict(os.environ, PYTHONPATH=os.path.dirname(__file__))
    worker = subprocess.Popen(
        ["vine_worker", "--cores", "2", "-t", str(WAIT_TIMEOUT), "localhost", str(dist.port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield dist
    finally:
        worker.terminate()
        worker.wait(timeout=10)
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


def test_free_result_allows_reuse(distributor):
    _submit_chunk(distributor, 1, Chunk("a.root", 0, 3))
    outcome = distributor.wait(timeout=WAIT_TIMEOUT)

    distributor.free_result(outcome.result_id)
    # free_result is fire-and-forget cleanup; the main guarantee is that it
    # doesn't raise, and that the distributor's own bookkeeping is cleared.
    assert outcome.result_id not in distributor._token_by_result_id


def test_hungry_reports_a_non_negative_capacity(distributor):
    assert distributor.hungry() >= 0


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


def test_engine_end_to_end_via_taskvine(tmp_path, dataset_input, distributor):
    """The distributor in isolation only proves submit/wait/retrieve work;
    this drives it through the real VineReduce pipeline (chunking, pooled
    reduction across two files, checkpointing) the way a user actually would."""
    input_path = dataset_input({"numbers": {"metadata": {}, "files": {"a.root": 7, "b.root": 3}}})

    vr = VineReduce(
        processors={"count": count_events},
        input=input_path,
        reducer=sum_reducer,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    vr.compute()

    dataset_dir = os.path.join(vr.results_dir, "numbers")
    files = os.listdir(dataset_dir)
    assert len(files) == 1
    assert serialization.load(os.path.join(dataset_dir, files[0])) == 10
