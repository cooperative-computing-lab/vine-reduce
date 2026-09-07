"""Toy functions shared across tests. Kept as plain, importable, module-level
callables (cloudpickle can serialize closures/lambdas too, but these are
reused across several test modules, so a shared name is simpler)."""

from __future__ import annotations

import os
import threading
from uuid import uuid4

from vine_reduce.defaults import default_chunk_to_args, executor_wrapper
from vine_reduce.executor import SimpleExecutor


def read_env_var(chunk):
    return os.environ.get("VINE_REDUCE_TEST_VAR", "")


def read_shipped_file(chunk):
    with open("shipped.txt") as f:
        return f.read()


def count_events(chunk):
    return chunk.stop - chunk.start


def submit_chunk(distributor, priority, chunk, is_checkpoint=False):
    result_id = uuid4().hex
    distributor.submit(
        result_id,
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
        SimpleExecutor(),
        is_checkpoint=is_checkpoint,
    )
    return result_id


def sum_reducer(a, b):
    return a + b


def double_postprocess(x):
    return x * 2


def failing_processor(chunk):
    raise ValueError("boom")


def exhausting_processor(chunk):
    raise MemoryError("simulated resource exhaustion")


def crashing_processor(chunk):
    os._exit(1)  # simulates a segfault/OOM-kill: no exception, no return


def make_flaky_n_times(n):
    """Returns a processor that raises ValueError on its first `n` calls
    (across all chunks - a single shared counter), then succeeds like
    count_events. Only ever run in-process via FakeDistributor, so it
    doesn't need to be picklable."""
    calls = {"count": 0}

    def processor(chunk):
        calls["count"] += 1
        if calls["count"] <= n:
            raise ValueError("transient boom")
        return chunk.stop - chunk.start

    processor.calls = calls
    return processor


def make_flaky_reducer_n_times(n):
    """Like make_flaky_n_times, but for a reducer: raises ValueError on its
    first `n` calls, then folds like sum_reducer."""
    calls = {"count": 0}

    def reducer(a, b):
        calls["count"] += 1
        if calls["count"] <= n:
            raise ValueError("transient boom")
        return a + b

    reducer.calls = calls
    return reducer


def unpicklable_processor(chunk):
    return threading.Lock()  # cloudpickle cannot serialize a lock
