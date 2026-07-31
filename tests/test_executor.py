from __future__ import annotations

import os

import dask

from vine_reduce.executor import _num_workers, cloudpickle_executor, dask_executor, simple_executor

from helpers import count_events


def test_simple_executor_calls_processor_directly():
    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    assert simple_executor(count_events, chunk, {}) == 5


def test_cloudpickle_executor_runs_in_a_subprocess_and_supports_closures():
    offset = 3

    def processor(chunk):
        import os

        return chunk.stop - chunk.start + offset, os.getpid()

    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    result, worker_pid = cloudpickle_executor(processor, chunk, {})

    assert result == 8
    assert worker_pid != __import__("os").getpid()


def test_dask_executor_computes_the_returned_dask_object():
    def processor(chunk):
        return dask.delayed(count_events)(chunk)

    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    assert dask_executor(processor, chunk, {}) == 5


def test_num_workers_uses_distributor_cores_when_reported():
    assert _num_workers({"cores": 3}) == 3


def test_num_workers_falls_back_to_machine_cores():
    assert _num_workers(None) == (os.process_cpu_count() or 1)
    assert _num_workers({}) == (os.process_cpu_count() or 1)
