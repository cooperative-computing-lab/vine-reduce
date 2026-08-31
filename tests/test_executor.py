from __future__ import annotations

import os
from concurrent.futures.process import BrokenProcessPool

import cloudpickle
import dask
import pytest

from vine_reduce.executor import CloudpickleExecutor, DaskExecutor, SimpleExecutor, _num_workers

from helpers import count_events


def _crash(chunk):
    os._exit(1)


def _raise_boom():
    raise ValueError("boom")


# -- SimpleExecutor -----------------------------------------------------------


def test_simple_executor_submit_calls_fn_directly():
    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    assert SimpleExecutor().submit(count_events, chunk).result() == 5


def test_simple_executor_submit_does_not_raise_but_result_does():
    future = SimpleExecutor().submit(_raise_boom)
    with pytest.raises(ValueError, match="boom"):
        future.result()


def test_simple_executor_map_preserves_order_and_stops_at_shortest():
    result = list(SimpleExecutor().map(lambda a, b: a + b, [1, 2, 3], [10, 20]))
    assert result == [11, 22]


def test_simple_executor_pickles_and_still_works():
    copy = cloudpickle.loads(cloudpickle.dumps(SimpleExecutor()))
    assert copy.submit(count_events, type("Chunk", (), {"start": 0, "stop": 5})()).result() == 5


def test_simple_executor_context_manager():
    with SimpleExecutor() as ex:
        assert ex.submit(count_events, type("Chunk", (), {"start": 0, "stop": 5})()).result() == 5


def test_simple_executor_shutdown_on_unused_instance_is_noop():
    SimpleExecutor().shutdown()


# -- CloudpickleExecutor -------------------------------------------------------


def test_cloudpickle_executor_runs_in_a_subprocess_and_supports_closures():
    offset = 3

    def processor(chunk):
        import os

        return chunk.stop - chunk.start + offset, os.getpid()

    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    result, worker_pid = CloudpickleExecutor().submit(processor, chunk).result()

    assert result == 8
    assert worker_pid != os.getpid()


def test_cloudpickle_executor_isolates_a_crash_from_the_caller():
    executor = CloudpickleExecutor()
    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    future = executor.submit(_crash, chunk)
    with pytest.raises(BrokenProcessPool):
        future.result()
    executor.shutdown()


def test_cloudpickle_executor_map_preserves_order():
    executor = CloudpickleExecutor(max_workers=2)
    result = list(executor.map(lambda a, b: a + b, [1, 2, 3], [10, 20, 30]))
    assert result == [11, 22, 33]
    executor.shutdown()


def test_cloudpickle_executor_pickles_without_a_live_pool():
    executor = CloudpickleExecutor()
    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    executor.submit(count_events, chunk).result()  # pool now exists
    assert executor._pool is not None

    copy = cloudpickle.loads(cloudpickle.dumps(executor))
    assert copy._pool is None
    assert copy.submit(count_events, chunk).result() == 5
    copy.shutdown()
    executor.shutdown()


def test_cloudpickle_executor_shutdown_on_unused_instance_is_noop():
    CloudpickleExecutor().shutdown()


def test_cloudpickle_executor_context_manager_shuts_pool_down_on_exit():
    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    with CloudpickleExecutor() as ex:
        ex.submit(count_events, chunk).result()
        assert ex._pool is not None
    assert ex._pool is None


# -- DaskExecutor ---------------------------------------------------------------


def test_dask_executor_computes_the_returned_dask_object():
    def processor(chunk):
        return dask.delayed(count_events)(chunk)

    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    assert DaskExecutor().submit(processor, chunk).result() == 5


def test_dask_executor_constructor_num_workers_overrides_distributor_metadata():
    seen_num_workers = []

    class RecordingDelayed:
        def compute(self, **kwargs):
            seen_num_workers.append(kwargs["num_workers"])
            return 5

    def processor(chunk):
        return RecordingDelayed()

    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    result = (
        DaskExecutor(num_workers=2)
        .submit(processor, chunk, distributor_metadata={"cores": 7})
        .result()
    )

    assert result == 5
    assert seen_num_workers == [2]


def test_dask_executor_raising_fn_surfaces_from_result_not_submit():
    def processor(chunk):
        raise ValueError("boom")

    chunk = type("Chunk", (), {"start": 0, "stop": 5})()
    future = DaskExecutor().submit(processor, chunk)
    with pytest.raises(ValueError, match="boom"):
        future.result()


def test_num_workers_uses_distributor_cores_when_reported(monkeypatch):
    monkeypatch.delenv("CORES", raising=False)
    assert _num_workers({"cores": 3}) == 3


def test_num_workers_uses_cores_env_var_when_distributor_reports_nothing(monkeypatch):
    monkeypatch.setenv("CORES", "5")
    assert _num_workers(None) == 5
    assert _num_workers({}) == 5


def test_num_workers_cores_env_var_beats_distributor_metadata(monkeypatch):
    """CORES reflects the execution site's real, dispatch-time allocation
    (e.g. TaskVine's worker), which may be less than distributor_metadata's
    static cap - so it takes precedence."""
    monkeypatch.setenv("CORES", "5")
    assert _num_workers({"cores": 3}) == 5


def test_num_workers_falls_back_to_machine_cores(monkeypatch):
    monkeypatch.delenv("CORES", raising=False)
    assert _num_workers(None) == (os.process_cpu_count() or 1)
    assert _num_workers({}) == (os.process_cpu_count() or 1)
