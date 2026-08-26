"""Implementations of the pipeline's `executor` step: the callable
executor_wrapper (see defaults.py) invokes to actually run processor(args).
All of these run remotely, at the execution site chosen by the distributor.

simple_executor is the default: it just calls processor(args) directly, in
the same process running executor_wrapper.

cloudpickle_executor and dask_executor are alternatives:
  - cloudpickle_executor runs processor(args) in its own subprocess (via
    CloudpickleProcessPoolExecutor), so a crash or memory leak in processor
    doesn't take down the worker task itself.
  - dask_executor expects processor(args) to return a dask-delayed object
    (or dask array/dataframe) and computes it at the execution site, using
    dask's "processes" scheduler backed by CloudpickleProcessPoolExecutor -
    so tasks inside the dask graph may be closures/lambdas too, same as
    cloudpickle_executor. dask is not a vine_reduce dependency; it must
    already be installed wherever this executor actually runs.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable

import cloudpickle


def simple_executor(
    processor: Callable[[Any], Any],
    args: Any,
    dataset_metadata: dict[str, Any],
    distributor_metadata: dict[str, Any] | None = None,
    executor_metadata: dict[str, Any] | None = None,
) -> Any:
    """Calls processor(args) directly, in the same process/task running
    executor_wrapper. The default `executor` for VineReduce."""
    return processor(args)


def _run_cloudpickled(payload: bytes) -> Any:
    """Runs in the subprocess. cloudpickle (unlike stdlib pickle) can
    serialize closures and lambdas, so processor may be either."""
    fn, args, kwargs = cloudpickle.loads(payload)
    return fn(*args, **kwargs)


class CloudpickleProcessPoolExecutor(ProcessPoolExecutor):
    """A ProcessPoolExecutor that cloudpickles fn/args/kwargs before they
    cross into the subprocess, so submit() accepts closures and lambdas,
    which stdlib pickle (what ProcessPoolExecutor normally relies on)
    cannot handle."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, mp_context=mp.get_context("fork"))

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Like ProcessPoolExecutor.submit, but fn/args/kwargs may be
        closures or lambdas."""
        payload = cloudpickle.dumps((fn, args, kwargs))
        return super().submit(_run_cloudpickled, payload)


def cloudpickle_executor(
    processor: Callable[[Any], Any],
    args: Any,
    dataset_metadata: dict[str, Any],
    distributor_metadata: dict[str, Any] | None = None,
    executor_metadata: dict[str, Any] | None = None,
) -> Any:
    """Runs processor(args) in its own subprocess, isolating this one call
    from the worker process that's running executor_wrapper."""
    with CloudpickleProcessPoolExecutor(max_workers=1) as pool:
        return pool.submit(processor, args).result()


def _num_workers(distributor_metadata: dict[str, Any] | None) -> int:
    """The task's own core allocation, as reported by the distributor, or
    every core on the machine if the distributor doesn't report one."""
    if distributor_metadata and "cores" in distributor_metadata:
        return distributor_metadata["cores"]
    return os.process_cpu_count() or 1


def dask_executor(
    processor: Callable[[Any], Any],
    args: Any,
    dataset_metadata: dict[str, Any],
    distributor_metadata: dict[str, Any] | None = None,
    executor_metadata: dict[str, Any] | None = None,
) -> Any:
    """Calls processor(args) and computes the dask object it returns, at the
    execution site, on dask's "processes" scheduler backed by
    CloudpickleProcessPoolExecutor, with one subprocess per core allocated
    to this task (see _num_workers)."""
    to_maybe_compute = processor(args)
    num_workers = _num_workers(distributor_metadata)
    with CloudpickleProcessPoolExecutor(max_workers=num_workers) as pool:
        return to_maybe_compute.compute(
            scheduler="processes",
            pool=pool,
            optimize_graph=True,
            num_workers=num_workers,
            max_height=None,
            max_width=1,
            subgraphs=False,
        )
