"""Implementations of the pipeline's `executor` step: the object
executor_wrapper (see defaults.py) calls to actually run processor(args).
All of these run remotely, at the execution site chosen by the distributor.

An executor implements the Executor protocol below - submit/map/shutdown,
named after concurrent.futures.Executor. It is configured once, in the
vine_reduce process, and cloudpickled into every remote call, so an executor
must always be picklable: any live resource (e.g. a process pool) is created
lazily on first submit/map and dropped before pickling (see
CloudpickleExecutor.__getstate__).

SimpleExecutor is the default: it just calls processor(args) directly, in
the same process running executor_wrapper.

CloudpickleExecutor and DaskExecutor are alternatives:
  - CloudpickleExecutor runs processor(args) in its own subprocess (via
    CloudpickleProcessPoolExecutor), so a crash or memory leak in processor
    doesn't take down the worker task itself.
  - DaskExecutor expects processor(args) to return a dask-delayed object
    (or dask array/dataframe) and computes it at the execution site, using
    dask's "processes" scheduler backed by CloudpickleProcessPoolExecutor -
    so tasks inside the dask graph may be closures/lambdas too, same as
    CloudpickleExecutor. dask is not a vine_reduce dependency; it must
    already be installed wherever this executor actually runs.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable, Iterator, Protocol

import cloudpickle


class Executor(Protocol):
    """The interface VineReduce needs from an executor. An executor runs
    fn(*args) at the execution site chosen by the distributor. It is
    configured once, in the vine_reduce process, and cloudpickled into every
    remote call (as part of executor_wrapper's arguments), so implementations
    must be picklable at all times - any live resource (e.g. a process pool)
    must be created lazily at submit/map time and dropped before pickling.
    executor_wrapper uses the fresh deserialized copy for exactly one call:
    `with executor: executor.submit(...).result()`."""

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        dataset_metadata: dict[str, Any] | None = None,
        distributor_metadata: dict[str, Any] | None = None,
        executor_metadata: dict[str, Any] | None = None,
    ) -> Future:
        """Run fn(*args) and return a Future for its result. The three
        metadata dicts are the same ones chunk_to_args receives - dataset
        metadata, per-task resource info from the distributor (e.g.
        "cores"), and free-form executor config."""
        ...

    def map(
        self,
        fn: Callable[..., Any],
        /,
        *iterables: Any,
        dataset_metadata: dict[str, Any] | None = None,
        distributor_metadata: dict[str, Any] | None = None,
        executor_metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Equivalent to submitting fn(*items) for each items in
        zip(*iterables) and yielding each result in order (stopping at the
        shortest iterable, like zip/concurrent.futures.Executor.map)."""
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Release whatever resources this executor owns (e.g. a process
        pool). Also reachable via `with executor: ...`, which calls this on
        exit."""
        ...

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown(wait=True)


class _ExecutorBase:
    """Shared map()/__enter__/__exit__ for every Executor implementation below.
    map() is defined once, in terms of submit(), and __enter__/__exit__ once,
    in terms of shutdown() - so a concrete implementation only needs to
    provide submit() and shutdown()."""

    def map(
        self,
        fn: Callable[..., Any],
        /,
        *iterables: Any,
        dataset_metadata: dict[str, Any] | None = None,
        distributor_metadata: dict[str, Any] | None = None,
        executor_metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        futures = [
            self.submit(
                fn,
                *items,
                dataset_metadata=dataset_metadata,
                distributor_metadata=distributor_metadata,
                executor_metadata=executor_metadata,
            )
            for items in zip(*iterables)
        ]
        return (future.result() for future in futures)

    def __enter__(self) -> "_ExecutorBase":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown(wait=True)


def _submitted(fn: Callable[..., Any], *args: Any) -> Future:
    """Runs fn(*args) inline and wraps its outcome in an already-done
    Future, the way a synchronous Executor.submit reports its result."""
    future: Future = Future()
    future.set_running_or_notify_cancel()
    try:
        future.set_result(fn(*args))
    except BaseException as exc:
        future.set_exception(exc)
    return future


class SimpleExecutor(_ExecutorBase):
    """Calls fn(*args) directly, in the same process/task running
    executor_wrapper. The default `executor` for VineReduce. Ignores all
    three metadata dicts. Stateless, so trivially picklable."""

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        dataset_metadata: dict[str, Any] | None = None,
        distributor_metadata: dict[str, Any] | None = None,
        executor_metadata: dict[str, Any] | None = None,
    ) -> Future:
        return _submitted(self._call, fn, *args)

    def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Hook for subclasses (see coffea.py's CoffeaExecutor) to change how
        fn is invoked or its result post-processed, without re-implementing
        the Future/exception plumbing above."""
        return fn(*args)

    def shutdown(self, wait: bool = True) -> None:
        pass


def _run_cloudpickled(payload: bytes) -> Any:
    """Runs in the subprocess. cloudpickle (unlike stdlib pickle) can
    serialize closures and lambdas, so fn may be either."""
    fn, args, kwargs = cloudpickle.loads(payload)
    return fn(*args, **kwargs)


class CloudpickleProcessPoolExecutor(ProcessPoolExecutor):
    """A ProcessPoolExecutor that cloudpickles fn/args/kwargs before they
    cross into the subprocess, so submit() accepts closures and lambdas,
    which stdlib pickle (what ProcessPoolExecutor normally relies on)
    cannot handle. The internal engine behind CloudpickleExecutor and
    DaskExecutor below."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, mp_context=mp.get_context("fork"))

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Like ProcessPoolExecutor.submit, but fn/args/kwargs may be
        closures or lambdas."""
        payload = cloudpickle.dumps((fn, args, kwargs))
        return super().submit(_run_cloudpickled, payload)


class CloudpickleExecutor(_ExecutorBase):
    """Runs each fn(*args) in its own subprocess (via
    CloudpickleProcessPoolExecutor), isolating a crash or memory leak in fn
    from the worker task running executor_wrapper. fn may be a closure or
    lambda. The pool is created lazily on first submit/map and dropped
    before pickling (see __getstate__), so a configured (even
    previously-used) instance always pickles cleanly; shutdown() tears the
    pool down. max_workers=1 (default) runs one call at a time; a larger
    value lets map() run items in parallel."""

    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max_workers
        self._pool: CloudpickleProcessPoolExecutor | None = None

    def _ensure_pool(self) -> CloudpickleProcessPoolExecutor:
        if self._pool is None:
            self._pool = CloudpickleProcessPoolExecutor(max_workers=self.max_workers)
        return self._pool

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        dataset_metadata: dict[str, Any] | None = None,
        distributor_metadata: dict[str, Any] | None = None,
        executor_metadata: dict[str, Any] | None = None,
    ) -> Future:
        return self._ensure_pool().submit(fn, *args)

    def shutdown(self, wait: bool = True) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None

    def __getstate__(self) -> dict[str, Any]:
        # A live pool (open pipes/subprocesses) never survives a pickle
        # boundary - the deserialized copy lazily builds its own on first use.
        state = self.__dict__.copy()
        state["_pool"] = None
        return state


def _num_workers(distributor_metadata: dict[str, Any] | None) -> int:
    """The task's actual core allocation, preferring the CORES environment
    variable - set by the execution site itself (e.g. TaskVine's worker) at
    dispatch time, so it reflects what was really handed to this task, which
    may be less than any configured cap. Falls back to
    distributor_metadata["cores"] - a static default the distributor can
    report ahead of dispatch (e.g. TaskVineDistributor's configured category
    cap) - then every core on the machine, if neither is available."""
    if "CORES" in os.environ:
        return int(os.environ["CORES"])
    if distributor_metadata and "cores" in distributor_metadata:
        return distributor_metadata["cores"]
    return os.process_cpu_count() or 1


class DaskExecutor(_ExecutorBase):
    """For an fn that returns a dask-delayed object (or dask
    array/dataframe): calls fn(*args), then computes the result at the
    execution site, on dask's "processes" scheduler backed by a
    CloudpickleProcessPoolExecutor created (and torn down) per call, with one
    subprocess per core allocated to this task - num_workers from the
    constructor if given, else the distributor/environment/machine core
    count (see _num_workers). dask is not a vine_reduce dependency; it must already be
    installed wherever this executor actually runs. Holds no state between
    calls, so trivially picklable; shutdown() is a no-op."""

    def __init__(self, num_workers: int | None = None) -> None:
        self.num_workers = num_workers

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        dataset_metadata: dict[str, Any] | None = None,
        distributor_metadata: dict[str, Any] | None = None,
        executor_metadata: dict[str, Any] | None = None,
    ) -> Future:
        def call() -> Any:
            to_maybe_compute = fn(*args)
            num_workers = self.num_workers or _num_workers(distributor_metadata)
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

        return _submitted(call)

    def shutdown(self, wait: bool = True) -> None:
        pass
