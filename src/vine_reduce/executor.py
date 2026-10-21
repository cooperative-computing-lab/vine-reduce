"""Pipeline `executor` step: the object executor_wrapper (defaults.py) calls to run
processor(args) at the remote execution site.

An executor is configured once locally and cloudpickled into every remote call, so it must always
be picklable: live resources (e.g. a process pool) are created lazily on first submit and dropped
in __getstate__.

SimpleExecutor (default) calls processor(args) in the process running executor_wrapper.
CloudpickleExecutor runs it in a subprocess, so a crash or leak doesn't take down the worker
task. DaskExecutor computes a dask-delayed result with dask's "processes" scheduler; dask is not
a dependency and must be installed at the execution site.
"""

from __future__ import annotations

import abc
import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable

import cloudpickle


class Executor(abc.ABC):
    """Interface for running fn(*args) at the execution site. Instances are cloudpickled into each
    remote call and used for exactly one call (`with executor: executor.submit(...).result()`),
    so they must be picklable at all times: create live resources lazily at submit time and drop
    them before pickling. Subclasses implement submit() and shutdown()."""

    @abc.abstractmethod
    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        metadata: dict[str, dict[str, Any] | None] | None = None,
    ) -> Future:
        """Run fn(*args) and return a Future for its result. metadata has the same "dataset",
        "distributor" (e.g. "cores") and "executor" dicts that chunk_to_args receives."""
        ...

    @abc.abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Release owned resources (e.g. a process pool); called on `with` exit."""
        ...

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown(wait=True)


def _completed_future(fn: Callable[..., Any], *args: Any) -> Future:
    """Run fn(*args) inline and return an already-done Future holding its result or exception."""
    future: Future = Future()
    future.set_running_or_notify_cancel()
    try:
        future.set_result(fn(*args))
    except BaseException as exc:
        future.set_exception(exc)
    return future


class SimpleExecutor(Executor):
    """Calls fn(*args) directly in the process running executor_wrapper. The default executor;
    ignores metadata and holds no state."""

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        metadata: dict[str, dict[str, Any] | None] | None = None,
    ) -> Future:
        return _completed_future(self._call, fn, *args)

    def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Hook for subclasses (e.g. CoffeaExecutor) to change how fn is invoked or its result
        post-processed."""
        return fn(*args)

    def shutdown(self, wait: bool = True) -> None:
        pass


def _run_cloudpickled(payload: bytes) -> Any:
    """Subprocess entry point; cloudpickle lets fn be a closure or lambda."""
    fn, args, kwargs = cloudpickle.loads(payload)
    return fn(*args, **kwargs)


class CloudpickleProcessPoolExecutor(ProcessPoolExecutor):
    """ProcessPoolExecutor that cloudpickles fn/args/kwargs so closures and lambdas can be
    submitted. The engine behind CloudpickleExecutor and DaskExecutor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, mp_context=mp.get_context("fork"))

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Like ProcessPoolExecutor.submit, but fn/args/kwargs may be closures or lambdas."""
        payload = cloudpickle.dumps((fn, args, kwargs))
        return super().submit(_run_cloudpickled, payload)


class CloudpickleExecutor(Executor):
    """Runs each fn(*args) in a subprocess, isolating crashes and memory leaks from the worker
    task. fn may be a closure or lambda. The pool is created lazily and dropped in __getstate__,
    so instances always pickle cleanly. executor_wrapper submits once per instance, so
    max_workers>1 has no effect within VineReduce."""

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
        metadata: dict[str, dict[str, Any] | None] | None = None,
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
    """Cores available to the task. The CORES environment variable (set by the execution site at
    dispatch, e.g. TaskVine's worker) takes precedence because it reflects the real allocation,
    which may be below any configured cap. Then distributor_metadata["cores"] (the distributor's
    static cap), then every core on the machine."""
    if "CORES" in os.environ:
        return int(os.environ["CORES"])
    if distributor_metadata and "cores" in distributor_metadata:
        return distributor_metadata["cores"]
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


class DaskExecutor(Executor):
    """Computes the dask-delayed object (or dask array/dataframe) returned by fn(*args) at the
    execution site, on dask's "processes" scheduler backed by a per-call
    CloudpickleProcessPoolExecutor of num_workers processes (default: see _num_workers).
    Stateless; shutdown() is a no-op."""

    def __init__(self, num_workers: int | None = None) -> None:
        self.num_workers = num_workers

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        metadata: dict[str, dict[str, Any] | None] | None = None,
    ) -> Future:
        def call() -> Any:
            to_maybe_compute = fn(*args)
            num_workers = self.num_workers or _num_workers((metadata or {}).get("distributor"))
            with CloudpickleProcessPoolExecutor(max_workers=num_workers) as pool:
                # max_height/max_width bound how far dask's optimizer fuses and
                # repartitions the task graph before scheduling it; subgraphs=False
                # disables dask's subgraph-callable fusion (each graph node stays
                # its own pool task instead of being bundled into a callable).
                return to_maybe_compute.compute(
                    scheduler="processes",
                    pool=pool,
                    optimize_graph=True,
                    num_workers=num_workers,
                    max_height=None,
                    max_width=1,
                    subgraphs=False,
                )

        return _completed_future(call)

    def shutdown(self, wait: bool = True) -> None:
        pass
