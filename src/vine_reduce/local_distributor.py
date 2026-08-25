"""A basic Distributor backed by concurrent.futures.ProcessPoolExecutor.

This exists to run vine_reduce locally for development and testing, and to
serve as a minimal reference for what a Distributor implementation needs to
do. It is intentionally simple, not production-grade:
  - priority is best-effort only. A pending call waits in a priority queue
    until a worker slot is free, but once dispatched to the pool it cannot
    be preempted by a higher-priority call submitted later.
  - "worker nodes" are local subprocesses that share vine_reduce's
    filesystem, so retrieve() is a plain file copy.
  - func/args are cloudpickled before being handed to the pool (see
    _run_cloudpickled below), so processor/reducer/etc. may be closures
    or lambdas, not just module-level callables.
"""

from __future__ import annotations

import concurrent.futures
import heapq
import itertools
import os
import shutil
import tempfile
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Callable

import cloudpickle

from .types import Outcome, RawOutcome


def _run_cloudpickled(payload: bytes) -> Any:
    """Runs in the worker subprocess. ProcessPoolExecutor pickles whatever it
    is given with stdlib pickle, which cannot handle closures or lambdas;
    cloudpickle can, so func/args are cloudpickled into a byte string here
    and only that string (plus this module-level function) crosses the
    stdlib-pickle boundary. env_vars is applied here, in the worker process,
    rather than in the parent, so it takes effect regardless of when the
    pool actually forked this worker relative to set_env_var being called."""
    func, args, env_vars = cloudpickle.loads(payload)
    os.environ.update(env_vars)
    return func(*args)


class LocalDistributor:
    def __init__(self, max_workers: int | None = None, work_dir: str | None = None):
        self._max_workers = max_workers or os.process_cpu_count() or 1
        self._pool = ProcessPoolExecutor(max_workers=self._max_workers)

        self._owns_work_dir = work_dir is None
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="vine_reduce_local_")
        os.makedirs(self._work_dir, exist_ok=True)

        self._next_id = itertools.count(1)
        self._seq = itertools.count()
        # Heap of (-priority, seq, result_id, func, args): negated priority so
        # the largest one pops first, seq to break ties in submission order.
        self._pending: list[tuple[int, int, int, Callable, tuple]] = []
        self._running: dict[Future, int] = {}  # future -> result_id, while dispatched
        self._files: dict[int, str] = {}  # result_id -> file, for completed Successes
        self._env_vars: dict[str, str] = {}

    def submit(
        self, priority: int, category: str, kind: str, func: Callable[..., Any], *args: Any
    ) -> int:
        result_id = next(self._next_id)
        heapq.heappush(self._pending, (-priority, next(self._seq), result_id, func, args))
        self._dispatch()
        return result_id

    def _dispatch(self) -> None:
        while self._pending and len(self._running) < self._max_workers:
            _, _, result_id, func, args = heapq.heappop(self._pending)
            dest_file = os.path.join(self._work_dir, f"{result_id}.pkl.zst")
            payload = cloudpickle.dumps((func, (dest_file, *args), self._env_vars))
            self._running[self._pool.submit(_run_cloudpickled, payload)] = result_id

    def wait(self, timeout: float | None = None) -> Outcome | None:
        if not self._running:
            return None
        done, _ = concurrent.futures.wait(
            self._running, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED
        )
        if not done:
            return None

        future = next(iter(done))
        result_id = self._running.pop(future)

        raw: RawOutcome = future.result()
        if raw.status == "success":
            self._files[result_id] = raw.file
        outcome = raw.to_outcome(result_id)

        self._dispatch()
        return outcome

    def release_result(self, result_id: int) -> None:
        path = self._files.pop(result_id, None)
        if path is not None and os.path.exists(path):
            os.remove(path)

    def capacity(self) -> int:
        target_queue_depth = 2 * self._max_workers
        in_flight = len(self._running) + len(self._pending)
        return max(0, target_queue_depth - in_flight)

    def retrieve(self, result_id: int, dest_path: str) -> None:
        shutil.copy(self._files[result_id], dest_path)

    def add_file(self, local_path: str, remote_path: str | None = None) -> None:
        """No-op: worker subprocesses already share vine_reduce's filesystem
        (see module docstring), so local_path is already visible to them
        under that same path without shipping anything. remote_path is
        accepted for interface compatibility but unused."""

    def set_env_var(self, name: str, value: str) -> None:
        self._env_vars[name] = value

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        if self._owns_work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def __enter__(self) -> "LocalDistributor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
