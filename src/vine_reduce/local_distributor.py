"""A basic Distributor backed by concurrent.futures.ProcessPoolExecutor.

This exists to run vine_reduce locally for development and testing, and to
serve as a minimal reference for what a Distributor implementation needs to
do. It is intentionally simple, not production-grade:
  - priority is best-effort only. A pending call waits in a priority queue
    until a worker slot is free, but once dispatched to the pool it cannot
    be preempted by a higher-priority call submitted later.
  - "worker nodes" are local subprocesses that share vine_reduce's
    filesystem, so retrieve() is a plain file copy.
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

from .types import Outcome, RawOutcome


class LocalDistributor:
    def __init__(self, max_workers: int | None = None, work_dir: str | None = None):
        self._max_workers = max_workers or os.process_cpu_count() or 1
        self._pool = ProcessPoolExecutor(max_workers=self._max_workers)

        self._owns_work_dir = work_dir is None
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="vine_reduce_local_")
        os.makedirs(self._work_dir, exist_ok=True)

        self._next_id = itertools.count(1)
        self._seq = itertools.count()
        self._pending: list[tuple[int, int, int, Callable, tuple]] = (
            []
        )  # heap of (-priority, seq, id, func, args)
        self._futures: dict[int, Future] = {}
        self._result_id_of: dict[Future, int] = {}
        self._files: dict[int, str] = {}  # result_id -> file, for completed Successes

    def submit(self, priority: int, category: str, func: Callable[..., Any], *args: Any) -> int:
        result_id = next(self._next_id)
        heapq.heappush(self._pending, (-priority, next(self._seq), result_id, func, args))
        self._dispatch()
        return result_id

    def _dispatch(self) -> None:
        while self._pending and len(self._futures) < self._max_workers:
            _, _, result_id, func, args = heapq.heappop(self._pending)
            dest_file = os.path.join(self._work_dir, f"{result_id}.pkl")
            future = self._pool.submit(func, dest_file, *args)
            self._futures[result_id] = future
            self._result_id_of[future] = result_id

    def wait(self, timeout: float | None = None) -> Outcome | None:
        if not self._futures:
            return None
        done, _ = concurrent.futures.wait(
            self._futures.values(), timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED
        )
        if not done:
            return None

        future = next(iter(done))
        result_id = self._result_id_of.pop(future)
        del self._futures[result_id]

        raw: RawOutcome = future.result()
        if raw.status == "success":
            self._files[result_id] = raw.file
        outcome = raw.to_outcome(result_id)

        self._dispatch()
        return outcome

    def free_result(self, result_id: int) -> None:
        path = self._files.pop(result_id, None)
        if path is not None and os.path.exists(path):
            os.remove(path)

    def hungry(self) -> int:
        target_queue_depth = 2 * self._max_workers
        in_flight = len(self._futures) + len(self._pending)
        return max(0, target_queue_depth - in_flight)

    def retrieve(self, result_id: int, dest_path: str) -> None:
        shutil.copy(self._files[result_id], dest_path)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        if self._owns_work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)
