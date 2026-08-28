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
  - a checkpoint (submit(..., is_checkpoint=True)) lands under
    checkpoint_dir, a real directory that shutdown() never removes; an
    ordinary result lands under work_dir instead, which is scratch space
    - a fresh temp directory removed on shutdown() unless the caller
    supplied its own. This split mirrors TaskVineDistributor's own
    checkpoint_dir/declare_temp() split, and matters for restart: a
    checkpoint has to still be there the next time this process starts,
    which a result sitting in a wiped temp dir would not be.
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
from uuid import uuid4

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
    """A Distributor that runs every processor/reducer call in a local
    ProcessPoolExecutor - the default when VineReduce is constructed without
    a `distributor=`. See the module docstring for what it is and isn't good
    for."""

    def __init__(
        self,
        max_workers: int | None = None,
        work_dir: str | None = None,
        checkpoint_dir: str = "checkpoints",
    ):
        """max_workers: size of the local process pool; defaults to the
        machine's CPU count. work_dir: directory to write ordinary
        (non-checkpoint) result files into; defaults to a fresh temp
        directory that is removed on shutdown() (a caller-supplied work_dir
        is left in place). checkpoint_dir: directory to write checkpoint
        (submit(..., is_checkpoint=True)) result files into; never removed
        by shutdown() - see the module docstring."""
        self._max_workers = max_workers or os.process_cpu_count() or 1
        self._pool = ProcessPoolExecutor(max_workers=self._max_workers)

        self._owns_work_dir = work_dir is None
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="vine_reduce_local_")
        os.makedirs(self._work_dir, exist_ok=True)

        self._checkpoint_dir = checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._seq = itertools.count()
        # Heap of (-priority, seq, result_id, func, args, is_checkpoint): negated
        # priority so the largest one pops first, seq to break ties in
        # submission order.
        self._pending: list[tuple[int, int, str, Callable, tuple, bool]] = []
        self._running: dict[Future, str] = {}  # future -> result_id, while dispatched
        self._files: dict[str, str] = {}  # result_id -> file, for completed Successes
        self._env_vars: dict[str, str] = {}

    def submit(
        self,
        result_id: str,
        priority: int,
        category: str,
        kind: str,
        func: Callable[..., Any],
        *args: Any,
        is_checkpoint: bool = False,
    ) -> None:
        """Queue func(dest_file, *args) to run in the process pool, ordered
        by priority (larger runs first). is_checkpoint picks which directory
        the result lands under - checkpoint_dir or work_dir, see _dispatch
        and the module docstring."""
        heapq.heappush(
            self._pending, (-priority, next(self._seq), result_id, func, args, is_checkpoint)
        )
        self._dispatch()

    def _dispatch(self) -> None:
        while self._pending and len(self._running) < self._max_workers:
            _, _, result_id, func, args, is_checkpoint = heapq.heappop(self._pending)
            base_dir = self._checkpoint_dir if is_checkpoint else self._work_dir
            dest_file = os.path.join(base_dir, f"{uuid4().hex}.pkl.zst")
            payload = cloudpickle.dumps((func, (dest_file, *args), self._env_vars))
            self._running[self._pool.submit(_run_cloudpickled, payload)] = result_id

    def wait(self, timeout: float | None = None) -> Outcome | None:
        """Block until a queued call finishes, returning its Outcome, or
        None if timeout elapses (or nothing is running) first."""
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

    def release_result(self, result_id: str) -> None:
        """Delete the result file for a completed (Success) result_id."""
        path = self._files.pop(result_id, None)
        if path is not None and os.path.exists(path):
            os.remove(path)

    def adopt_checkpoint(self, result_id: str, path: str) -> str:
        """Register an existing durable checkpoint file at `path` under
        result_id, as if it were a completed Success result submitted with
        is_checkpoint=True - see the Distributor protocol docstring. Worker
        subprocesses already share vine_reduce's filesystem, so `path` itself
        is usable as-is and is also this distributor's handle for it."""
        self._files[result_id] = path
        return path

    def capacity(self) -> int:
        """Room left before the pool + its pending queue reaches twice
        max_workers, this distributor's target queue depth."""
        target_queue_depth = 2 * self._max_workers
        in_flight = len(self._running) + len(self._pending)
        return max(0, target_queue_depth - in_flight)

    def retrieve(self, result_id: str, dest_path: str) -> None:
        """Copy the result file for a completed (Success) result_id to
        dest_path - a plain file copy, since worker subprocesses already
        share vine_reduce's filesystem."""
        shutil.copy(self._files[result_id], dest_path)

    def checkpoint_path(self, result_id: str) -> str:
        """The real path a completed (Success) result_id already lives at -
        see submit()."""
        return self._files[result_id]

    def add_file(self, local_path: str, remote_path: str | None = None) -> None:
        """No-op: worker subprocesses already share vine_reduce's filesystem
        (see module docstring), so local_path is already visible to them
        under that same path without shipping anything. remote_path is
        accepted for interface compatibility but unused."""

    def set_env_var(self, name: str, value: str) -> None:
        """Set an environment variable for every call submitted from now on,
        applied inside each worker subprocess (see _run_cloudpickled)."""
        self._env_vars[name] = value

    def shutdown(self) -> None:
        """Shut down the process pool and, if this distributor created its
        own work_dir, remove it. checkpoint_dir is never removed here -
        see the module docstring."""
        self._pool.shutdown(wait=True)
        if self._owns_work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def __enter__(self) -> "LocalDistributor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
