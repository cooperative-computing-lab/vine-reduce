"""A basic Distributor backed by executor.py's CloudpickleProcessPoolExecutor, for local
development and testing and as a minimal reference implementation.

  - Priority is best-effort: pending calls wait in a priority queue for a free worker slot, but a
    dispatched call is never preempted.
  - Workers are local subprocesses sharing vine_reduce's filesystem, so retrieve() is a file copy.
  - func/args are cloudpickled, so they may be closures or lambdas. The pool uses mp_context="fork",
    which is safe because this class starts no threads before creating it.
  - A checkpoint (is_checkpoint=True) lands under checkpoint_dir, which shutdown() never removes,
    so it survives restart. Ordinary results land in work_dir, scratch space that is a temp
    directory removed on shutdown() unless the caller supplied it.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import heapq
import itertools
import os
import shutil
import tempfile
import traceback as traceback_module
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable
from uuid import uuid4

from .distributor import Distributor, TaskKind
from .executor import CloudpickleProcessPoolExecutor
from .types import UNMEASURED, Outcome, RuntimeFailure, Success


def _run_with_env(func: Callable[..., Any], args: tuple, env_vars: dict[str, str]) -> Any:
    """Run func(*args) in the worker subprocess. env_vars is applied here rather than in the
    parent so it takes effect regardless of when the pool forked this worker."""
    os.environ.update(env_vars)
    return func(*args)


class LocalDistributor(Distributor):
    """Runs every call in a local process pool; the default when VineReduce gets no
    `distributor=`."""

    def __init__(
        self,
        max_workers: int | None = None,
        work_dir: str | None = None,
    ):
        """max_workers: size of the process pool; defaults to the CPU count. work_dir: where
        ordinary (non-checkpoint) results are written; defaults to a temp directory removed on
        shutdown() (a caller-supplied one is left in place). Checkpoints go to the directory
        given to set_checkpoint_dir()."""
        self._max_workers = max_workers or os.process_cpu_count() or 1
        self._pool = CloudpickleProcessPoolExecutor(max_workers=self._max_workers)

        self._owns_work_dir = work_dir is None
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="vine_reduce_local_")
        os.makedirs(self._work_dir, exist_ok=True)

        self._checkpoint_dir: str | None = None

        self._seq = itertools.count()
        # Heap of (-priority, seq, result_id, func, args, is_checkpoint): priority is negated so
        # the largest pops first; seq breaks ties in submission order.
        self._pending: list[tuple[int, int, str, Callable, tuple, bool]] = []
        # future -> (result_id, dest_file), while dispatched. wait() needs dest_file whatever the
        # outcome, since a failed call leaves a placeholder there that must be removed.
        self._running: dict[Future, tuple[str, str]] = {}
        self._files: dict[str, str] = {}  # result_id -> file, for completed Successes
        self._env_vars: dict[str, str] = {}

    def submit(
        self,
        result_id: str,
        priority: int,
        category: str,
        kind: TaskKind,
        func: Callable[..., Any],
        *args: Any,
        is_checkpoint: bool = False,
    ) -> None:
        """Queue func(dest_file, *args) for the pool, larger priority first. is_checkpoint picks
        checkpoint_dir over work_dir for dest_file."""
        heapq.heappush(
            self._pending, (-priority, next(self._seq), result_id, func, args, is_checkpoint)
        )
        self._dispatch()

    def _dispatch(self) -> None:
        while self._pending and len(self._running) < self._max_workers:
            _, _, result_id, func, args, is_checkpoint = heapq.heappop(self._pending)
            base_dir = self._checkpoint_dir if is_checkpoint else self._work_dir
            dest_file = os.path.join(base_dir, f"{uuid4().hex}.pkl.zst")
            future = self._pool.submit(_run_with_env, func, (dest_file, *args), self._env_vars)
            self._running[future] = (result_id, dest_file)

    def wait(self, timeout: float | None = None) -> Outcome | None:
        """Block until a call finishes and return its Outcome, or None on timeout or if nothing
        is running. Never raises for a task-level fault: a worker that dies outright (segfault,
        OOM-kill) breaks the pool, which is turned into a RuntimeFailure and the pool rebuilt."""
        if not self._running:
            return None
        done, _ = concurrent.futures.wait(
            self._running, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED
        )
        if not done:
            return None

        future = next(iter(done))
        result_id, dest_file = self._running.pop(future)

        try:
            raw: Outcome = future.result()
        except BrokenProcessPool:
            self._pool.shutdown(wait=False)
            self._pool = CloudpickleProcessPoolExecutor(max_workers=self._max_workers)
            if os.path.exists(dest_file):
                os.remove(dest_file)
            outcome: Outcome = RuntimeFailure(
                result_id=result_id,
                resources=UNMEASURED,
                std_output=None,
                traceback=traceback_module.format_exc(),
            )
            self._dispatch()
            return outcome

        if isinstance(raw, Success):
            self._files[result_id] = raw.file
        elif os.path.exists(dest_file):
            os.remove(dest_file)
        outcome = dataclasses.replace(raw, result_id=result_id)

        self._dispatch()
        return outcome

    def release_result(self, result_id: str) -> None:
        """Delete the result file for a completed (Success) result_id."""
        path = self._files.pop(result_id, None)
        if path is not None and os.path.exists(path):
            os.remove(path)

    def adopt_checkpoint(self, result_id: str, path: str) -> str:
        """Register `path` under result_id. Workers share the filesystem, so `path` is itself
        the handle."""
        self._files[result_id] = path
        return path

    def resources(self, kind: TaskKind) -> dict[str, Any] | None:
        """Always {"cores": 1}: the pool shares a machine (often a shared frontend), so a task
        must not assume it can use every core."""
        return {"cores": 1}

    def capacity(self) -> int:
        """Room left before running plus pending calls reach twice max_workers."""
        target_queue_depth = 2 * self._max_workers
        in_flight = len(self._running) + len(self._pending)
        return max(0, target_queue_depth - in_flight)

    def retrieve(self, result_id: str, dest_path: str) -> None:
        """Copy the result file for a completed (Success) result_id to dest_path."""
        shutil.copy(self._files[result_id], dest_path)

    def checkpoint_path(self, result_id: str) -> str:
        """The path the result already lives at."""
        return self._files[result_id]

    def set_checkpoint_dir(self, path: str) -> None:
        """Directory checkpoint results land under from now on. Must precede any
        is_checkpoint=True submit()."""
        self._checkpoint_dir = path
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def add_file(self, local_path: str, remote_path: str | None = None) -> None:
        """No-op: workers already see local_path. remote_path is unused."""

    def set_env_var(self, name: str, value: str) -> None:
        """Set an environment variable, applied in each worker, for calls submitted from now on."""
        self._env_vars[name] = value

    def shutdown(self) -> None:
        """Shut down the pool (dropping calls not yet started, so an abort doesn't block) and
        remove work_dir if this distributor created it. checkpoint_dir is never removed."""
        self._pool.shutdown(wait=True, cancel_futures=True)
        if self._owns_work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)
