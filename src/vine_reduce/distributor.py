"""The interface vine_reduce needs from a distributor.

A distributor submits opaque callables to worker nodes and reports their outcome. It knows nothing
about processors, chunks, or reductions.

`func` is always `executor_wrapper` or `reducer_wrapper` (defaults.py), which take the worker-local
destination file as their first argument. The distributor chooses that path and calls
`func(dest_file, *args)`; vine_reduce only ever sees it echoed back as `Outcome.file`.
"""

from __future__ import annotations

import abc
import enum
from typing import Any, Callable

from .types import Outcome


class TaskKind(enum.StrEnum):
    """Which stage a submitted call belongs to."""

    PROCESSOR = "processor"
    REDUCER = "reducer"


class Distributor(abc.ABC):
    """Implement this protocol (see LocalDistributor, TaskVineDistributor) to run vine_reduce on
    another backend."""

    @abc.abstractmethod
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
        """Submit func(dest_file, *args) for remote execution under a caller-minted result_id,
        unique for this distributor's lifetime and used by every later call and Outcome.result_id.

        Larger priority runs first. category groups related calls (logging, scheduling); kind lets
        the distributor apply different resources to processors and reducers. is_checkpoint marks a
        result that must outlive its producer (a checkpoint or final result), so the distributor
        should put it in durable rather than disposable storage; ignore it if there is only one
        kind."""
        ...

    @abc.abstractmethod
    def wait(self, timeout: float | None = None) -> Outcome | None:
        """Block until a submitted call finishes, returning its Outcome, or
        return None if timeout elapses first."""
        ...

    @abc.abstractmethod
    def release_result(self, result_id: str) -> None:
        """Release resources held for result_id. For a result submitted with
        is_checkpoint=True (or adopted), this must also delete its durable on-disk copy; the
        distributor is the sole owner of that file and Pipeline relies on it."""
        ...

    @abc.abstractmethod
    def adopt_checkpoint(self, result_id: str, path: str) -> str:
        """Register `path`, a durable checkpoint file from a previous run, under result_id as if it
        were a completed Success submitted with is_checkpoint=True. checkpoint_path() then returns
        `path`. Returns the distributor's handle for the file (like Outcome.file)."""
        ...

    @abc.abstractmethod
    def resources(self, kind: TaskKind) -> dict[str, Any] | None:
        """A static default resource dict (e.g. {"cores": ...}) for calls of this kind, or None.
        Passed as metadata["distributor"] into executor_wrapper, as a fallback when the execution
        site reports no more precise allocation (see DaskExecutor's _num_workers)."""
        ...

    @abc.abstractmethod
    def capacity(self) -> int:
        """How many more chunks the distributor could usefully accept right now."""
        ...

    @abc.abstractmethod
    def retrieve(self, result_id: str, dest_path: str) -> None:
        """Copy the file of a completed (Success) result_id to dest_path, local to the vine_reduce
        process. Used for final results, whose location is vine_reduce's choice."""
        ...

    @abc.abstractmethod
    def checkpoint_path(self, result_id: str) -> str:
        """Local durable path of a completed result_id submitted with is_checkpoint=True. The
        distributor picks it (under set_checkpoint_dir's path); vine_reduce records it for
        restart."""
        ...

    @abc.abstractmethod
    def set_checkpoint_dir(self, path: str) -> None:
        """Set the directory under which every checkpoint_path() will fall from now on. VineReduce
        owns the directory, so the distributor takes `path` as given. Called once, before any
        submit()."""
        ...

    @abc.abstractmethod
    def add_file(self, local_path: str, remote_path: str | None = None) -> None:
        """Make local_path available as remote_path (default: its basename) to every call submitted
        afterwards. May be a no-op if workers share vine_reduce's filesystem."""
        ...

    @abc.abstractmethod
    def set_env_var(self, name: str, value: str) -> None:
        """Set an environment variable for every processor/reducer call
        submitted after this point."""
        ...

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release whatever this distributor owns (worker pools, temp directories). Also called
        on exit from `with distributor: ...`."""
        ...

    def __enter__(self) -> "Distributor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
