"""The interface vine_reduce needs from a distributor.

A distributor manages submitting calls to worker nodes and reporting their
outcome back. It knows nothing about processors, chunks, or reductions -
just opaque callables and their results.

Convention for `func`: vine_reduce always submits `executor_wrapper` or
`reducer_wrapper` from defaults.py, both of which take the worker-local
destination file path as their *first* argument. A distributor is
responsible for choosing that path and prepending it itself, i.e. it should
call `func(dest_file, *args)`, not `func(*args)`. This is what "this file is
entirely maintained by the distributor" means in PLAN.md: vine_reduce never
picks the path, it only ever sees it echoed back on `Outcome.file`.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Protocol

from .types import Outcome

TaskKind = Literal["processor", "reducer"]


class Distributor(Protocol):
    def submit(
        self,
        priority: int,
        category: str,
        kind: TaskKind,
        func: Callable[..., Any],
        *args: Any,
    ) -> int:
        """Submit a call for remote execution. Larger priority runs first.
        category groups calls belonging to the same processing/reduction set
        (e.g. for logging or scheduling heuristics). kind says whether this is
        a processor or reducer call, so a distributor can apply different
        resource requests to each. Returns a result_id."""
        ...

    def wait(self, timeout: float | None = None) -> Outcome | None:
        """Block until a submitted call finishes, returning its Outcome, or
        return None if timeout elapses first."""
        ...

    def release_result(self, result_id: int) -> None:
        """Release any resources (e.g. worker-local files) held for result_id."""
        ...

    def capacity(self) -> int:
        """How many more chunks the distributor could usefully accept right now."""
        ...

    def retrieve(self, result_id: int, dest_path: str) -> None:
        """Copy the file for a completed (Success) result_id to dest_path, a
        path local to the vine_reduce process."""
        ...

    def add_file(self, local_path: str) -> None:
        """Make local_path (readable from the vine_reduce process) available,
        under its basename, wherever every processor/reducer call submitted
        after this point runs. For a distributor whose workers already share
        vine_reduce's filesystem, this can be a no-op."""
        ...

    def set_env_var(self, name: str, value: str) -> None:
        """Set an environment variable for every processor/reducer call
        submitted after this point."""
        ...

    def shutdown(self) -> None:
        """Release whatever resources this distributor owns (worker pools,
        temp directories, ...). Also reachable via `with distributor: ...`,
        which calls this on exit."""
        ...

    def __enter__(self) -> "Distributor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
