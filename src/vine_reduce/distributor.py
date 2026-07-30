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

    def free_result(self, result_id: int) -> None:
        """Release any resources (e.g. worker-local files) held for result_id."""
        ...

    def hungry(self) -> int:
        """How many more chunks the distributor could usefully accept right now."""
        ...

    def retrieve(self, result_id: int, dest_path: str) -> None:
        """Copy the file for a completed (Success) result_id to dest_path, a
        path local to the vine_reduce process."""
        ...
