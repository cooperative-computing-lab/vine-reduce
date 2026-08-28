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

from .types import Outcome, ResultHandle

TaskKind = Literal["processor", "reducer"]


class Distributor(Protocol):
    """The interface VineReduce needs from a distributor. Implement this
    protocol (see LocalDistributor and TaskVineDistributor for two examples)
    to run vine_reduce against a different backend."""

    def submit(
        self,
        priority: int,
        category: str,
        kind: TaskKind,
        func: Callable[..., Any],
        *args: Any,
        is_checkpoint: bool = False,
    ) -> int:
        """Submit a call for remote execution. Larger priority runs first.
        category groups calls belonging to the same processing/reduction set
        (e.g. for logging or scheduling heuristics). kind says whether this is
        a processor or reducer call, so a distributor can apply different
        resource requests to each. is_checkpoint marks a call whose result
        vine_reduce needs to survive independently of whatever produced it -
        a non-final checkpoint or a final result (see PLAN.md's "Temporary
        Results, Checkpoints, and Restart") - so a distributor that
        distinguishes durable from disposable storage (e.g.
        TaskVineDistributor's vine_file(cache=True) vs vine_temp()) knows
        which to use; a distributor with only one kind of storage can ignore
        it. Returns a result_id."""
        ...

    def wait(self, timeout: float | None = None) -> Outcome | None:
        """Block until a submitted call finishes, returning its Outcome, or
        return None if timeout elapses first."""
        ...

    def release_result(self, result_id: int) -> None:
        """Release any resources (e.g. worker-local files) held for
        result_id. This is a hard requirement, not just cleanup: for a
        result submitted with is_checkpoint=True (or adopted via
        adopt_checkpoint), release_result must also remove the checkpoint's
        durable on-disk copy - the distributor is the single owner of that
        file. Both shipped distributors (LocalDistributor,
        TaskVineDistributor) already behave this way; Pipeline relies on it
        rather than deleting checkpoint files itself."""
        ...

    def adopt_checkpoint(self, path: str) -> ResultHandle:
        """Register `path` - an existing durable checkpoint file written by
        a previous run and recorded in the checkpoint store - as if it were
        a completed Success result of this run submitted with
        is_checkpoint=True. The returned handle's result_id is valid for
        release_result/retrieve/checkpoint_path (checkpoint_path returns
        `path`), and its file may appear in later submit() args exactly like
        an Outcome.file."""
        ...

    def capacity(self) -> int:
        """How many more chunks the distributor could usefully accept right now."""
        ...

    def retrieve(self, result_id: int, dest_path: str) -> None:
        """Copy the file for a completed (Success) result_id to dest_path, a
        path local to the vine_reduce process. Used for final results, whose
        location (results_dir) and naming are vine_reduce's own convention,
        independent of the distributor."""
        ...

    def checkpoint_path(self, result_id: int) -> str:
        """Local, durable on-disk path for a completed (Success) result_id
        that was submitted with is_checkpoint=True. Unlike retrieve(), the
        distributor chooses this path itself (e.g. TaskVineDistributor's own
        checkpoint_dir) - vine_reduce only needs to know where it ended up,
        to record in the checkpoint db for restart. Only valid for a result
        submitted with is_checkpoint=True."""
        ...

    def add_file(self, local_path: str, remote_path: str | None = None) -> None:
        """Make local_path (readable from the vine_reduce process) available,
        under remote_path (defaulting to local_path's basename), wherever
        every processor/reducer call submitted after this point runs. For a
        distributor whose workers already share vine_reduce's filesystem,
        this can be a no-op."""
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
