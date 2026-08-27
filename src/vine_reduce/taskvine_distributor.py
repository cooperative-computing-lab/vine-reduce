"""A Distributor backed by ndcctools.taskvine, for running vine_reduce across
a real cluster of machines instead of local subprocesses.

Manager-only: this class starts a vine.Manager and nothing else. Worker
processes (vine_worker, a factory, batch-system submission, ...) are the
caller's responsibility, same as any other TaskVine application - see
https://cctools.readthedocs.io/en/latest/taskvine/.

Bridging the Distributor protocol's plain file-path strings onto TaskVine's
file model (see distributor.py's docstring on what "file" means) works like
this: an ordinary (non-checkpoint) result gets a manager.declare_temp() file,
which TaskVine keeps at/near the worker that produced it rather than pulling
it back to the manager. A checkpoint or final result (submit(...,
is_checkpoint=True) - see PLAN.md's "Temporary Results, Checkpoints, and
Restart") instead gets a manager.declare_file(path, cache=True) file, with
`path` a fresh name inside checkpoint_dir: TaskVine transfers a
declare_file() output back to the manager unconditionally as soon as the
task completes (unlike a temp file, which stays remote until explicitly
fetched), so by the time wait() reports success the checkpoint already sits
durably on local disk at `path` - see checkpoint_path(). Either way,
Outcome.file is not that file's real location but an opaque token this class
mints, e.g. "result_7.p". When that token later shows up inside another
submit() call's args (as one of reducer_wrapper's input_files),
_remap_files recognizes it, attaches the underlying vine.File as a task
input under a fresh sandbox name, and substitutes that sandbox name into the
args actually sent to the task - so reducer_wrapper's
`serialization.load(path)` opens a name that exists in its own sandbox,
never the manager-side token. For a non-checkpoint (temp) result,
retrieve() is the only place its bytes are ever pulled to the manager, via
manager.fetch_file() + File.contents(); for a checkpoint, retrieve() still
works the same way but the bytes are already local by then.

A restart-seeded checkpoint (Pipeline._pool_item_from_checkpoint) has no
token - there was no live run to mint one in - so it shows up in submit()
args as its own real on-disk path instead. _remap_files declares that path
on demand and caches the vine.File under the path itself in _files_by_key,
the same dict dest_tokens live in, so a later submit() reusing the same path
(e.g. a ResourceExhaustion retry resubmitting the same group) reuses the
declaration rather than leaking a fresh one. release_path() is the
counterpart to release_result() for these path-keyed entries, since they
have no result_id for release_result() to be called with.

Resource exhaustion: monitoring is enabled with watchdog=True, so TaskVine
itself can kill and report a task that overruns its resource allocation -
something a plain ProcessPoolExecutor (see local_distributor.py) can't do.
wait() checks task.successful() first and only trusts the RawOutcome
returned by executor_wrapper/reducer_wrapper (a Python-level exception
caught inside the wrapper) when that's True; otherwise it translates
TaskVine's own result string into ResourceExhaustion or RuntimeFailure.
task.successful() means "the wrapper ran to completion and returned a
RawOutcome", not "that RawOutcome was a success" - a Python-level failure
or caught MemoryError still returns normally, and dest_file is always
written (see defaults.py's _run_and_wrap) precisely so TaskVine's own
missing-output check can't itself mark such a task unsuccessful and
discard the real RawOutcome. task.successful() is False only when the
wrapper never returned at all: it crashed outright (unhandled exception,
bug) or the worker process was killed by TaskVine's resource watchdog.
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import ndcctools.taskvine as vine

from .distributor import TaskKind
from .types import Outcome, RawOutcome, ResourceExhaustion, RuntimeFailure, Success

# TaskVine result strings (Task.result) that mean the task was killed for
# overrunning a resource allocation, as opposed to a genuine execution error.
_RESOURCE_EXHAUSTION_RESULTS = {"resource exhaustion", "max wall time", "disk alloc full"}

# resources_processor/resources_reducer use vine_reduce's own key names; this maps
# them onto the resource_monitor's rmsummary field names expected by
# Manager.set_category_resources_max.
_RESOURCE_KEY_TO_RMSUMMARY = {"cores": "cores", "memory_mb": "memory", "disk_mb": "disk"}


def _result_token(result_id: int) -> str:
    """The opaque manager-side name for a result (see module docstring). Derived
    from result_id rather than stored, so the two can never drift apart."""
    return f"result_{result_id}.p"


@dataclass
class _InFlight:
    result_id: int
    kind: TaskKind


class TaskVineDistributor:
    """A Distributor backed by ndcctools.taskvine, running vine_reduce
    across a cluster of TaskVine workers instead of local subprocesses. See
    the module docstring for how it bridges the Distributor protocol onto
    TaskVine's file/task model, and the README's "Packaging an environment
    for remote workers" for `environment=`."""

    def __init__(
        self,
        port: int | tuple[int, int] = 9123,
        name: str | None = None,
        resources_processor: dict[str, int] | None = None,
        resources_reducer: dict[str, int] | None = None,
        environment: str | None = None,
        manager: vine.Manager | None = None,
        checkpoint_dir: str = "checkpoints",
    ):
        """port: port (or [min, max] range) the manager listens on, or 0 to
        pick one automatically - see `port` below. name: the manager's
        TaskVine project name, for workers to find it by name instead of
        host:port. resources_processor/resources_reducer: per-category
        resource caps (e.g. {"cores": 1, "memory_mb": 2000, "disk_mb": 4000})
        applied to every processor/reducer call respectively, via
        Manager.set_category_resources_max. environment: path to a packed
        poncho package tarball (see get_environment() in
        remote_environment.py) to ship and activate on every worker task;
        None runs tasks in whatever Python environment the worker itself was
        started with. manager: an already-constructed vine.Manager (or
        subclass, e.g. vine.DaskVine) to use instead of building one from
        port/name - lets vine_reduce's tasks and a caller's own tasks share
        one manager/port and worker pool. checkpoint_dir: local directory
        (on this process's filesystem, i.e. wherever the manager runs) this
        distributor writes a result's file to when submit() is called with
        is_checkpoint=True - see the module docstring and checkpoint_path()."""
        # manager lets a caller hand in an already-constructed vine.Manager
        # (or a subclass, e.g. vine.DaskVine) instead of having this class
        # build its own - the way to run coffea's own preprocess() and this
        # distributor's tasks against the same manager/port, sharing workers
        # between the two. port/name are ignored when manager is given.
        self._owns_manager = manager is None
        self._manager = manager if manager is not None else vine.Manager(port=port, name=name)
        self._manager.enable_monitoring(watchdog=True)

        self._resources_by_kind: dict[TaskKind, dict[str, int]] = {
            "processor": resources_processor or {},
            "reducer": resources_reducer or {},
        }
        self._environment = self._manager.declare_poncho(environment) if environment else None

        self._checkpoint_dir = checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._next_id = itertools.count(1)
        # Keyed on whatever string _remap_files sees in task args: either a
        # dest_token this run minted (see submit(), below) or a restart-
        # seeded checkpoint path declared on demand (see _remap_files) - one
        # dict either way, since the two key spaces never collide (a token
        # is never an absolute, existing path) and both need the same
        # lookup-or-declare-once / undeclare-on-release treatment.
        self._files_by_key: dict[str, vine.File] = {}
        self._checkpoint_paths_by_token: dict[str, str] = {}
        self._in_flight_by_taskvine_id: dict[int, _InFlight] = {}
        self._categories_configured: set[str] = set()

        # Files/env vars added via add_file/set_env_var, attached to every
        # task submitted from then on - see those methods below.
        self._extra_files: list[tuple[str, vine.File]] = []
        self._extra_env: dict[str, str] = {}

    @property
    def port(self) -> int:
        """The manager's actual listening port - useful when `port=0` (or a
        range) was passed to __init__ and the resolved port is needed to
        point workers at this manager."""
        return self._manager.port

    def submit(
        self,
        priority: int,
        category: str,
        kind: TaskKind,
        func: Callable[..., Any],
        *args: Any,
        is_checkpoint: bool = False,
    ) -> int:
        """Submit func(dest_token, *args) as a vine.PythonTask, ordered by
        priority (larger runs first) and grouped under `category` for
        resource-limit purposes. kind selects resources_processor vs.
        resources_reducer the first time this category is seen.
        is_checkpoint declares the result durable (see module docstring):
        its file becomes a manager.declare_file(cache=True) under
        checkpoint_dir instead of an ordinary manager.declare_temp(), and
        its path becomes available via checkpoint_path() once the task
        succeeds. Returns a result_id."""
        result_id = next(self._next_id)
        dest_token = _result_token(result_id)

        remapped_args, extra_inputs = self._remap_files(args)

        task = vine.PythonTask(func, dest_token, *remapped_args)
        task.set_priority(priority)
        task.set_category(category)
        self._configure_category(category, kind)

        if self._environment is not None:
            task.add_environment(self._environment)

        for sandbox_name, vine_file in extra_inputs:
            task.add_input(vine_file, sandbox_name)

        for remote_name, vine_file in self._extra_files:
            task.add_input(vine_file, remote_name)

        for name, value in self._extra_env.items():
            task.set_env_var(name, value)

        if is_checkpoint:
            checkpoint_path = os.path.join(self._checkpoint_dir, dest_token)
            result_file = self._manager.declare_file(checkpoint_path, cache=True)
            self._checkpoint_paths_by_token[dest_token] = checkpoint_path
        else:
            result_file = self._manager.declare_temp()
        task.add_output(result_file, dest_token)

        taskvine_id = self._manager.submit(task)
        self._files_by_key[dest_token] = result_file
        self._in_flight_by_taskvine_id[taskvine_id] = _InFlight(result_id=result_id, kind=kind)
        return result_id

    def _configure_category(self, category: str, kind: TaskKind) -> None:
        """Apply resources_processor/resources_reducer to `category` in
        TaskVine, once, the first time that category is submitted to -
        category is a resource-allocation grouping in TaskVine, not a
        per-task setting."""
        if category in self._categories_configured:
            return
        limits = {
            _RESOURCE_KEY_TO_RMSUMMARY[key]: value
            for key, value in self._resources_by_kind[kind].items()
            if key in _RESOURCE_KEY_TO_RMSUMMARY
        }
        self._manager.set_category_resources_max(category, limits)
        self._categories_configured.add(category)

    def _remap_files(self, args: tuple[Any, ...]) -> tuple[list[Any], list[tuple[str, vine.File]]]:
        """Replace tokens from earlier Success outcomes - and restart-seeded
        checkpoint paths - with fresh sandbox names. Tokens/paths only ever
        appear as bare strings or inside a flat list of strings
        (reducer_wrapper's input_files), so this only looks one level deep
        rather than walking arbitrary nested structures."""
        extra_inputs: list[tuple[str, vine.File]] = []

        def remap_one(value: Any) -> Any:
            if isinstance(value, str) and value in self._files_by_key:
                sandbox_name = f"input_{len(extra_inputs)}"
                extra_inputs.append((sandbox_name, self._files_by_key[value]))
                return sandbox_name
            if isinstance(value, str) and os.path.isabs(value) and os.path.exists(value):
                # A restart-seeded checkpoint path (Pipeline._pool_item_from_
                # checkpoint): not a token this run ever minted, but a real
                # file already durable on this manager's disk - declare it on
                # demand, under its own path as the key, so a later reduction
                # that resubmits the same group (e.g. after a
                # ResourceExhaustion retry) hits the branch above instead of
                # declaring it again. release_path() undoes this once
                # Pipeline supersedes the checkpoint.
                sandbox_name = f"input_{len(extra_inputs)}"
                self._files_by_key[value] = self._manager.declare_file(value, cache=True)
                extra_inputs.append((sandbox_name, self._files_by_key[value]))
                return sandbox_name
            return value

        remapped: list[Any] = []
        for arg in args:
            if isinstance(arg, list):
                remapped.append([remap_one(value) for value in arg])
            else:
                remapped.append(remap_one(arg))
        return remapped, extra_inputs

    def wait(self, timeout: float | None = None) -> Outcome | None:
        """Block until a submitted task finishes, returning its Outcome
        (Success/RuntimeFailure/ResourceExhaustion, translated from
        TaskVine's own result string when the task didn't run its Python
        function to completion), or None if timeout elapses first."""
        # TaskVine's C API only accepts an integer number of seconds; round
        # up so a small positive float still waits at least that long
        # instead of truncating to 0 ("return immediately").
        vine_timeout = "wait_forever" if timeout is None else max(0, math.ceil(timeout))
        task = self._manager.wait(vine_timeout)
        if task is None:
            return None

        entry = self._in_flight_by_taskvine_id.pop(task.id)
        result_id, kind = entry.result_id, entry.kind

        if task.successful():
            raw: RawOutcome = task.output
            outcome = raw.to_outcome(result_id)
            if not isinstance(outcome, Success):
                # The wrapper ran to completion but reported a Python-level
                # failure/exhaustion (see defaults.py's _run_and_wrap) -
                # dest_file exists (it's always written, even on failure, so
                # TaskVine doesn't itself report "output missing" and
                # discard this very outcome) but is just a placeholder,
                # and vine_reduce only ever calls release_result() for a
                # Success - so drop it here, or it would leak for the rest
                # of the run.
                self.release_result(result_id)
            return outcome

        # A task that didn't run to completion at all (crashed before
        # returning, or was killed by TaskVine's own resource watchdog) has
        # no result to hand back, and vine_reduce only ever calls
        # release_result() for a Success - so drop the file declared for it
        # here, or it would leak for the rest of the run (a resource-
        # exhausted chunk, say, is simply retried).
        self.release_result(result_id)
        resources = self._resources_from_task(task, kind)
        if task.result in _RESOURCE_EXHAUSTION_RESULTS:
            return ResourceExhaustion(result_id=result_id, resources=resources)
        return RuntimeFailure(
            result_id=result_id,
            resources=resources,
            traceback=f"taskvine result: {task.result}\n{task.std_output}",
        )

    def _resources_from_task(self, task: vine.Task, kind: TaskKind) -> dict[str, Any]:
        default_cores = self._resources_by_kind[kind].get("cores", 1)
        measured = task.resources_measured
        if measured is None:
            return {"cores": default_cores, "memory_mb": 0.0, "wall_time_s": 0.0}
        return {
            "cores": measured.cores or default_cores,
            "memory_mb": measured.memory or 0.0,
            "wall_time_s": (measured.wall_time or 0) / 1e6,
        }

    def _undeclare(self, key: str) -> None:
        """Drop and undeclare the vine.File cached under `key` in
        _files_by_key, if any - shared by release_result (key: a result
        token) and release_path (key: a restart-seeded checkpoint path)."""
        file = self._files_by_key.pop(key, None)
        if file is not None:
            self._manager.undeclare_file(file)

    def release_result(self, result_id: int) -> None:
        """Undeclare the vine.File backing a completed (Success) result_id,
        letting TaskVine reclaim its storage on the worker(s) holding it,
        and remove its checkpoint_dir mirror from local disk, if it has one
        (see submit's is_checkpoint) - either because a further checkpoint
        has superseded it, or because a final result was safely retrieved
        elsewhere and no longer needs this durable copy."""
        token = _result_token(result_id)
        self._undeclare(token)
        checkpoint_path = self._checkpoint_paths_by_token.pop(token, None)
        if checkpoint_path is not None:
            try:
                os.remove(checkpoint_path)
            except FileNotFoundError:
                pass

    def release_path(self, path: str) -> None:
        """Undeclare the vine.File cached (see _remap_files) for a restart-
        seeded checkpoint path, once Pipeline has removed the file itself
        from disk because a later checkpoint superseded it - the on-demand
        counterpart to release_result, for items with no result_id of their
        own. A no-op if `path` was never declared on demand (e.g. it's a
        this-run checkpoint's path, whose declaration lives under its token
        instead and is already released via release_result)."""
        self._undeclare(path)

    def checkpoint_path(self, result_id: int) -> str:
        """Local, durable on-disk path for a completed (Success) result_id
        that was submitted with is_checkpoint=True - see submit(). TaskVine
        already wrote the file there as part of retrieving the task's
        outputs, so this is a lookup, not a copy."""
        return self._checkpoint_paths_by_token[_result_token(result_id)]

    def capacity(self) -> int:
        """How many more tasks the manager's connected workers could
        currently run, per TaskVine's own Manager.hungry()."""
        return self._manager.hungry()

    def retrieve(self, result_id: int, dest_path: str) -> None:
        """Pull the result file for a completed (Success) result_id back to
        the manager and write it to dest_path."""
        file = self._files_by_key[_result_token(result_id)]
        self._manager.fetch_file(file)
        with open(dest_path, "wb") as f:
            f.write(file.contents())

    def add_file(self, local_path: str, remote_path: str | None = None) -> None:
        """Declare local_path once, and attach it as an input - under
        remote_path, defaulting to local_path's basename - to every task
        submitted from now on."""
        remote_name = remote_path or os.path.basename(local_path)
        self._extra_files.append((remote_name, self._manager.declare_file(local_path)))

    def set_env_var(self, name: str, value: str) -> None:
        """Set an environment variable on every task submitted from now on."""
        self._extra_env[name] = value

    def shutdown(self) -> None:
        """Workers are owned by the caller, not this distributor, so there is
        nothing to do about them. The vine.Manager itself, though, is this
        distributor's own to close if it built it (manager= was not passed
        to __init__): drop this class's reference to it so its listening
        port is freed right away, rather than only whenever the whole
        TaskVineDistributor object eventually gets garbage collected. A
        caller-supplied manager is left alone - it is the caller's to close."""
        if self._owns_manager:
            self._manager = None

    def __enter__(self) -> "TaskVineDistributor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
