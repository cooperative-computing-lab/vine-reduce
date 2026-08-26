"""A Distributor backed by ndcctools.taskvine, for running vine_reduce across
a real cluster of machines instead of local subprocesses.

Manager-only: this class starts a vine.Manager and nothing else. Worker
processes (vine_worker, a factory, batch-system submission, ...) are the
caller's responsibility, same as any other TaskVine application - see
https://cctools.readthedocs.io/en/latest/taskvine/.

Bridging the Distributor protocol's plain file-path strings onto TaskVine's
file model (see distributor.py's docstring on what "file" means) works like
this: every result gets a manager.declare_temp() file, which TaskVine keeps
at/near the worker that produced it rather than pulling it back to the
manager. Outcome.file is not that file's real location (there isn't one
`vine_reduce` can read directly) but an opaque token this class mints, e.g.
"result_7.p". When that token later shows up inside another submit() call's
args (as one of reducer_wrapper's input_files), _remap_files recognizes it,
attaches the underlying vine.File as a task input under a fresh sandbox name,
and substitutes that sandbox name into the args actually sent to the task -
so reducer_wrapper's `serialization.load(path)` opens a name that exists in
its own sandbox, never the manager-side token. retrieve() is the only place
a result's bytes are actually pulled to the manager, via
manager.fetch_file() + File.contents().

Resource exhaustion: monitoring is enabled with watchdog=True, so TaskVine
itself can kill and report a task that overruns its resource allocation -
something a plain ProcessPoolExecutor (see local_distributor.py) can't do.
wait() checks task.successful() first and only trusts the RawOutcome
returned by executor_wrapper/reducer_wrapper (a Python-level exception
caught inside the wrapper) when that's True; otherwise it translates
TaskVine's own result string into ResourceExhaustion or RuntimeFailure.
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import ndcctools.taskvine as vine

from .distributor import TaskKind
from .types import Outcome, RawOutcome, ResourceExhaustion, RuntimeFailure

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
        one manager/port and worker pool."""
        # manager lets a caller hand in an already-constructed vine.Manager
        # (or a subclass, e.g. vine.DaskVine) instead of having this class
        # build its own - the way to run coffea's own preprocess() and this
        # distributor's tasks against the same manager/port, sharing workers
        # between the two. port/name are ignored when manager is given.
        self._manager = manager if manager is not None else vine.Manager(port=port, name=name)
        self._manager.enable_monitoring(watchdog=True)

        self._resources_by_kind: dict[TaskKind, dict[str, int]] = {
            "processor": resources_processor or {},
            "reducer": resources_reducer or {},
        }
        self._environment = self._manager.declare_poncho(environment) if environment else None

        self._next_id = itertools.count(1)
        self._files_by_token: dict[str, vine.File] = {}
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
        self, priority: int, category: str, kind: TaskKind, func: Callable[..., Any], *args: Any
    ) -> int:
        """Submit func(dest_token, *args) as a vine.PythonTask, ordered by
        priority (larger runs first) and grouped under `category` for
        resource-limit purposes. kind selects resources_processor vs.
        resources_reducer the first time this category is seen. Returns a
        result_id."""
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

        result_file = self._manager.declare_temp()
        task.add_output(result_file, dest_token)

        taskvine_id = self._manager.submit(task)
        self._files_by_token[dest_token] = result_file
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

    def _remap_files(
        self, args: tuple[Any, ...]
    ) -> tuple[list[Any], list[tuple[str, "vine.File"]]]:
        """Replace tokens from earlier Success outcomes with fresh sandbox
        names. Tokens only ever appear as bare strings or inside a flat list
        of strings (reducer_wrapper's input_files), so this only looks one
        level deep rather than walking arbitrary nested structures."""
        extra_inputs: list[tuple[str, vine.File]] = []

        def remap_one(value: Any) -> Any:
            if isinstance(value, str) and value in self._files_by_token:
                sandbox_name = f"input_{len(extra_inputs)}"
                extra_inputs.append((sandbox_name, self._files_by_token[value]))
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
            return raw.to_outcome(result_id)

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

    def release_result(self, result_id: int) -> None:
        """Undeclare the vine.File backing a completed (Success) result_id,
        letting TaskVine reclaim its storage on the worker(s) holding it."""
        file = self._files_by_token.pop(_result_token(result_id), None)
        if file is not None:
            self._manager.undeclare_file(file)

    def capacity(self) -> int:
        """How many more tasks the manager's connected workers could
        currently run, per TaskVine's own Manager.hungry()."""
        return self._manager.hungry()

    def retrieve(self, result_id: int, dest_path: str) -> None:
        """Pull the result file for a completed (Success) result_id back to
        the manager and write it to dest_path."""
        file = self._files_by_token[_result_token(result_id)]
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
        """No-op: workers are owned by the caller, not this distributor, so
        there is nothing here to tear down beyond letting the vine.Manager be
        garbage collected."""

    def __enter__(self) -> "TaskVineDistributor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
