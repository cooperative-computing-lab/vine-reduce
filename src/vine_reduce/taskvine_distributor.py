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
mints from result_id, e.g. "result_3f9a1c...p". When that token later shows up inside another
submit() call's args (as one of reducer_wrapper's input_files),
_remap_files recognizes it, attaches the underlying vine.File as a task
input under a fresh sandbox name, and substitutes that sandbox name into the
args actually sent to the task - so reducer_wrapper's
`serialization.load(path)` opens a name that exists in its own sandbox,
never the manager-side token. For a non-checkpoint (temp) result,
retrieve() is the only place its bytes are ever pulled to the manager, via
manager.fetch_file() + File.contents(); for a checkpoint, retrieve() still
works the same way but the bytes are already local by then.

A restart-seeded checkpoint is adopted via adopt_checkpoint() before it ever
appears in a submit() call: adoption mints a token and declares the file
under it exactly like a this-run checkpoint (see adopt_checkpoint below), so
from then on it flows through the normal token path in _remap_files -
remapping, release, retrieve - with no special-casing.

Resource exhaustion: monitoring is enabled with watchdog=True, so TaskVine
itself can kill and report a task that overruns its resource allocation -
something a plain ProcessPoolExecutor (see local_distributor.py) can't do.
wait() checks task.successful() first and only trusts the Outcome returned
by executor_wrapper/reducer_wrapper (a Python-level exception caught inside
the wrapper) when that's True; otherwise it translates TaskVine's own result
string into ResourceExhaustion or RuntimeFailure. task.successful() means
"the wrapper ran to completion and returned an Outcome", not "that Outcome
was a Success" - a Python-level failure or caught MemoryError still returns
normally, and dest_file is always written (see defaults.py's _run_and_wrap)
precisely so TaskVine's own missing-output check can't itself mark such a
task unsuccessful and discard the real Outcome. task.successful() is False
only when the wrapper never returned at all: it crashed outright (unhandled
exception, bug) or the worker process was killed by TaskVine's resource
watchdog.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

import ndcctools.taskvine as vine

from .distributor import Distributor, TaskKind
from .types import Outcome, ResourceExhaustion, RuntimeFailure, Success

# TaskVine result strings (Task.result) that mean the task was killed for
# overrunning a resource allocation, as opposed to a genuine execution error.
_RESOURCE_EXHAUSTION_RESULTS = {
    "resource exhaustion",
    "max wall time",
    "sandbox exhaustion",
    "max end time",
}

# resources_processor/resources_reducer use vine_reduce's own key names; this maps
# them onto the resource_monitor's rmsummary field names expected by
# Manager.set_category_resources_max.
_RESOURCE_KEY_TO_RMSUMMARY = {"cores": "cores", "memory_mb": "memory", "disk_mb": "disk"}


def _result_token(result_id: str) -> str:
    """The opaque manager-side name for a result (see module docstring). Derived
    from result_id rather than stored, so the two can never drift apart."""
    return f"result_{result_id}.p"


@dataclass
class _InFlight:
    result_id: str
    kind: TaskKind


class TaskVineDistributor(Distributor):
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
        ssl: bool = True,
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
        is_checkpoint=True - see the module docstring and checkpoint_path().
        ssl: whether the manager encrypts its connections to workers, via a
        self-signed cert vine.Manager generates on the fly; ignored when
        manager is given (that manager's own ssl setting, if any, applies
        instead). Workers started with vine.Factory(manager=...) pick this
        up automatically (Factory reads it off the manager); Factory started
        with manager_host_port= instead needs `ssl=True` passed to it too."""
        self._owns_manager = manager is None
        self._manager = (
            manager if manager is not None else vine.Manager(port=port, name=name, ssl=ssl)
        )
        self._manager.enable_monitoring(watchdog=True)

        # Every task this distributor submits is tagged with this unique
        # value, and wait() only ever waits for that tag (via
        # Manager.wait_for_tag) rather than Manager.wait()/wait_for_tag(None)
        # - so on a caller-supplied manager= shared with the caller's own
        # tasks (see manager= above), wait() can never pick up one of the
        # caller's tasks and KeyError on _in_flight_by_taskvine_id.
        self._tag = f"vine_reduce_{uuid4().hex}"

        if self._owns_manager:
            self._manager.tune("category-steady-n-tasks", 2)
            self._manager.tune("hungry-minimum", 100)
            self._manager.tune("prefer-dispatch", 1)
            self._manager.tune("temp-replica-count", 3)

        self._resources_by_kind: dict[TaskKind, dict[str, int]] = {
            "processor": resources_processor or {},
            "reducer": resources_reducer or {},
        }
        self._environment = (
            self._manager.declare_poncho(environment, cache=True) if environment else None
        )

        self._checkpoint_dir = checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        # Keyed on the dest_token minted for every result - by submit() for
        # a this-run result, or by adopt_checkpoint() for a restart-seeded
        # one - so _remap_files always finds it by simple lookup.
        self._files_by_key: dict[str, vine.File] = {}
        self._checkpoint_paths_by_token: dict[str, str] = {}
        self._in_flight_by_taskvine_id: dict[int, _InFlight] = {}
        self._categories_configured: set[str] = set()

        # Files/env vars added via add_file/set_env_var, attached to every
        # task submitted from then on - see those methods below.
        self._extra_files: list[tuple[str, vine.File]] = []
        self._extra_env: dict[str, str] = {}

        # Whether last wait received a task. If yes, we set timeout to 0
        # to try to receive as many tasks as possible.
        self._task_last_wait = False

    @property
    def port(self) -> int:
        """The manager's actual listening port - useful when `port=0` (or a
        range) was passed to __init__ and the resolved port is needed to
        point workers at this manager."""
        return self._manager.port

    @property
    def manager(self) -> vine.Manager:
        """The underlying vine.Manager - pass this to vine.Factory(manager=...)
        so it provisions workers against the right host/port and picks up
        settings (e.g. ssl) straight from the manager, rather than
        duplicating them via manager_host_port=."""
        return self._manager

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
        """Submit func(dest_token, *args) as a vine.PythonTask, identified by
        result_id (see the Distributor protocol docstring), ordered by
        priority (larger runs first) and grouped under `category` for
        resource-limit purposes. kind selects resources_processor vs.
        resources_reducer the first time this category is seen.
        is_checkpoint declares the result durable (see module docstring):
        its file becomes a manager.declare_file(cache=True) under
        checkpoint_dir instead of an ordinary manager.declare_temp(), and
        its path becomes available via checkpoint_path() once the task
        succeeds."""
        dest_token = _result_token(result_id)

        remapped_args, extra_inputs = self._remap_files(args)

        task = vine.PythonTask(func, dest_token, *remapped_args)
        task.set_tag(self._tag)
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
            checkpoint_path = os.path.join(self._checkpoint_dir, f"{uuid4().hex}.p")
            result_file = self._manager.declare_file(checkpoint_path, cache=True)
            self._checkpoint_paths_by_token[dest_token] = checkpoint_path
        else:
            result_file = self._manager.declare_temp()
        task.add_output(result_file, dest_token)

        taskvine_id = self._manager.submit(task)
        self._files_by_key[dest_token] = result_file
        self._in_flight_by_taskvine_id[taskvine_id] = _InFlight(result_id=result_id, kind=kind)

    def resources(self, kind: TaskKind) -> dict[str, Any] | None:
        """The configured resources_processor/resources_reducer dict for
        kind, e.g. {"cores": ...} - see the Distributor protocol docstring.
        This is the category's cap, not necessarily what a given task
        actually gets; TaskVine's own allocation, decided at dispatch time,
        may be less, and takes precedence via the CORES environment variable
        it sets on the worker (see DaskExecutor's _num_workers)."""
        return self._resources_by_kind.get(kind) or None

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
        self._manager.set_category_mode(category, "max")
        self._categories_configured.add(category)

    def _remap_files(self, args: tuple[Any, ...]) -> tuple[list[Any], list[tuple[str, vine.File]]]:
        """Replace tokens from earlier Success outcomes (this-run or
        adopted, see adopt_checkpoint) with fresh sandbox names. Tokens only
        ever appear as bare strings or inside a flat list of strings
        (reducer_wrapper's input_files), so this only looks one level deep
        rather than walking arbitrary nested structures."""
        extra_inputs: list[tuple[str, vine.File]] = []

        def remap_one(value: Any) -> Any:
            if isinstance(value, str) and value in self._files_by_key:
                sandbox_name = f"input_{len(extra_inputs)}"
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
        function to completion), or None if timeout elapses first.
        If timeout is None, a default of 5 seconds is used."""
        # TaskVine's C API only accepts an integer number of seconds; round
        # up so a small positive float still waits at least that long
        # instead of truncating to 0 ("return immediately").
        if self._task_last_wait is True:
            vine_timeout = 0
        else:
            if timeout is None:
                vine_timeout = 5
            else:
                vine_timeout = max(0, math.ceil(timeout))

        task = self._manager.wait_for_tag(self._tag, vine_timeout)
        if task is None:
            self._task_last_wait = False
            return None

        self._task_last_wait = True

        entry = self._in_flight_by_taskvine_id.pop(task.id)
        result_id, kind = entry.result_id, entry.kind
        allocated = self._allocated_from_task(task)

        if task.successful():
            raw = task.output
            if not isinstance(raw, Outcome):
                # cloudpickle.load of the task's output failed on the
                # manager side; PythonTask.output then hands back the
                # exception object it raised, not an Outcome - task-level
                # (the wrapper really did run and return something), not a
                # reason to let wait() itself raise.
                self.release_result(result_id)
                return RuntimeFailure(
                    result_id=result_id,
                    resources=self._measured_from_task(task),
                    resources_allocated=allocated,
                    std_output=task.std_output,
                    traceback=f"failed to load task output: {raw!r}",
                )
            outcome = dataclasses.replace(
                raw, result_id=result_id, std_output=task.std_output, resources_allocated=allocated
            )
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
        resources = self._measured_from_task(task)
        if task.result in _RESOURCE_EXHAUSTION_RESULTS:
            return ResourceExhaustion(
                result_id=result_id,
                resources=resources,
                resources_allocated=allocated,
                std_output=task.std_output,
            )
        return RuntimeFailure(
            result_id=result_id,
            resources=resources,
            resources_allocated=allocated,
            std_output=task.std_output,
            traceback=f"taskvine result: {task.result}\n{task.std_output}",
        )

    def _measured_from_task(self, task: vine.Task) -> dict[str, Any]:
        """Usage actually measured for this task, or an all-zero placeholder
        if TaskVine has no resource_monitor data for it (e.g. it crashed
        before monitoring even started)."""
        return self._rmsummary_to_dict(task.resources_measured) or {
            "cores": 0.0,
            "memory_mb": 0.0,
            "wall_time_s": 0.0,
        }

    def _allocated_from_task(self, task: vine.Task) -> dict[str, Any] | None:
        """What TaskVine actually allocated this task on its latest attempt
        - may differ from the category's static resources_processor/
        resources_reducer cap once automatic resource allocation ("max"
        mode - see _configure_category) starts adapting per task from
        measured history. None if TaskVine has no allocation info for it,
        so the caller falls back to that static cap (see Pipeline.
        _report_task) instead of a misleading zero."""
        return self._rmsummary_to_dict(task.resources_allocated)

    @staticmethod
    def _rmsummary_to_dict(summary: Any) -> dict[str, Any] | None:
        if summary is None:
            return None
        return {
            "cores": summary.cores or 0.0,
            "memory_mb": summary.memory or 0.0,
            "wall_time_s": (summary.wall_time or 0) / 1e6,
        }

    def release_result(self, result_id: str) -> None:
        """Undeclare the vine.File backing a completed (Success) result_id,
        letting TaskVine reclaim its storage on the worker(s) holding it,
        and remove its checkpoint_dir mirror from local disk, if it has one
        (see submit's is_checkpoint) - either because a further checkpoint
        has superseded it, or because a final result was safely retrieved
        elsewhere and no longer needs this durable copy."""
        token = _result_token(result_id)
        file = self._files_by_key.pop(token, None)
        if file is not None:
            self._manager.undeclare_file(file)
        checkpoint_path = self._checkpoint_paths_by_token.pop(token, None)
        if checkpoint_path is not None:
            try:
                os.remove(checkpoint_path)
            except FileNotFoundError:
                pass

    def adopt_checkpoint(self, result_id: str, path: str) -> str:
        """Register an existing durable checkpoint file at `path` under
        result_id, as if it were a completed Success result submitted with
        is_checkpoint=True - see the Distributor protocol docstring. Mints a
        token from result_id the same way submit() does, declares `path`
        under that token (cache=True, same as a this-run checkpoint), and
        records it in _checkpoint_paths_by_token - from then on the adopted
        item flows through every existing code path (remapping, release,
        retrieve) with no special cases. Returns the token."""
        token = _result_token(result_id)
        self._files_by_key[token] = self._manager.declare_file(path, cache=True)
        self._checkpoint_paths_by_token[token] = path
        return token

    def checkpoint_path(self, result_id: str) -> str:
        """Local, durable on-disk path for a completed (Success) result_id
        that was submitted with is_checkpoint=True - see submit(). TaskVine
        already wrote the file there as part of retrieving the task's
        outputs, so this is a lookup, not a copy."""
        return self._checkpoint_paths_by_token[_result_token(result_id)]

    def capacity(self) -> int:
        """How many more tasks the manager's connected workers could
        currently run, per TaskVine's own Manager.hungry()."""
        return self._manager.hungry()

    def retrieve(self, result_id: str, dest_path: str) -> None:
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
