"""A Distributor backed by ndcctools.taskvine, running vine_reduce across a cluster.

Manager-only: this starts a vine.Manager; workers (vine_worker, a factory, batch submission) are
the caller's responsibility.

Files: Outcome.file is not a real location but an opaque token minted from result_id (see
_result_token). An ordinary result is a manager.declare_temp() file, which stays on the producing
worker; retrieve() is the only place its bytes are pulled to the manager. A checkpoint or final
result (is_checkpoint=True) is a manager.declare_file(path, cache=True) file with `path` inside
checkpoint_dir; TaskVine transfers such outputs back immediately, so it is durably on local disk by
the time wait() reports success. When a token appears in a later submit()'s args, _remap_files
attaches the underlying vine.File as a task input under a fresh sandbox name and substitutes that
name, so the task opens a file that exists in its own sandbox. An adopted checkpoint
(adopt_checkpoint) is declared under a token like any other and needs no special-casing.

Resource exhaustion: monitoring runs with watchdog=True, so TaskVine itself kills tasks that overrun
their allocation. wait() trusts the Outcome returned by executor_wrapper/reducer_wrapper only when
task.successful(), which means "the wrapper returned an Outcome" (even a Python-level failure or
caught MemoryError), not "that Outcome was a Success". dest_file is always written even on failure
(see defaults.py's _run_and_wrap) so TaskVine's missing-output check can't mark such a task
unsuccessful and discard the real Outcome. task.successful() is False only when the wrapper never
returned (crash, or the worker was killed by the watchdog); wait() then maps TaskVine's result
string to ResourceExhaustion or RuntimeFailure.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Callable
from uuid import uuid4

import ndcctools.taskvine as vine

from .distributor import Distributor, TaskKind
from .types import Outcome, ResourceExhaustion, ResourceUsage, RuntimeFailure, Success

# TaskVine result strings (Task.result) meaning the task overran a resource allocation, as opposed
# to a genuine execution error.
_RESOURCE_EXHAUSTION_RESULTS = {
    "resource exhaustion",
    "max wall time",
    "sandbox exhaustion",
    "max end time",
}


def _result_token(result_id: str) -> str:
    """The opaque manager-side name for a result, derived from result_id so they can't drift."""
    return f"result_{result_id}.p"


class TaskVineDistributor(Distributor):
    """Runs calls as TaskVine tasks on a cluster of workers. See the module docstring for the
    file model and the README's "Packaging an environment for remote workers" for `environment=`."""

    def __init__(
        self,
        port: int | tuple[int, int] = 9123,
        name: str | None = None,
        resources_processor: dict[str, int] | None = None,
        resources_reducer: dict[str, int] | None = None,
        environment: str | None = None,
        manager: vine.Manager | None = None,
        ssl: bool = True,
    ):
        """port: port (or [min, max] range) the manager listens on, or 0 to pick one.
        name: the manager's project name, for workers to find it by name instead of host:port.
        resources_processor/resources_reducer: per-category resource caps (e.g. {"cores": 1,
            "memory_mb": 2000, "disk_mb": 4000}) applied via Manager.set_category_resources_max.
        environment: path to a packed poncho tarball (see get_environment() in
            remote_environment.py) to activate on every task; None uses the worker's own Python.
        manager: an existing vine.Manager (or subclass, e.g. vine.DaskVine) to use instead of
            building one, so vine_reduce's tasks and the caller's share a manager and worker pool.
        ssl: whether the manager encrypts worker connections with a generated self-signed cert;
            ignored when manager is given. A vine.Factory(manager=...) picks this up
            automatically; one started with manager_host_port= needs `ssl=True` too.
        """
        self._owns_manager = manager is None
        self._manager = (
            manager if manager is not None else vine.Manager(port=port, name=name, ssl=ssl)
        )
        self._manager.enable_monitoring(watchdog=True)

        # Every task is tagged with this and wait() only waits on this tag, so on a shared
        # manager= it never picks up the caller's own tasks.
        self._tag = f"vine_reduce_{uuid4().hex}"

        if self._owns_manager:
            self._manager.tune("category-steady-n-tasks", 1)
            self._manager.tune("hungry-minimum", 100)
            self._manager.tune("prefer-dispatch", 1)
            self._manager.tune("temp-replica-count", 3)

        self._resources_by_kind: dict[TaskKind, dict[str, int]] = {
            TaskKind.PROCESSOR: resources_processor or {},
            TaskKind.REDUCER: resources_reducer or {},
        }
        self._environment = (
            self._manager.declare_poncho(environment, cache=True) if environment else None
        )

        self._checkpoint_dir: str | None = None

        # Keyed on the token minted for every result (by submit() or adopt_checkpoint()).
        self._files_by_key: dict[str, vine.File] = {}
        self._checkpoint_paths_by_token: dict[str, str] = {}
        self._result_ids_by_taskvine_id: dict[int, str] = {}
        self._categories_configured: set[str] = set()

        # Attached to every task submitted after add_file/set_env_var.
        self._extra_files: list[tuple[str, vine.File]] = []
        self._extra_env: dict[str, str] = {}

        # Set once wait() returns a task, so the next call polls with timeout 0 to pick up a burst
        # of already-finished tasks before waiting for real.
        self._drain_next_wait = False

    @property
    def port(self) -> int:
        """The manager's actual listening port (resolved when `port=0` or a range was given)."""
        return self._manager.port

    @property
    def manager(self) -> vine.Manager:
        """The underlying vine.Manager; pass it to vine.Factory(manager=...) so the factory picks
        up host, port, and settings like ssl from it."""
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
        """Submit func(dest_token, *args) as a vine.PythonTask. `category` is the TaskVine
        resource category; kind selects the resource caps the first time it is seen.
        is_checkpoint makes the result a durable declare_file under checkpoint_dir instead of a
        declare_temp (see module docstring)."""
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
        self._result_ids_by_taskvine_id[taskvine_id] = result_id

    def resources(self, kind: TaskKind) -> dict[str, Any] | None:
        """The configured resources_processor/resources_reducer dict for kind. This is the
        category cap; a task's real allocation may be less and is exposed on the worker via the
        CORES environment variable (see DaskExecutor's _num_workers)."""
        return self._resources_by_kind.get(kind) or None

    def _configure_category(self, category: str, kind: TaskKind) -> None:
        """Once per category, apply the resource caps for `kind` and the automatic allocation
        mode: "max" for reducers (largest usage measured so far), "min waste" for processors."""
        if category in self._categories_configured:
            return
        caps = self._resources_by_kind[kind]
        limits = ResourceUsage(
            cores=caps.get("cores"), memory_mb=caps.get("memory_mb"), disk_mb=caps.get("disk_mb")
        ).to_rmsummary()
        mode = "max" if kind == TaskKind.REDUCER else "min waste"
        self._manager.set_category_mode(category, mode)
        self._manager.set_category_resources_max(category, limits)
        self._categories_configured.add(category)

    def _remap_files(self, args: tuple[Any, ...]) -> tuple[list[Any], list[tuple[str, vine.File]]]:
        """Replace result tokens in args with fresh sandbox names and return the vine.Files to
        attach under them. Tokens only appear as bare strings or in a flat list of strings
        (reducer_wrapper's input_files), so only one level is searched."""
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
        """Block until a task finishes and return its Outcome, or None on timeout. A timeout of
        None means 5 seconds, except right after a previous wait() returned a task, when it is 0
        to drain already-finished tasks."""
        if self._drain_next_wait:
            vine_timeout = 0
        elif timeout is None:
            vine_timeout = 5
        else:
            # TaskVine only accepts whole seconds; round up so a small positive timeout doesn't
            # truncate to 0 ("return immediately").
            vine_timeout = max(0, math.ceil(timeout))

        task = self._manager.wait_for_tag(self._tag, vine_timeout)
        if task is None:
            self._drain_next_wait = False
            return None

        self._drain_next_wait = True

        result_id = self._result_ids_by_taskvine_id.pop(task.id)
        allocated = self._allocated_from_task(task)

        if task.successful():
            raw = task.output
            if not isinstance(raw, Outcome):
                # Loading the task's output failed on the manager side, so PythonTask.output is
                # the exception, not an Outcome. Report it rather than let wait() raise.
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
                # A Python-level failure/exhaustion: dest_file is just a placeholder, and
                # release_result() is only called for a Success, so drop it here or it leaks.
                self.release_result(result_id)
            return outcome

        # The wrapper never returned, so there is no result. Drop the declared file here since
        # release_result() is only called for a Success.
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

    def _measured_from_task(self, task: vine.Task) -> ResourceUsage:
        """Usage measured for this task, or all zeros if TaskVine has no monitor data for it."""
        summary = task.resources_measured
        if summary is None:
            return ResourceUsage(cores=0.0, memory_mb=0.0, wall_time_s=0.0)
        return ResourceUsage(
            cores=summary.cores or 0.0,
            memory_mb=summary.memory or 0.0,
            wall_time_s=(summary.wall_time or 0) / 1e6,
        )

    def _allocated_from_task(self, task: vine.Task) -> dict[str, Any] | None:
        """What TaskVine allocated on this task's latest attempt, which automatic allocation may
        make differ from the category cap. None if unknown; fields with no known allocation are
        omitted rather than reported as zero."""
        summary = task.resources_allocated
        if summary is None:
            return None
        allocated: dict[str, Any] = {}
        if summary.cores:
            allocated["cores"] = summary.cores
        if summary.memory:
            allocated["memory_mb"] = summary.memory
        if summary.wall_time:
            allocated["wall_time_s"] = summary.wall_time / 1e6
        return allocated

    def release_result(self, result_id: str) -> None:
        """Undeclare the vine.File for result_id so TaskVine reclaims worker storage, and delete
        its durable checkpoint_dir copy if it has one."""
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
        """Declare `path` under result_id's token exactly as a this-run checkpoint would be.
        Returns the token."""
        token = _result_token(result_id)
        self._files_by_key[token] = self._manager.declare_file(path, cache=True)
        self._checkpoint_paths_by_token[token] = path
        return token

    def checkpoint_path(self, result_id: str) -> str:
        """A lookup, not a copy: TaskVine already wrote the file there when retrieving outputs."""
        return self._checkpoint_paths_by_token[_result_token(result_id)]

    def set_checkpoint_dir(self, path: str) -> None:
        """Directory checkpoint results land under from now on. Must precede any
        is_checkpoint=True submit()."""
        self._checkpoint_dir = path
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def capacity(self) -> int:
        """How many more tasks the connected workers could run now, per Manager.hungry()."""
        return self._manager.hungry()

    def retrieve(self, result_id: str, dest_path: str) -> None:
        """Fetch the result file to the manager and write it to dest_path."""
        file = self._files_by_key[_result_token(result_id)]
        self._manager.fetch_file(file)
        with open(dest_path, "wb") as f:
            f.write(file.contents())

    def add_file(self, local_path: str, remote_path: str | None = None) -> None:
        """Attach local_path, as remote_path (default: its basename), to every later task."""
        remote_name = remote_path or os.path.basename(local_path)
        self._extra_files.append((remote_name, self._manager.declare_file(local_path)))

    def set_env_var(self, name: str, value: str) -> None:
        """Set an environment variable on every later task."""
        self._extra_env[name] = value

    def shutdown(self) -> None:
        """Workers belong to the caller. If this distributor built the manager, drop it so its
        port is freed now rather than at garbage collection; a caller-supplied manager is left
        alone."""
        if self._owns_manager:
            self._manager = None
