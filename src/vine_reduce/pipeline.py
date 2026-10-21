"""Per-(processor, dataset) state: chunk generation, the reduction pool, and
checkpoint bookkeeping. See "Temporary Results, Checkpoints, and Restart" in
PLAN.md for the pooling and checkpointing rules this implements.
"""

from __future__ import annotations

import enum
import json
import os
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Iterator, NoReturn, Protocol
from uuid import uuid4

from . import defaults, serialization
from .checkpoint_store import CheckpointRecord, CheckpointStore
from .distributor import Distributor, TaskKind
from .event_ranges import Coverage, Ranges, covers_all, uncovered_chunks, union_coverage
from .failure_log import FailureLog, FailureRecord
from .size_log import SizeLog, SizeRecord
from .types import (
    Chunk,
    Outcome,
    ResourceExhaustion,
    ResourceUsage,
    ResultHandle,
    RuntimeFailure,
    Success,
)

if TYPE_CHECKING:
    from .engine import VineReduce


class VineReduceError(RuntimeError):
    """Raised when a processing or reduction function fails remotely. Carries
    the remote traceback so the failure can be debugged from the local side."""


class ReportKind(enum.StrEnum):
    """What a reported task was: a processor call, or a reduction that is a
    plain accumulation, a checkpoint, a final result, or the forced-final
    drain group."""

    PROCESSOR = "processor"
    REDUCER = "reducer"
    CHECKPOINT = "checkpoint"
    RESULT = "result"
    FINAL = "final"


class OutcomeStatus(enum.StrEnum):
    SUCCESS = "success"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    FAILURE = "failure"


@dataclass(frozen=True)
class TaskReport:
    """One finished processor/reducer task, reported before Pipeline acts on
    its Outcome. See progress.py's ProgressReporter."""

    processor_name: str
    dataset_name: str
    kind: ReportKind
    result_id: str
    description: str
    status: OutcomeStatus
    resources: ResourceUsage
    resources_allocated: dict[str, Any] | None
    std_output: str | None


class TaskReporter(Protocol):
    """What a Pipeline reports finished tasks to - see progress.py's
    ProgressReporter and NullProgressReporter."""

    def report(self, task: TaskReport) -> None:
        """Called once per finished task, whatever its outcome."""
        ...


def _resolve_sized_config(
    config: int | dict | None, processor_name: str, dataset_name: str
) -> int | None:
    """Most-specific-wins lookup for chunksize/reduction_size: a "datasets"
    entry beats a "processors" entry, which beats "default"."""
    if config is None or isinstance(config, int):
        return config
    if dataset_name in config.get("datasets", {}):
        return config["datasets"][dataset_name]
    if processor_name in config.get("processors", {}):
        return config["processors"][processor_name]
    return config.get("default")


def _resolve_reduction_size(config: int | dict, processor_name: str, dataset_name: str) -> int:
    """Like _resolve_sized_config, but reduction_size has no "unset" value:
    it must resolve to an int >= 2."""
    resolved = _resolve_sized_config(config, processor_name, dataset_name)
    if resolved is None or resolved < 2:
        raise VineReduceError(
            f"reduction_size for processor {processor_name!r}, dataset {dataset_name!r} "
            f"resolved to {resolved!r}; it must be an int >= 2. Check the reduction_size "
            f'dict has a "default" entry or a matching "processors"/"datasets" override.'
        )
    return resolved


def _truncate_name(name: str, width: int = 20) -> str:
    """Keep only the last `width` characters of a long name (e.g. a file URL)
    for display."""
    return name if len(name) <= width else "..." + name[-width:]


def _status_of(outcome: Outcome) -> OutcomeStatus:
    if isinstance(outcome, Success):
        return OutcomeStatus.SUCCESS
    if isinstance(outcome, ResourceExhaustion):
        return OutcomeStatus.RESOURCE_EXHAUSTION
    return OutcomeStatus.FAILURE


def _update_resource_max(maxes: dict[str, float], resources: ResourceUsage) -> None:
    """Fold one Success's measured resources into a running per-resource peak
    (see SizeRecord). Failed tasks' numbers are not trusted as peaks."""
    for key, value in resources.to_rmsummary().items():
        maxes[key] = max(maxes.get(key, value), value)


class _ShrinkDecision(enum.Enum):
    REPOOL = "repool"
    SHRINK = "shrink"
    GIVE_UP = "give_up"


def _shrink_decision(unit_size: int, current_size: int, floor: int) -> _ShrinkDecision:
    """The exhaustion-retry rule shared by chunk and reduce tasks. unit_size
    is the size the task was tried at; current_size is the live
    chunksize/reduction_size.
      - REPOOL: unit_size > current_size - a sibling already shrank
        current_size while this task was in flight, so this is no evidence
        current_size is too big. Retry unshrunk; it is re-split on retry.
      - GIVE_UP: already at the floor.
      - SHRINK: halve current_size (clamped to the floor)."""
    if unit_size > current_size:
        return _ShrinkDecision.REPOOL
    if current_size <= floor:
        return _ShrinkDecision.GIVE_UP
    return _ShrinkDecision.SHRINK


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' if count != 1 else ''}"


@dataclass(frozen=True)
class CheckpointRef:
    """A durable item's checkpoint store row and on-disk path."""

    row_id: int
    path: str


@dataclass
class PoolItem:
    """A result this pipeline holds: a chunk's output, a reduction's output,
    a restart-seeded checkpoint, or a final result.

    handle: live distributor identity (file goes into submit() args,
        result_id into release_result/retrieve/checkpoint_path). None once
        the distributor's copy is never needed again: a final result copied
        into results_dir, a released item, or a restart-seeded final row.
    num_events / wall_time_s / memory_mb: totals folded into this item.
    covers: event ranges, per file URL, this item represents.
    since_checkpoint_time / since_checkpoint_distance: wall time and number
        of folds since this item's inputs were last checkpointed. Both reset
        to 0 by _checkpoint; they drive _checkpoint_due.
    checkpoint: set once the item is durable.
    lineage: every item folded into this one since its inputs were last
        checkpointed (flat; an uncheckpointed input contributes its own
        lineage too). Kept so TaskVine can recover a lost uncheckpointed
        result; released, and the checkpointed ones' rows superseded, once
        this item is itself checkpointed.
    attempts: failed reduction attempts in the current grouping. A
        RuntimeFailure raises it; a ResourceExhaustion resets it to 0, since a
        smaller reduction_size is a fresh start.
    """

    handle: ResultHandle | None
    num_events: int
    wall_time_s: float
    memory_mb: float
    covers: Coverage
    since_checkpoint_time: float
    since_checkpoint_distance: int
    checkpoint: CheckpointRef | None = None
    lineage: list[PoolItem] = field(default_factory=list)
    attempts: int = 0

    @property
    def is_checkpointed(self) -> bool:
        return self.checkpoint is not None


@dataclass(frozen=True)
class RestartPlan:
    """What plan_restart decided to do with the checkpoint rows on file."""

    finished: bool
    final_rows: list[CheckpointRecord]
    pool_rows: list[CheckpointRecord]  # empty when finished (partials are moot)
    covered: dict[str, Ranges]  # what the replayed rows cover, per file


def next_result_id() -> str:
    """A fresh result_id, unique for the lifetime of the distributor."""
    return uuid4().hex


def plan_restart(rows: list[CheckpointRecord], dataset_sizes: dict[str, int]) -> RestartPlan:
    """The restart rules:
    1. Every final row is replayed as a final result.
    2. The run is finished iff the final rows cover every event of every
       dataset file; partial rows are then moot and ignored.
    3. Otherwise every partial row is replayed as a pool item.
    4. Events the replayed rows cover are not processed again (see
       Pipeline._trim_covered)."""
    final_rows = [row for row in rows if row.is_final]
    partial_rows = [row for row in rows if not row.is_final]

    finalized = union_coverage(row.covers for row in final_rows)
    if covers_all(finalized, dataset_sizes):
        return RestartPlan(finished=True, final_rows=final_rows, pool_rows=[], covered=finalized)
    return RestartPlan(
        finished=False,
        final_rows=final_rows,
        pool_rows=partial_rows,
        covered=union_coverage(row.covers for row in rows),
    )


@dataclass
class _ChunkTask:
    chunk: Chunk
    attempts: int = 0


@dataclass
class _ReduceTask:
    group: list[PoolItem]
    is_final: bool
    is_checkpoint: bool
    force_final: bool = False

    @property
    def attempts(self) -> int:
        return max((item.attempts for item in self.group), default=0)


@dataclass
class ProgressCounters:
    """Progress-bar counters for one pipeline, read directly by progress.py.
    Cumulative over the whole run (never reset on retry)."""

    events_completed: int = 0
    events_failed: int = 0
    events_submitted: int = 0
    proc_tasks_completed: int = 0
    proc_tasks_failed: int = 0
    proc_tasks_submitted: int = 0
    reduce_tasks_completed: int = 0
    reduce_tasks_failed: int = 0
    reduce_tasks_submitted: int = 0

    @classmethod
    def summed(cls, pipelines: "list[Pipeline]") -> "ProgressCounters":
        return cls(
            **{f.name: sum(getattr(p.counters, f.name) for p in pipelines) for f in fields(cls)}
        )


class Pipeline:
    """Drives one (processor, dataset) pair from chunk generation through to
    its final result(s), including checkpointing and restart."""

    def __init__(
        self,
        config: VineReduce,
        *,
        processor_name: str,
        dataset_name: str,
        dataset: dict[str, Any],
        distributor: Distributor,
        db: CheckpointStore,
        process_priority: int,
        reduce_priority: int,
        failure_log: FailureLog | None = None,
        size_log: SizeLog | None = None,
        task_reporter: TaskReporter | None = None,
    ):
        """config supplies every setting shared across pipelines; per-pair
        sizes (chunksize, reduction_size and their floors) and is_result are
        resolved from it here."""
        self._config = config
        self.processor_name = processor_name
        self.dataset_name = dataset_name
        self._processor = config.processors[processor_name]
        self._dataset = dataset
        self._dataset_metadata = dataset.get("metadata", {})
        self._distributor = distributor
        self._db = db
        self.events_total = sum(dataset["files"].values())
        self._is_result = config.is_result or defaults.make_default_is_result(self.events_total)
        self.chunksize = _resolve_sized_config(config.chunksize, processor_name, dataset_name)
        self._minimum_chunksize = max(1, config.minimum_chunksize)
        self.reduction_size = _resolve_reduction_size(
            config.reduction_size, processor_name, dataset_name
        )
        self._minimum_reduction_size = min(
            max(2, config.minimum_reduction_size), self.reduction_size
        )
        self._results_dir = os.path.join(config.results_dir, dataset_name, processor_name)
        self._process_priority = process_priority
        self._reduce_priority = reduce_priority
        self._process_category = f"{processor_name}:{dataset_name}:process"
        self._reduce_category = f"{processor_name}:{dataset_name}:reduce"
        self._failure_log = failure_log
        self._size_log = size_log
        self._size_logged = False
        self._processing_max: dict[str, float] = {}
        self._reduction_max: dict[str, float] = {}
        self._task_reporter = task_reporter

        self.pool: list[PoolItem] = []
        self.final_results: list[PoolItem] = []
        self._retry_chunks: list[_ChunkTask] = []
        self._in_flight: dict[str, _ChunkTask | _ReduceTask] = {}
        self._generator_exhausted = False
        self._generator: Iterator[Chunk] | None = None
        self._covered: dict[str, Ranges] = {}
        self.finished = False
        # Events in checkpoint rows not yet superseded (see events_safe).
        self._events_safe = 0

        # Chunks given up on (see _give_up_on_chunk); their events count
        # against failure_proportion, whose denominator is events_total.
        self._failed_chunks: list[Chunk] = []
        self._failed_events = 0
        self.counters = ProgressCounters()

        self._seed_from_checkpoints()
        if self.finished:
            # The scheduling loop skips a pipeline that starts out finished,
            # so refresh_finished never gets to log its size row.
            self._log_size_once()
        else:
            os.makedirs(self._results_dir, exist_ok=True)

    # -- restart -----------------------------------------------------------

    @staticmethod
    def _seeded_item(row: CheckpointRecord, handle: ResultHandle | None) -> PoolItem:
        """A PoolItem replaying one checkpoint row from a previous run."""
        return PoolItem(
            handle=handle,
            num_events=row.num_events,
            wall_time_s=row.wall_time_s,
            memory_mb=row.memory_mb,
            covers=row.covers,
            since_checkpoint_time=0,
            since_checkpoint_distance=0,
            checkpoint=CheckpointRef(row.id, row.path),
        )

    def _seed_from_checkpoints(self) -> None:
        """Replay the checkpoint rows on file per plan_restart. Pool rows are
        adopted by the distributor, so they behave exactly like results of
        this run; final rows get no handle, as they are never resubmitted."""
        rows = self._db.checkpoints_for(self.processor_name, self.dataset_name)
        plan = plan_restart(rows, self._dataset["files"])
        self.final_results = [self._seeded_item(row, handle=None) for row in plan.final_rows]
        self._covered = plan.covered
        self.finished = plan.finished
        for row in plan.pool_rows:
            result_id = next_result_id()
            file = self._distributor.adopt_checkpoint(result_id, row.path)
            self.pool.append(self._seeded_item(row, handle=ResultHandle(result_id, file)))
        self._events_safe = sum(row.num_events for row in (*plan.final_rows, *plan.pool_rows))

    # -- state queries -------------------------------------------------------

    @property
    def events_safe(self) -> int:
        """Events whose result is durably checkpointed right now, i.e. would
        survive a restart without recomputation. Never decreases: a checkpoint
        only ever supersedes checkpoints whose events it covers."""
        return self._events_safe

    @property
    def failed_chunks(self) -> frozenset[Chunk]:
        """Chunks permanently given up on (see _give_up_on_chunk)."""
        return frozenset(self._failed_chunks)

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    def owns(self, result_id: str) -> bool:
        """Whether result_id is one of this pipeline's in-flight tasks."""
        return result_id in self._in_flight

    def refresh_finished(self) -> None:
        """Set `finished` if there is nothing left to do. Also called by the
        scheduling loop each cycle, to catch a pipeline that finishes without
        any outcome to react to (e.g. an empty dataset)."""
        if not self.finished:
            self.finished = self.chunks_all_done and not self.pool and self.in_flight_count == 0
            if self.finished:
                self._log_size_once()

    @property
    def chunks_all_done(self) -> bool:
        """Whether nothing more will ever be added to the pool from chunk
        processing: generation exhausted, no retry pending or in flight."""
        return (
            self._generator_exhausted
            and not self._retry_chunks
            and not any(isinstance(t, _ChunkTask) for t in self._in_flight.values())
        )

    # -- chunk processing ------------------------------------------------------

    def feed(self, budget: int) -> int:
        """Submit up to `budget` chunk-processing tasks; return how many were
        submitted."""
        if self.finished or budget <= 0:
            return 0
        if self._generator is None:
            self._generator = self._trim_covered(
                self._config.datasets_to_chunks(self._dataset, lambda: self.chunksize)
            )

        submitted = 0
        while submitted < budget:
            next_task = self._next_chunk()
            if next_task is None:
                break
            self._submit_chunk(next_task)
            submitted += 1
        return submitted

    def _trim_covered(self, chunks: Iterator[Chunk]) -> Iterator[Chunk]:
        """chunks cut down to the events no restart-seeded row covers yet,
        whatever chunk boundaries this run's datasets_to_chunks picks."""
        for chunk in chunks:
            yield from uncovered_chunks(chunk, self._covered.get(chunk.url, ()))

    def _next_chunk(self) -> _ChunkTask | None:
        """The next chunk to submit - retries first, then fresh ones - or None
        if there is nothing to submit right now."""
        if self._retry_chunks:
            task = self._retry_chunks.pop()
            chunk, attempts_used = task.chunk, task.attempts
            # A retry may predate the last chunksize halving: split it so it
            # is retried at the smaller size. Both pieces start fresh.
            if self.chunksize is not None and chunk.num_events > self.chunksize:
                split_point = chunk.start + self.chunksize
                self._retry_chunks.append(_ChunkTask(Chunk(chunk.url, split_point, chunk.stop), 0))
                chunk = Chunk(chunk.url, chunk.start, split_point)
                attempts_used = 0
            return _ChunkTask(chunk, attempts_used)

        if self._generator_exhausted:
            return None
        chunk = next(self._generator, None)
        if chunk is None:
            self._generator_exhausted = True
            return None
        return _ChunkTask(chunk, 0)

    def _submit_chunk(self, task: _ChunkTask) -> None:
        chunk = task.chunk
        result_id = next_result_id()
        self._distributor.submit(
            result_id,
            self._process_priority,
            self._process_category,
            TaskKind.PROCESSOR,
            defaults.executor_wrapper,
            self._processor,
            chunk,
            {
                "dataset": self._dataset_metadata,
                "distributor": self._distributor.resources(TaskKind.PROCESSOR),
                "executor": None,
            },
            self._config.chunk_to_args,
            self._config.executor,
        )
        self._in_flight[result_id] = task
        self.counters.proc_tasks_submitted += 1
        self.counters.events_submitted += chunk.num_events

    # -- reduction pool ------------------------------------------------------

    def submit_ready_reductions(self) -> None:
        """Submit every full-size group currently available in the pool."""
        while len(self.pool) >= self.reduction_size:
            group, self.pool = self.pool[: self.reduction_size], self.pool[self.reduction_size :]
            self._submit_reduction(group)

    def maybe_drain_final_group(self) -> None:
        """Once nothing more can ever arrive in the pool, reduce whatever is
        left as one last group, forced final. is_result is skipped: no other
        group will ever exist for it to judge, so it could only agree or
        deadlock resubmitting the same fold."""
        if (
            self.pool
            and self.chunks_all_done
            and self.in_flight_count == 0
            and len(self.pool) <= self.reduction_size
        ):
            group, self.pool = self.pool, []
            self._submit_reduction(group, force_final=True)

    def _submit_reduction(self, group: list[PoolItem], force_final: bool = False) -> None:
        """Submit a reduction of group, deciding up front whether it is final
        (force_final skips is_result) and whether to checkpoint it."""
        num_events = sum(item.num_events for item in group)
        total_time = sum(item.wall_time_s for item in group)
        total_memory = sum(item.memory_mb for item in group)
        is_final = force_final or self._is_result(num_events, total_time, total_memory)
        is_checkpoint = is_final or self._checkpoint_due(
            sum(item.since_checkpoint_time for item in group),
            max(item.since_checkpoint_distance for item in group),
        )

        result_id = next_result_id()
        self._distributor.submit(
            result_id,
            self._reduce_priority,
            self._reduce_category,
            TaskKind.REDUCER,
            defaults.reducer_wrapper,
            self._config.reducer,
            [item.handle.file for item in group],
            is_final,
            self._config.result_postprocess,
            is_checkpoint=is_checkpoint,
        )
        self._in_flight[result_id] = _ReduceTask(
            group=group,
            is_final=is_final,
            is_checkpoint=is_checkpoint,
            force_final=force_final,
        )
        self.counters.reduce_tasks_submitted += 1

    def _checkpoint_due(self, since_checkpoint_time: float, since_checkpoint_distance: int) -> bool:
        """Whether the reduction about to be submitted should be checkpointed,
        from its group's summed time and max distance since the last
        checkpoint (see PLAN.md's "When a checkpoint is taken")."""
        checkpoint_time = self._config.checkpoint_time
        checkpoint_distance = self._config.checkpoint_distance
        return (checkpoint_time is not None and since_checkpoint_time >= checkpoint_time) or (
            checkpoint_distance is not None and since_checkpoint_distance >= checkpoint_distance
        )

    # -- outcome handling ------------------------------------------------------

    def handle_outcome(self, outcome: Outcome) -> None:
        """React to the Outcome of one of this pipeline's tasks: pool a
        chunk's output, fold a reduction's output into the pool or
        final_results, or retry (halving chunksize/reduction_size on
        ResourceExhaustion). Raises VineReduceError once retries run out -
        for a chunk only if that also trips failure_proportion, for a
        reduction always."""
        task = self._in_flight.pop(outcome.result_id)
        if isinstance(task, _ChunkTask):
            self._handle_chunk_outcome(task, outcome)
        else:
            self._handle_reduce_outcome(task, outcome)
        self.refresh_finished()

    def _report_task(self, kind: ReportKind, description: str, outcome: Outcome) -> None:
        if self._task_reporter is None:
            return
        resources_allocated = outcome.resources_allocated
        if resources_allocated is None:
            # No per-call allocation reported; fall back to the static cap.
            task_kind = TaskKind.PROCESSOR if kind == ReportKind.PROCESSOR else TaskKind.REDUCER
            resources_allocated = self._distributor.resources(task_kind)
        self._task_reporter.report(
            TaskReport(
                processor_name=self.processor_name,
                dataset_name=self.dataset_name,
                kind=kind,
                result_id=outcome.result_id,
                description=description,
                status=_status_of(outcome),
                resources=outcome.resources,
                resources_allocated=resources_allocated,
                std_output=outcome.std_output,
            )
        )

    def _handle_chunk_outcome(self, task: _ChunkTask, outcome: Outcome) -> None:
        chunk = task.chunk
        self._report_task(
            ReportKind.PROCESSOR,
            f"{_truncate_name(chunk.url)}[{chunk.start}:{chunk.stop}]",
            outcome,
        )
        if isinstance(outcome, RuntimeFailure):
            self._handle_chunk_runtime_failure(task, outcome)
        elif isinstance(outcome, ResourceExhaustion):
            self._handle_chunk_resource_exhaustion(task, outcome)
        else:
            assert isinstance(outcome, Success)
            self._handle_chunk_success(task, outcome)

    def _handle_chunk_success(self, task: _ChunkTask, outcome: Success) -> None:
        chunk = task.chunk
        self.counters.proc_tasks_completed += 1
        self.counters.events_completed += chunk.num_events
        _update_resource_max(self._processing_max, outcome.resources)
        wall_time_s = outcome.resources.wall_time_s or 0.0
        self.pool.append(
            PoolItem(
                handle=ResultHandle(outcome.result_id, outcome.file),
                num_events=chunk.num_events,
                wall_time_s=wall_time_s,
                memory_mb=outcome.resources.memory_mb or 0.0,
                covers={chunk.url: ((chunk.start, chunk.stop),)},
                since_checkpoint_time=wall_time_s,
                since_checkpoint_distance=0,
            )
        )

    def _record_chunk_failure(self, chunk: Chunk) -> None:
        self.counters.proc_tasks_failed += 1
        self.counters.events_failed += chunk.num_events

    def _handle_chunk_runtime_failure(self, task: _ChunkTask, outcome: RuntimeFailure) -> None:
        chunk = task.chunk
        attempts_used = task.attempts + 1
        self._record_chunk_failure(chunk)
        if attempts_used < self._config.attempts:
            self._retry_chunks.append(_ChunkTask(chunk, attempts_used))
            return
        self._give_up_on_chunk(
            chunk,
            attempts=attempts_used,
            resources_measured=outcome.resources,
            traceback=outcome.traceback,
            message=(
                f"processor {self.processor_name!r} failed on "
                f"{chunk.url}[{chunk.start}:{chunk.stop}] after "
                f"{_plural(attempts_used, 'attempt')} (attempts={self._config.attempts}):\n"
                f"{outcome.traceback}"
            ),
        )

    def _handle_chunk_resource_exhaustion(
        self, task: _ChunkTask, outcome: ResourceExhaustion
    ) -> None:
        chunk = task.chunk
        self._record_chunk_failure(chunk)
        current_size = self.chunksize if self.chunksize is not None else chunk.num_events
        decision = _shrink_decision(chunk.num_events, current_size, self._minimum_chunksize)
        if decision is _ShrinkDecision.REPOOL:
            # Re-split down to the current chunksize when retried (_next_chunk).
            self._retry_chunks.append(_ChunkTask(chunk, 0))
        elif decision is _ShrinkDecision.GIVE_UP:
            self._give_up_on_chunk(
                chunk,
                attempts=task.attempts + 1,
                resources_measured=outcome.resources,
                traceback=None,
                message=(
                    f"processor {self.processor_name!r} exhausted resources on "
                    f"{chunk.url}[{chunk.start}:{chunk.stop}] at the minimum chunk "
                    f"size ({self._minimum_chunksize} events); cannot retry smaller."
                ),
            )
        else:
            # Clamp to the floor so e.g. 1500 with a floor of 1000 still tries
            # 1000 before giving up. attempts carries over; it resets once
            # _next_chunk actually splits this chunk at the new size.
            self.chunksize = max(current_size // 2, self._minimum_chunksize)
            self._retry_chunks.append(_ChunkTask(chunk, task.attempts))

    def _give_up_on_chunk(
        self,
        chunk: Chunk,
        *,
        attempts: int,
        resources_measured: ResourceUsage | None,
        traceback: str | None,
        message: str,
    ) -> None:
        """A processor permanent failure: log it and drop just this chunk's
        events (its file's other chunks carry on, leaving a hole). Raises
        VineReduceError(message) only if that trips failure_proportion."""
        self._failed_chunks.append(chunk)
        self._failed_events += chunk.num_events
        if self._failure_log is not None:
            self._failure_log.log(
                FailureRecord(
                    dataset_name=self.dataset_name,
                    filename=chunk.url,
                    ranges=((chunk.start, chunk.stop),),
                    kind=TaskKind.PROCESSOR,
                    attempts=attempts,
                    resources_allocated=self._distributor.resources(TaskKind.PROCESSOR),
                    resources_measured=resources_measured,
                    traceback=traceback,
                )
            )
        if self._failed_events > self._config.failure_proportion * self.events_total:
            raise VineReduceError(message)

    def _handle_reduce_outcome(self, task: _ReduceTask, outcome: Outcome) -> None:
        if task.force_final:
            kind = ReportKind.FINAL
        elif task.is_final:
            kind = ReportKind.RESULT
        elif task.is_checkpoint:
            kind = ReportKind.CHECKPOINT
        else:
            kind = ReportKind.REDUCER
        description = f"reduction of {_plural(len(task.group), 'partial')}"
        if task.is_final:
            description += " (final)"
        self._report_task(kind, description, outcome)

        if isinstance(outcome, RuntimeFailure):
            self._handle_reduce_runtime_failure(task, outcome)
        elif isinstance(outcome, ResourceExhaustion):
            self._handle_reduce_resource_exhaustion(task, outcome)
        else:
            assert isinstance(outcome, Success)
            self._handle_reduce_success(task, outcome)

    def _handle_reduce_success(self, task: _ReduceTask, outcome: Success) -> None:
        """Fold a completed reduction's group into a new PoolItem. The group
        (and its lineage) is carried as the new item's lineage, and released
        only once the new item (or a descendant) is checkpointed."""
        group = task.group
        lineage = [ancestor for item in group for ancestor in (item, *item.lineage)]
        self.counters.reduce_tasks_completed += 1
        _update_resource_max(self._reduction_max, outcome.resources)
        wall_time_s = outcome.resources.wall_time_s or 0.0
        new_item = PoolItem(
            handle=ResultHandle(outcome.result_id, outcome.file),
            num_events=sum(item.num_events for item in group),
            wall_time_s=sum(item.wall_time_s for item in group) + wall_time_s,
            memory_mb=sum(item.memory_mb for item in group) + (outcome.resources.memory_mb or 0.0),
            covers=union_coverage(item.covers for item in group),
            since_checkpoint_time=sum(item.since_checkpoint_time for item in group) + wall_time_s,
            since_checkpoint_distance=max(item.since_checkpoint_distance for item in group) + 1,
            lineage=lineage,
        )

        if task.is_checkpoint:
            self._checkpoint(new_item, task.is_final)

        if task.is_final:
            self.final_results.append(new_item)
        else:
            self.pool.append(new_item)

    def _handle_reduce_runtime_failure(self, task: _ReduceTask, outcome: RuntimeFailure) -> None:
        group = task.group
        attempts_used = task.attempts + 1
        self.counters.reduce_tasks_failed += 1
        if attempts_used >= self._config.attempts:
            self._give_up_on_reduction(
                group,
                attempts=attempts_used,
                resources_measured=outcome.resources,
                traceback=outcome.traceback,
                message=(
                    f"reducer for {self.processor_name!r}/{self.dataset_name!r} failed after "
                    f"{_plural(attempts_used, 'attempt')} "
                    f"(attempts={self._config.attempts}):\n{outcome.traceback}"
                ),
            )
        for item in group:
            item.attempts = attempts_used
        # force_final must carry over: if this is the drain group, letting
        # is_result recompute is_final could leave the dataset with no final.
        self._submit_reduction(group, force_final=task.force_final)

    def _handle_reduce_resource_exhaustion(
        self, task: _ReduceTask, outcome: ResourceExhaustion
    ) -> None:
        group = task.group
        self.counters.reduce_tasks_failed += 1
        decision = _shrink_decision(len(group), self.reduction_size, self._minimum_reduction_size)
        if decision is _ShrinkDecision.GIVE_UP:
            self._give_up_on_reduction(
                group,
                attempts=task.attempts + 1,
                resources_measured=outcome.resources,
                traceback=None,
                message=(
                    f"reducer for {self.processor_name!r}/{self.dataset_name!r} exhausted "
                    "resources at the minimum reduction_size "
                    f"({self._minimum_reduction_size}); cannot retry smaller."
                ),
            )
        if decision is _ShrinkDecision.SHRINK:
            # Clamp to the floor so e.g. 3 with a floor of 2 still tries 2.
            self.reduction_size = max(self.reduction_size // 2, self._minimum_reduction_size)
        # SHRINK or REPOOL: requeue at the front, re-split at the current
        # reduction_size next cycle. Exhaustion is a fresh start, not a strike.
        for item in group:
            item.attempts = 0
        self.pool[:0] = group

    def _give_up_on_reduction(
        self,
        group: list[PoolItem],
        *,
        attempts: int,
        resources_measured: ResourceUsage | None,
        traceback: str | None,
        message: str,
    ) -> NoReturn:
        """A reducer permanent failure: log every file range the group held,
        release the group's uncheckpointed items and lineage, and raise
        VineReduceError(message). Always aborts, regardless of
        failure_proportion: a partially folded result can't be trusted."""
        if self._failure_log is not None:
            allocated = self._distributor.resources(TaskKind.REDUCER)
            covers = union_coverage(item.covers for item in group)
            for url in sorted(covers):
                self._failure_log.log(
                    FailureRecord(
                        dataset_name=self.dataset_name,
                        filename=url,
                        ranges=covers[url],
                        kind=TaskKind.REDUCER,
                        attempts=attempts,
                        resources_allocated=allocated,
                        resources_measured=resources_measured,
                        traceback=traceback,
                    )
                )
        # Checkpoints keep their files: nothing supersedes their rows, which a
        # restart still needs to adopt.
        for item in group:
            for ancestor in (item, *item.lineage):
                if not ancestor.is_checkpointed:
                    self._release(ancestor)
        raise VineReduceError(message)

    # -- checkpoints -----------------------------------------------------------

    def _checkpoint(self, new_item: PoolItem, is_final: bool) -> None:
        """Make new_item durable, record it, and release its lineage,
        superseding the lineage's checkpoint rows. The new row and the
        deletion of the superseded rows are one transaction, so a crash never
        leaves both on file."""
        path, offloaded = self._durable_path(new_item, is_final)
        superseded = [item for item in new_item.lineage if item.is_checkpointed]
        row_id = self._db.record(
            processor=self.processor_name,
            dataset=self.dataset_name,
            covers=new_item.covers,
            num_events=new_item.num_events,
            wall_time_s=new_item.wall_time_s,
            memory_mb=new_item.memory_mb,
            is_final=is_final,
            path=path,
            supersedes=[item.checkpoint.row_id for item in superseded],
            offloaded=offloaded,
        )
        self._events_safe += new_item.num_events - sum(item.num_events for item in superseded)
        for item in new_item.lineage:
            self._release(item)
        new_item.lineage = []
        new_item.checkpoint = CheckpointRef(row_id, path)
        new_item.since_checkpoint_time = 0
        new_item.since_checkpoint_distance = 0

    def _durable_path(self, new_item: PoolItem, is_final: bool) -> tuple[str, bool]:
        """new_item's durable path, and whether it is an offload pointer file.

        A non-final checkpoint is already durable (submitted with
        is_checkpoint=True), so its path is just looked up and its handle
        stays live for later reductions. A final result is copied into
        results_dir (or replaced by a pointer file, under
        result_postprocess_offload) and its distributor copy released."""
        if not is_final:
            return self._distributor.checkpoint_path(new_item.handle.result_id), False
        if self._config.result_postprocess_offload:
            path = self._write_offload_pointer(new_item)
        else:
            path = self._new_result_path(".pkl.zst")
            self._distributor.retrieve(new_item.handle.result_id, path)
        self._distributor.release_result(new_item.handle.result_id)
        new_item.handle = None
        return path, self._config.result_postprocess_offload

    def _new_result_path(self, suffix: str, prefix: str = "") -> str:
        return os.path.join(
            self._results_dir, f"{prefix}{self.processor_name}__{uuid4().hex}{suffix}"
        )

    def _write_offload_pointer(self, new_item: PoolItem) -> str:
        """Write the pointer file for an offloaded final result. The locator
        result_postprocess returned is small by contract, and is retrieved to
        a hidden temp file so a failed validation leaves nothing behind for a
        restart to mistake for a result."""
        tmp_path = self._new_result_path(".offload.tmp", prefix=".")
        self._distributor.retrieve(new_item.handle.result_id, tmp_path)
        try:
            locator = serialization.load(tmp_path)
        finally:
            os.remove(tmp_path)

        try:
            json.dumps(locator)
        except TypeError as e:
            raise VineReduceError(
                f"result_postprocess for processor {self.processor_name!r}, dataset "
                f"{self.dataset_name!r} must return a JSON-serializable locator when "
                f"result_postprocess_offload=True; got {locator!r} ({type(locator).__name__}): {e}"
            ) from e

        path = self._new_result_path(".pointer.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "offloaded": True,
                    "processor_name": self.processor_name,
                    "dataset_name": self.dataset_name,
                    "locator": locator,
                },
                f,
                indent=2,
            )
        return path

    def _release(self, item: PoolItem) -> None:
        """Release item's distributor copy (for a checkpoint, its durable file
        too). Clearing the handle makes a double release impossible."""
        if item.handle is not None:
            self._distributor.release_result(item.handle.result_id)
            item.handle = None

    # -- size.jsonl ------------------------------------------------------------

    def _log_size_once(self) -> None:
        """Write this pipeline's size.jsonl row (see size_log.py), exactly once."""
        if self._size_logged or self._size_log is None:
            return
        self._size_logged = True
        self._size_log.log(
            SizeRecord(
                dataset_name=self.dataset_name,
                processor_name=self.processor_name,
                chunk_size=self.chunksize,
                reduction_size=self.reduction_size,
                processing=self._rounded_max(self._processing_max),
                reduction=self._rounded_max(self._reduction_max),
            )
        )

    @staticmethod
    def _rounded_max(maxes: dict[str, float]) -> dict[str, float] | None:
        """None (not {}) when nothing was measured this run, so size.jsonl can
        tell "not measured" from "measured, all zero"."""
        if not maxes:
            return None
        return {key: round(value, 2) for key, value in maxes.items()}
