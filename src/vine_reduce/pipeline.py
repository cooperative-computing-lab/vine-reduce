"""Per-(processor, dataset) state: chunk generation, the reduction pool, and
checkpoint bookkeeping. See "Temporary Results, Checkpoints, and Restart" in
PLAN.md for the pooling and checkpointing rules this implements.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol
from uuid import uuid4

from .checkpoint_store import CheckpointRecord, CheckpointStore
from .distributor import Distributor
from .executor import Executor
from .failure_log import FailureLog, FailureRecord
from .types import Chunk, Outcome, ResourceExhaustion, ResultHandle, RuntimeFailure, Success


class VineReduceError(RuntimeError):
    """Raised when a processing or reduction function fails remotely. Carries
    the remote traceback so the failure can be debugged from the local side."""


@dataclass(frozen=True)
class TaskReport:
    """One finished processor/reducer task, handed to a TaskReporter right
    when its Outcome is examined - before Pipeline acts on it (pooling a
    chunk's output, folding a reduction, retrying, or raising on failure).
    See progress.py's ProgressReporter for the status display this feeds."""

    processor_name: str
    dataset_name: str
    kind: str  # "processor" | "reducer"
    result_id: str
    description: str
    status: str  # "success" | "resource_exhaustion" | "failure"
    resources: dict[str, Any]
    std_output: str | None


class TaskReporter(Protocol):
    """What a Pipeline needs to surface its finished tasks - see
    progress.py's ProgressReporter (the rich-based implementation) and
    NullProgressReporter (used when VineReduce(progress=False))."""

    def report(self, task: TaskReport) -> None:
        """Called once per finished processor/reducer task, regardless of
        outcome (success, resource exhaustion, or failure)."""
        ...


def _status_of(outcome: Outcome) -> str:
    if isinstance(outcome, Success):
        return "success"
    if isinstance(outcome, ResourceExhaustion):
        return "resource_exhaustion"
    return "failure"


@dataclass(frozen=True)
class CheckpointRef:
    """The durable identity of a checkpointed item: its store row and the
    on-disk path recorded there. Present iff the item is durable."""

    row_id: int
    path: str


@dataclass
class PoolItem:
    """A result this pipeline is holding: a chunk's output, a reduction's
    output, a restart-seeded checkpoint, or a final result.

    handle: the item's live distributor identity - handle.file is what goes
        into a later submit()'s args, handle.result_id is what
        release_result/retrieve/checkpoint_path take. None only once nothing
        will ever need the distributor's copy again: a final result safely
        copied into results_dir, a released item, or a restart-seeded final
        row (a final result is never resubmitted, so it never gets a handle).
        A non-final checkpoint keeps its handle live so it can still be
        folded into a later reduction without the manager re-sending it.
    num_events / wall_time_s / memory_mb: totals accumulated into this item.
    files: dataset file URLs whose data this item represents.
    since_checkpoint_time: wall time accumulated since this item (or one of
        its inputs) was last checkpointed - reset to 0 whenever a checkpoint
        is written (see Pipeline._checkpoint).
    since_checkpoint_distance: number of reductions folded into this item's
        lineage since it (or one of its inputs) was last checkpointed - 0 for
        a fresh chunk result, max(group)+1 whenever a group is folded, reset
        to 0 whenever a checkpoint is written (see Pipeline._checkpoint).
    checkpoint: the item's durable identity (store row id + on-disk path),
        once checkpointed. None while the item is not durable.
    inputs: the items folded together to produce this one (empty for a raw
        chunk's output). Kept around so that, if this item is lost before
        anything checkpoints it, it can be recovered by re-folding its
        inputs instead of recomputing everything from scratch - see
        Pipeline._release_covered, which frees this whole lineage once a
        durable checkpoint finally covers it, clearing `inputs` as it goes
        (a checkpointed item's own durable copy is backup enough, so there
        is nothing left in its lineage still worth holding onto).
    attempts: how many failed reduction attempts this item has survived
        while in its current grouping, for the `attempts` retry budget (see
        Pipeline._handle_reduce_outcome). 0 for a freshly produced item (a
        raw chunk's output, or a fresh fold result) - a RuntimeFailure on a
        reduction this item is part of raises it towards `attempts`, while a
        ResourceExhaustion resets it back to 0 (a smaller reduction_size is
        a fresh start, not a strike against the budget).
    """

    handle: ResultHandle | None
    num_events: int
    wall_time_s: float
    memory_mb: float
    files: frozenset[str]
    since_checkpoint_time: float
    since_checkpoint_distance: int
    checkpoint: CheckpointRef | None = None
    inputs: list[PoolItem] = field(default_factory=list)
    attempts: int = 0

    @property
    def is_checkpointed(self) -> bool:
        return self.checkpoint is not None


@dataclass(frozen=True)
class RestartPlan:
    """What plan_restart decided a restart should do with the checkpoint
    rows on file - see plan_restart for the rules."""

    finished: bool
    final_rows: list[CheckpointRecord]
    pool_rows: list[CheckpointRecord]  # empty when finished (partials are moot)
    skip_files: set[str]


def next_result_id() -> str:
    """A fresh result_id for submit()/adopt_checkpoint() - unique for the
    lifetime of the distributor (see Distributor.submit)."""
    return uuid4().hex


def plan_restart(rows: list[CheckpointRecord], dataset_files: set[str]) -> RestartPlan:
    """The restart rules, stated once:
    1. Every final row is replayed as a final result.
    2. The run is finished iff the final rows cover every dataset file;
       partial rows are then moot and ignored.
    3. Otherwise every partial row is replayed as a pool item.
    4. A file is skipped iff a replayed row covers it."""
    final_rows = [row for row in rows if row.is_final]
    partial_rows = [row for row in rows if not row.is_final]

    finalized_files = {url for row in final_rows for url in row.covers_files}
    if dataset_files <= finalized_files:
        return RestartPlan(
            finished=True,
            final_rows=final_rows,
            pool_rows=[],
            skip_files=set(dataset_files),
        )

    covered_by_partial = {url for row in partial_rows for url in row.covers_files}
    return RestartPlan(
        finished=False,
        final_rows=final_rows,
        pool_rows=partial_rows,
        skip_files=finalized_files | covered_by_partial,
    )


@dataclass
class _FileProgress:
    num_entries: int
    covered_events: int = 0
    staged_items: list[PoolItem] = field(default_factory=list)


@dataclass
class _ChunkTask:
    chunk: Chunk
    attempts_used: int = 0


@dataclass
class _ReduceTask:
    group: list[PoolItem]
    is_final: bool
    is_checkpoint: bool
    num_events: int
    total_time: float
    total_memory: float


class Pipeline:
    """Drives one (processor, dataset) pair from chunk generation through to
    its final result(s), including checkpointing and restart."""

    def __init__(
        self,
        *,
        processor_name: str,
        processor: Callable[[Any], Any],
        dataset_name: str,
        dataset: dict[str, Any],
        distributor: Distributor,
        db: CheckpointStore,
        datasets_to_chunks: Callable[[dict, Callable[[], int | None], set[str]], Iterator[Chunk]],
        chunk_to_args: Callable,
        executor: Executor,
        executor_wrapper: Callable,
        reducer: Callable,
        reducer_wrapper: Callable,
        is_result: Callable[[int, float, float], bool],
        result_postprocess: Callable | None,
        chunksize: int | None,
        reduction_size: int,
        checkpoint_time: float | None,
        checkpoint_distance: int | None,
        checkpoint_accumulations: bool,
        results_dir: str,
        process_priority: int,
        reduce_priority: int,
        attempts: int = 3,
        failure_proportion: float = 0.0,
        failure_log: FailureLog | None = None,
        task_reporter: TaskReporter | None = None,
    ):
        self.processor_name = processor_name
        self.dataset_name = dataset_name
        self._processor = processor
        self._dataset = dataset
        self._dataset_metadata = dataset.get("metadata", {})
        self._distributor = distributor
        self._db = db
        self._datasets_to_chunks = datasets_to_chunks
        self._chunk_to_args = chunk_to_args
        self._executor = executor
        self._executor_wrapper = executor_wrapper
        self._reducer = reducer
        self._reducer_wrapper = reducer_wrapper
        self._is_result = is_result
        self._result_postprocess = result_postprocess
        self.chunksize = chunksize
        self.reduction_size = reduction_size
        self._checkpoint_time = checkpoint_time
        self._checkpoint_distance = checkpoint_distance
        self._checkpoint_accumulations = checkpoint_accumulations
        self._results_dir = os.path.join(results_dir, dataset_name, processor_name)
        self._process_priority = process_priority
        self._reduce_priority = reduce_priority
        self._attempts = attempts
        self._failure_proportion = failure_proportion
        self._failure_log = failure_log
        self._process_category = f"{processor_name}:{dataset_name}:process"
        self._reduce_category = f"{processor_name}:{dataset_name}:reduce"
        self._task_reporter = task_reporter

        self.pool: list[PoolItem] = []
        self.final_results: list[PoolItem] = []
        self._files_in_progress: dict[str, _FileProgress] = {}
        self._retry_chunks: list[tuple[Chunk, int]] = []  # (chunk, attempts_used)
        self._in_flight: dict[str, _ChunkTask | _ReduceTask] = {}
        self._generator_exhausted = False
        self._generator: Iterator[Chunk] | None = None
        self._skip_files: set[str] = set()
        self.finished = False

        # Failure tolerance: files given up on during preprocessing (see
        # _give_up_on_file) are removed from the pool going forward, never
        # retried, and counted against failure_proportion -
        # _files_concluded is the proportion's denominator (files that
        # reached a terminal outcome, successful or not; see
        # PLAN.md's "Attempts and Retries").
        self._failed_files: set[str] = set()
        self._files_concluded = 0

        # Progress-bar counters (see events_*/proc_tasks_*/reduce_tasks_*
        # properties below) - cumulative across the whole run, not reset on
        # retry, so progress.py's totals estimate has a stable ratio to
        # extrapolate from. events_total is fixed at construction; the dataset
        # dict is never mutated after this.
        self.events_total = sum(dataset["files"].values())
        self._events_completed = 0
        self._events_failed = 0
        self._events_submitted = 0
        self._proc_tasks_completed = 0
        self._proc_tasks_failed = 0
        self._proc_tasks_submitted = 0
        self._reduce_tasks_completed = 0
        self._reduce_tasks_failed = 0
        self._reduce_tasks_submitted = 0

        self._seed_from_checkpoints()
        if not self.finished:
            os.makedirs(self._results_dir, exist_ok=True)

    # -- restart -----------------------------------------------------------

    @staticmethod
    def _seeded_item(row: CheckpointRecord, handle: ResultHandle | None) -> PoolItem:
        """A PoolItem replaying one checkpoint row from a previous run. The
        since-checkpoint counters start at zero: the row IS a checkpoint, so
        nothing has accumulated on top of it yet."""
        return PoolItem(
            handle=handle,
            num_events=row.num_events,
            wall_time_s=row.wall_time_s,
            memory_mb=row.memory_mb,
            files=row.covers_files,
            since_checkpoint_time=0,
            since_checkpoint_distance=0,
            checkpoint=CheckpointRef(row.id, row.path),
        )

    def _seed_from_checkpoints(self) -> None:
        """Restart support: replay the checkpoint rows on file (per
        plan_restart's rules) as final results and pool items, and record
        which dataset files they already cover so chunk generation can skip
        them. A replayed pool item is adopted by the distributor
        (adopt_checkpoint), so from here on it is indistinguishable from a
        result this run produced itself: same handle for resubmission, same
        release_result for cleanup. A replayed final row gets no handle -
        a final result is never resubmitted."""
        rows = self._db.checkpoints_for(self.processor_name, self.dataset_name)
        plan = plan_restart(rows, set(self._dataset["files"]))
        self.final_results = [self._seeded_item(row, handle=None) for row in plan.final_rows]
        self._skip_files = plan.skip_files
        self.finished = plan.finished
        for row in plan.pool_rows:
            result_id = next_result_id()
            file = self._distributor.adopt_checkpoint(result_id, row.path)
            handle = ResultHandle(result_id, file)
            self.pool.append(self._seeded_item(row, handle=handle))

    # -- chunk generation ----------------------------------------------------

    def in_flight_count(self) -> int:
        """How many chunk/reduce tasks this pipeline currently has submitted
        and not yet resolved."""
        return len(self._in_flight)

    # -- progress-bar counters -----------------------------------------------
    # Raw facts only - progress.py owns all bar-rendering and totals-estimate
    # math, so a pipeline's contribution to a processor's four aggregate bars
    # (see ProgressReporter) is just these properties summed across every
    # pipeline sharing that processor_name.

    @property
    def events_completed(self) -> int:
        return self._events_completed

    @property
    def events_failed(self) -> int:
        return self._events_failed

    @property
    def events_submitted(self) -> int:
        return self._events_submitted

    @property
    def events_safe(self) -> int:
        """Events whose result is durably checkpointed right now - i.e.
        covered by a PoolItem or final result with is_checkpointed True, and
        so would survive a crash/restart without recomputation. See
        PoolItem.is_checkpointed and Pipeline._checkpoint."""
        return sum(item.num_events for item in self.pool if item.is_checkpointed) + sum(
            item.num_events for item in self.final_results
        )

    @property
    def proc_tasks_completed(self) -> int:
        return self._proc_tasks_completed

    @property
    def proc_tasks_failed(self) -> int:
        return self._proc_tasks_failed

    @property
    def proc_tasks_submitted(self) -> int:
        return self._proc_tasks_submitted

    @property
    def proc_tasks_in_flight(self) -> int:
        return sum(1 for task in self._in_flight.values() if isinstance(task, _ChunkTask))

    @property
    def reduce_tasks_completed(self) -> int:
        return self._reduce_tasks_completed

    @property
    def reduce_tasks_failed(self) -> int:
        return self._reduce_tasks_failed

    @property
    def reduce_tasks_submitted(self) -> int:
        return self._reduce_tasks_submitted

    @property
    def reduce_tasks_in_flight(self) -> int:
        return self.in_flight_count() - self.proc_tasks_in_flight

    @property
    def failed_files(self) -> frozenset[str]:
        """Dataset file URLs permanently given up on during preprocessing -
        see _give_up_on_file. Does not include files caught up in a reducer
        permanent failure, since that always aborts the whole run instead
        (see _give_up_on_reduction)."""
        return frozenset(self._failed_files)

    def owns(self, result_id: str) -> bool:
        """Whether result_id was submitted by this pipeline (as opposed to
        another pipeline sharing the same distributor)."""
        return result_id in self._in_flight

    def refresh_finished(self) -> None:
        """Set `finished` if there is nothing left to do. Called after every
        outcome, and by the scheduling loop each cycle - the latter catches a
        pipeline that is done without ever producing an outcome to react to,
        e.g. an empty dataset, or one covered by seeded checkpoints for every
        file but not yet marked finished at construction."""
        if not self.finished:
            self.finished = self.chunks_all_done and not self.pool and self.in_flight_count() == 0

    @property
    def chunks_all_done(self) -> bool:
        """Whether chunk generation is exhausted and no chunk task (fresh or
        retry) is pending or in flight - i.e. nothing more will ever be
        added to the pool from chunk processing."""
        return (
            self._generator_exhausted
            and not self._retry_chunks
            and not self._files_in_progress
            and not any(isinstance(t, _ChunkTask) for t in self._in_flight.values())
        )

    def feed(self, budget: int) -> int:
        """Submit up to `budget` new chunk-processing tasks. Returns how many
        were actually submitted."""
        if self.finished or budget <= 0:
            return 0
        if self._generator is None:
            self._generator = self._datasets_to_chunks(
                self._dataset, lambda: self.chunksize, self._skip_files
            )

        submitted = 0
        while submitted < budget:
            next_chunk = self._next_chunk()
            if next_chunk is None:
                break
            chunk, attempts_used = next_chunk
            self._submit_chunk(chunk, attempts_used)
            submitted += 1
        return submitted

    def _next_chunk(self) -> tuple[Chunk, int] | None:
        """The next chunk to submit (with the attempts already used against
        it) - retries first, then freshly generated ones - or None when
        there is nothing left to submit right now."""
        if self._retry_chunks:
            chunk, attempts_used = self._retry_chunks.pop()
            # A retry chunk may predate the last chunksize halving; re-split it
            # so we actually retry at the smaller size, not the size that just
            # failed. A split is a fresh start for both pieces - see
            # PoolItem.attempts's docstring for the equivalent reduction rule.
            if self.chunksize is not None and chunk.num_events > self.chunksize:
                split_point = chunk.start + self.chunksize
                self._retry_chunks.append((Chunk(chunk.url, split_point, chunk.stop), 0))
                chunk = Chunk(chunk.url, chunk.start, split_point)
                attempts_used = 0
            return chunk, attempts_used

        if self._generator_exhausted:
            return None
        while True:
            chunk = next(self._generator, None)
            if chunk is None:
                self._generator_exhausted = True
                return None
            if chunk.url not in self._failed_files:
                return chunk, 0
            # A file already given up on (see _give_up_on_file) may still
            # have chunks left to yield from the generator's current
            # position - skip them without submitting.

    def _submit_chunk(self, chunk: Chunk, attempts_used: int = 0) -> None:
        self._files_in_progress.setdefault(
            chunk.url, _FileProgress(num_entries=self._dataset["files"][chunk.url])
        )
        result_id = next_result_id()
        self._distributor.submit(
            result_id,
            self._process_priority,
            self._process_category,
            "processor",
            self._executor_wrapper,
            self._processor,
            chunk,
            self._dataset_metadata,
            self._distributor.resources("processor"),
            None,
            self._chunk_to_args,
            self._executor,
        )
        self._in_flight[result_id] = _ChunkTask(chunk=chunk, attempts_used=attempts_used)
        self._proc_tasks_submitted += 1
        self._events_submitted += chunk.num_events

    # -- reduction pool ------------------------------------------------------

    def submit_ready_reductions(self) -> None:
        """Submit every full-size group currently available in the pool."""
        while len(self.pool) >= self.reduction_size:
            group, self.pool = self.pool[: self.reduction_size], self.pool[self.reduction_size :]
            self._submit_reduction(group)

    def maybe_drain_final_group(self) -> None:
        """If nothing more can ever arrive in the pool, reduce whatever's left
        as one last group, however small."""
        if self.pool and self.chunks_all_done and self.in_flight_count() == 0:
            group, self.pool = self.pool, []
            self._submit_reduction(group)

    def _submit_reduction(self, group: list[PoolItem]) -> None:
        num_events = sum(item.num_events for item in group)
        total_time = sum(item.wall_time_s for item in group)
        total_memory = sum(item.memory_mb for item in group)
        is_final = self._is_result(num_events, total_time, total_memory)
        is_checkpoint = is_final or self._checkpoint_due(
            sum(item.since_checkpoint_time for item in group),
            max(item.since_checkpoint_distance for item in group),
        )

        result_id = next_result_id()
        self._distributor.submit(
            result_id,
            self._reduce_priority,
            self._reduce_category,
            "reducer",
            self._reducer_wrapper,
            self._reducer,
            [item.handle.file for item in group],
            is_final,
            self._result_postprocess,
            is_checkpoint=is_checkpoint,
        )
        self._in_flight[result_id] = _ReduceTask(
            group=group,
            is_final=is_final,
            is_checkpoint=is_checkpoint,
            num_events=num_events,
            total_time=total_time,
            total_memory=total_memory,
        )
        self._reduce_tasks_submitted += 1

    # -- outcome handling ------------------------------------------------------

    def _report_task(self, kind: str, description: str, outcome: Outcome) -> None:
        if self._task_reporter is None:
            return
        self._task_reporter.report(
            TaskReport(
                processor_name=self.processor_name,
                dataset_name=self.dataset_name,
                kind=kind,
                result_id=outcome.result_id,
                description=description,
                status=_status_of(outcome),
                resources=outcome.resources,
                std_output=outcome.std_output,
            )
        )

    def handle_outcome(self, outcome: Outcome) -> None:
        """React to the Outcome of one of this pipeline's own chunk/reduce
        tasks: pool a chunk's output, fold a reduction's output into
        final_results or back into the pool, retry on ResourceExhaustion (
        halving chunksize/reduction_size), or raise VineReduceError on
        RuntimeFailure. Updates `finished` once nothing is left to do."""
        task = self._in_flight.pop(outcome.result_id)
        if isinstance(task, _ChunkTask):
            self._handle_chunk_outcome(task, outcome)
        else:
            self._handle_reduce_outcome(task, outcome)
        self.refresh_finished()

    def _handle_chunk_outcome(self, task: _ChunkTask, outcome: Outcome) -> None:
        chunk = task.chunk
        self._report_task("processor", f"{chunk.url}[{chunk.start}:{chunk.stop}]", outcome)

        if chunk.url in self._failed_files:
            # A sibling chunk of this (now abandoned) file already exhausted
            # its attempts and gave up on the whole file - see
            # _give_up_on_file. Discard whatever this one produced instead
            # of staging or retrying it.
            if isinstance(outcome, Success):
                self._distributor.release_result(outcome.result_id)
            return

        if isinstance(outcome, RuntimeFailure):
            attempts_used = task.attempts_used + 1
            self._proc_tasks_failed += 1
            self._events_failed += chunk.num_events
            if attempts_used >= self._attempts:
                self._give_up_on_file(
                    chunk,
                    kind="processor",
                    attempts=attempts_used,
                    resources_measured=outcome.resources,
                    traceback=outcome.traceback,
                    abort_message=(
                        f"processor {self.processor_name!r} failed on "
                        f"{chunk.url}[{chunk.start}:{chunk.stop}] after {attempts_used} "
                        f"attempt{'s' if attempts_used != 1 else ''} (attempts={self._attempts}):\n"
                        f"{outcome.traceback}"
                    ),
                )
                return
            self._retry_chunks.append((chunk, attempts_used))
            return
        if isinstance(outcome, ResourceExhaustion):
            current_size = self.chunksize if self.chunksize is not None else chunk.num_events
            self._proc_tasks_failed += 1
            self._events_failed += chunk.num_events
            if current_size <= 1:
                self._give_up_on_file(
                    chunk,
                    kind="processor",
                    attempts=task.attempts_used + 1,
                    resources_measured=outcome.resources,
                    traceback=None,
                    abort_message=(
                        f"processor {self.processor_name!r} exhausted resources on "
                        f"{chunk.url}[{chunk.start}:{chunk.stop}] at the minimum chunk size "
                        "(1 event); cannot retry smaller."
                    ),
                )
                return
            self.chunksize = max(1, current_size // 2)
            # attempts_used carries over unchanged here - it is reset to 0 in
            # _next_chunk once this chunk is actually split at the new,
            # smaller chunksize (a halving is a fresh start, not a strike
            # against the budget - see PoolItem.attempts's docstring for the
            # equivalent reduction rule).
            self._retry_chunks.append((chunk, task.attempts_used))
            return

        assert isinstance(outcome, Success)
        self._proc_tasks_completed += 1
        self._events_completed += chunk.num_events
        wall_time_s = outcome.resources.get("wall_time_s", 0.0)
        memory_mb = outcome.resources.get("memory_mb", 0.0)

        progress = self._files_in_progress[chunk.url]
        progress.covered_events += chunk.num_events
        progress.staged_items.append(
            PoolItem(
                handle=ResultHandle(outcome.result_id, outcome.file),
                num_events=chunk.num_events,
                wall_time_s=wall_time_s,
                memory_mb=memory_mb,
                files=frozenset({chunk.url}),
                since_checkpoint_time=wall_time_s,
                since_checkpoint_distance=0,
            )
        )
        if progress.covered_events >= progress.num_entries:
            self.pool.extend(progress.staged_items)
            del self._files_in_progress[chunk.url]
            self._files_concluded += 1

    def _give_up_on_file(
        self,
        chunk: Chunk,
        *,
        kind: str,
        attempts: int,
        resources_measured: dict[str, Any] | None,
        traceback: str | None,
        abort_message: str,
    ) -> None:
        """A processor permanent failure: log it, drop the file from the
        pool for good (it is never retried or staged), and either continue
        (most of the dataset's other files are unaffected) or abort the
        whole run, per failure_proportion - see PLAN.md's "Attempts and
        Retries". Unlike a reducer permanent failure (_give_up_on_reduction),
        this does NOT unconditionally abort."""
        url = chunk.url
        self._failed_files.add(url)
        self._files_concluded += 1

        if self._failure_log is not None:
            self._failure_log.log(
                FailureRecord(
                    dataset_name=self.dataset_name,
                    filename=url,
                    kind=kind,
                    attempts=attempts,
                    resources_allocated=self._distributor.resources(kind),
                    resources_measured=resources_measured,
                    traceback=traceback,
                )
            )

        # This file will never be pooled - release any sibling chunks of it
        # already staged (partial progress, now moot) instead of leaking
        # them at the distributor.
        progress = self._files_in_progress.pop(url, None)
        if progress is not None:
            for item in progress.staged_items:
                assert item.handle is not None
                self._distributor.release_result(item.handle.result_id)
        self._retry_chunks = [(c, a) for c, a in self._retry_chunks if c.url != url]

        ratio = len(self._failed_files) / max(self._files_concluded, 100)
        if ratio > self._failure_proportion:
            raise VineReduceError(abort_message)

    def _handle_reduce_outcome(self, task: _ReduceTask, outcome: Outcome) -> None:
        group = task.group
        description = f"fold of {len(group)} item{'s' if len(group) != 1 else ''}"
        if task.is_final:
            description += " (final)"
        self._report_task("reducer", description, outcome)

        if isinstance(outcome, RuntimeFailure):
            attempts_used = 1 + max((item.attempts for item in group), default=0)
            self._reduce_tasks_failed += 1
            if attempts_used >= self._attempts:
                self._give_up_on_reduction(
                    group,
                    attempts=attempts_used,
                    resources_measured=outcome.resources,
                    traceback=outcome.traceback,
                )
                raise VineReduceError(
                    f"reducer for {self.processor_name!r}/{self.dataset_name!r} failed after "
                    f"{attempts_used} attempt{'s' if attempts_used != 1 else ''} "
                    f"(attempts={self._attempts}):\n{outcome.traceback}"
                )
            for item in group:
                item.attempts = attempts_used
            self._submit_reduction(group)  # retry the exact same group unchanged
            return
        if isinstance(outcome, ResourceExhaustion):
            self._reduce_tasks_failed += 1
            if self.reduction_size <= 2:
                self._give_up_on_reduction(
                    group,
                    attempts=1 + max((item.attempts for item in group), default=0),
                    resources_measured=outcome.resources,
                    traceback=None,
                )
                raise VineReduceError(
                    f"reducer for {self.processor_name!r}/{self.dataset_name!r} exhausted "
                    "resources at the minimum reduction_size (2); cannot retry smaller."
                )
            self.reduction_size = max(2, self.reduction_size // 2)
            # A halved reduction_size is a fresh start for these items, not a
            # strike against the budget - see PoolItem.attempts's docstring.
            for item in group:
                item.attempts = 0
            self.pool[:0] = group  # retry with a (now smaller) reduction_size next cycle
            return

        assert isinstance(outcome, Success)
        self._reduce_tasks_completed += 1
        # group's own handles are deliberately not released here: new_item
        # isn't durable yet, so if it's lost before something checkpoints
        # it, group (kept below as new_item.inputs) is the only way to
        # recover it without recomputing everything from scratch. See
        # _release_covered, invoked from _checkpoint once new_item
        # eventually does become durable.
        wall_time_s = outcome.resources.get("wall_time_s", 0.0)
        memory_mb = outcome.resources.get("memory_mb", 0.0)
        new_item = PoolItem(
            handle=ResultHandle(outcome.result_id, outcome.file),
            num_events=task.num_events,
            wall_time_s=task.total_time + wall_time_s,
            memory_mb=task.total_memory + memory_mb,
            files=frozenset().union(*(item.files for item in group)),
            since_checkpoint_time=sum(item.since_checkpoint_time for item in group) + wall_time_s,
            since_checkpoint_distance=max(item.since_checkpoint_distance for item in group) + 1,
            inputs=group,
        )

        if task.is_checkpoint:
            self._checkpoint(new_item, group, task.is_final)

        if task.is_final:
            self.final_results.append(new_item)
        elif len(group) == 1:
            # Only maybe_drain_final_group ever reduces a group of size 1 -
            # folding a single item with itself changes nothing, so if
            # is_result still rejects the result, nothing else ever will
            # either: chunk generation is exhausted and nothing else is in
            # flight, so there is no future group is_result could see instead.
            # Looping this back into the pool would just resubmit the same
            # no-op reduction forever.
            raise VineReduceError(
                f"is_result never accepted a final result for "
                f"{self.processor_name!r}/{self.dataset_name!r}: "
                f"{new_item.num_events} events reduced, no chunks or reductions "
                "left to add more. is_result is unsatisfiable for this run."
            )
        else:
            self.pool.append(new_item)

    def _give_up_on_reduction(
        self,
        group: list[PoolItem],
        *,
        attempts: int,
        resources_measured: dict[str, Any] | None,
        traceback: str | None,
    ) -> None:
        """A reducer permanent failure: log every file folded into the
        failed group and release the group's own distributor-held results
        (they will never be used again). Unlike a processor permanent
        failure (_give_up_on_file), this never checks failure_proportion -
        the caller always aborts the whole run right after this returns, since
        a partially-folded result can't be trusted not to be corrupted."""
        if self._failure_log is not None:
            allocated = self._distributor.resources("reducer")
            files = frozenset().union(*(item.files for item in group))
            for url in sorted(files):
                self._failure_log.log(
                    FailureRecord(
                        dataset_name=self.dataset_name,
                        filename=url,
                        kind="reducer",
                        attempts=attempts,
                        resources_allocated=allocated,
                        resources_measured=resources_measured,
                        traceback=traceback,
                    )
                )
        for item in group:
            self._release_covered(item)

    def _checkpoint_due(self, since_checkpoint_time: float, since_checkpoint_distance: int) -> bool:
        """Whether enough work has piled up since the last checkpoint - in wall
        time or in un-checkpointed accumulations - to be worth checkpointing
        the reduction about to be submitted. Decided from the group's inputs
        alone, before the reduction itself has run, so this reduction's own
        cost is not part of the decision - only what the reduction it is
        about to fold in already carries. since_checkpoint_time is summed
        across the group (total wall time at stake), while
        since_checkpoint_distance is the max across the group (the worst-off
        ancestor's accumulations-since-checkpoint - "has some ancestor gone
        checkpoint_distance accumulations without being checkpointed").
        When checkpoint_accumulations is set, every accumulation (non-final
        reduction) result is checkpointed regardless of the thresholds."""
        if self._checkpoint_accumulations:
            return True
        if self._checkpoint_time is not None and since_checkpoint_time >= self._checkpoint_time:
            return True
        if (
            self._checkpoint_distance is not None
            and since_checkpoint_distance >= self._checkpoint_distance
        ):
            return True
        return False

    def _checkpoint(self, new_item: PoolItem, inputs: list[PoolItem], is_final: bool) -> None:
        # Phase 1 - locate/create the durable copy.
        if is_final:
            # A final result is vine_reduce's own deliverable, so it gets
            # vine_reduce's own naming/location (results_dir), independent of
            # whatever durable storage the distributor itself used. Nothing
            # will ever reduce it further, so once it is safely on disk here
            # the distributor's every copy of it can go, handle included.
            path = os.path.join(self._results_dir, f"{self.processor_name}__{uuid4().hex}.pkl.zst")
            self._distributor.retrieve(new_item.handle.result_id, path)
            self._distributor.release_result(new_item.handle.result_id)
            new_item.handle = None
        else:
            # The distributor already made this durable at submit time
            # (is_checkpoint=True - see _submit_reduction), so there is
            # nothing to copy, just a path to learn. The handle stays live
            # and reusable as input to a later reduction, so the manager
            # never has to re-send a checkpoint it already generated; it is
            # released only once a FURTHER checkpoint covers this item (via
            # _release_covered) - the other half of invariant #2.
            path = self._distributor.checkpoint_path(new_item.handle.result_id)

        # Phase 2 - record durably, atomically superseding the rows this
        # replaces: the new row and the superseded rows' deletes are one
        # transaction, so this checkpoint event either fully lands or, on a
        # crash mid-way, fully doesn't (never leaves both old and new rows
        # covering the same files on disk).
        superseded = [item.checkpoint.row_id for item in inputs if item.is_checkpointed]
        row_id = self._db.record(
            processor=self.processor_name,
            dataset=self.dataset_name,
            covers_files=new_item.files,
            num_events=new_item.num_events,
            wall_time_s=new_item.wall_time_s,
            memory_mb=new_item.memory_mb,
            is_final=is_final,
            path=path,
            supersedes=superseded,
        )

        # Phase 3 - new_item is durable, so everything it covers can be
        # freed: its inputs' whole not-yet-checkpointed lineage (only ever
        # kept alive as a fallback in case new_item was lost before being
        # checkpointed), and any superseded checkpoint's durable file too -
        # release_result removes that file as part of its contract (the
        # distributor is the single owner of non-final checkpoint files),
        # and the store stopped pointing at it when phase 2 committed.
        for item in inputs:
            self._release_covered(item)
        new_item.inputs = []
        new_item.checkpoint = CheckpointRef(row_id, path)
        new_item.since_checkpoint_time = 0
        new_item.since_checkpoint_distance = 0

    def _release_covered(self, root: PoolItem) -> None:
        """Retention rule (invariant #3): a result may be freed only once a
        durable checkpoint covers it downstream; this is called exactly when
        such a checkpoint lands, once per item folded into it. Frees root
        and its whole not-yet-checkpointed lineage, stopping its descent at
        any item that is itself a checkpoint - that item's own lineage was
        already freed when IT was checkpointed (invariant #2). An explicit
        stack, not recursion, so an arbitrarily long uncheckpointed chain
        cannot hit Python's recursion limit; clearing handle/inputs as it
        goes makes double-release structurally impossible."""
        stack = [root]
        while stack:
            item = stack.pop()
            if item.handle is not None:
                self._distributor.release_result(item.handle.result_id)
                item.handle = None
            if not item.is_checkpointed:
                stack.extend(item.inputs)
            item.inputs = []
