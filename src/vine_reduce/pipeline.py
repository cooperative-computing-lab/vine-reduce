"""Per-(processor, dataset) state: chunk generation, the reduction pool, and
checkpoint bookkeeping. See "Implementation Clarifications" in PLAN.md for
the pooling and checkpointing rules this implements.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from uuid import uuid4

from .checkpoint_db import CheckpointDB
from .distributor import Distributor
from .types import Chunk, Outcome, ResourceExhaustion, RuntimeFailure, Success


class VineReduceError(RuntimeError):
    """Raised when a processing or reduction function fails remotely. Carries
    the remote traceback so the failure can be debugged from the local side."""


@dataclass
class PoolItem:
    """A not-yet-final result eligible for reduction: either a single chunk's
    output, or the output of a previous (non-final) reduction call."""

    file: str
    num_events: int
    wall_time_s: float
    memory_mb: float
    files: frozenset[str]  # dataset file URLs whose data this item represents
    since_checkpoint_time: float
    since_checkpoint_memory: float
    checkpoint_row_id: int | None = None
    source_result_id: int | None = None  # distributor result_id to free once consumed


@dataclass
class _FileProgress:
    num_entries: int
    covered_events: int = 0
    staged_items: list[PoolItem] = field(default_factory=list)


@dataclass
class _ChunkTask:
    chunk: Chunk


@dataclass
class _ReduceTask:
    group: list[PoolItem]
    is_final: bool


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
        db: CheckpointDB,
        datasets_to_chunks: Callable[[dict, Callable[[], int | None], set[str]], Iterator[Chunk]],
        chunk_to_args: Callable,
        executor: Callable,
        executor_wrapper: Callable,
        reducer: Callable,
        reducer_wrapper: Callable,
        is_result: Callable[[int, float, float], bool],
        result_postprocess: Callable | None,
        chunksize: int | None,
        reduction_size: int,
        checkpoint_time: float | None,
        checkpoint_size: float | None,
        checkpoint_dir: str,
        checkpoint_retrieve: bool,
        results_dir: str,
        results_retrieve: bool,
        process_priority: int,
        reduce_priority: int,
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
        self._checkpoint_size = checkpoint_size
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_retrieve = checkpoint_retrieve
        self._results_dir = os.path.join(results_dir, dataset_name)
        self._results_retrieve = results_retrieve
        self._process_priority = process_priority
        self._reduce_priority = reduce_priority
        self._process_category = f"{processor_name}:{dataset_name}:process"
        self._reduce_category = f"{processor_name}:{dataset_name}:reduce"

        self.pool: list[PoolItem] = []
        self.final_results: list[PoolItem] = []
        self._files_in_progress: dict[str, _FileProgress] = {}
        self._retry_chunks: list[Chunk] = []
        self._in_flight: dict[int, _ChunkTask | _ReduceTask] = {}
        self._generator_exhausted = False
        self._generator: Iterator[Chunk] | None = None
        self.finished = False

        self._seed_from_checkpoints()
        if not self.finished:
            os.makedirs(self._results_dir, exist_ok=True)
            if self._checkpoint_retrieve:
                os.makedirs(self._checkpoint_dir, exist_ok=True)

    # -- restart -----------------------------------------------------------

    def _seed_from_checkpoints(self) -> None:
        rows = self._db.checkpoints_for(self.processor_name, self.dataset_name)
        all_files = set(self._dataset["files"].keys())

        finalized_files: set[str] = set()
        for row in rows:
            if row.is_final:
                finalized_files |= set(row.covers_files)
                self.final_results.append(
                    PoolItem(
                        file=row.path,
                        num_events=row.num_events,
                        wall_time_s=row.wall_time_s,
                        memory_mb=row.memory_mb,
                        files=frozenset(row.covers_files),
                        since_checkpoint_time=0,
                        since_checkpoint_memory=0,
                        checkpoint_row_id=row.id,
                    )
                )

        remaining_files = all_files - finalized_files
        if not remaining_files:
            self.finished = True
            self._skip_files = all_files
            return

        covered_by_nonfinal: set[str] = set()
        for row in rows:
            if row.is_final:
                continue
            self.pool.append(
                PoolItem(
                    file=row.path,
                    num_events=row.num_events,
                    wall_time_s=row.wall_time_s,
                    memory_mb=row.memory_mb,
                    files=frozenset(row.covers_files),
                    since_checkpoint_time=0,
                    since_checkpoint_memory=0,
                    checkpoint_row_id=row.id,
                )
            )
            covered_by_nonfinal |= set(row.covers_files)

        self._skip_files = finalized_files | covered_by_nonfinal

    # -- chunk generation ----------------------------------------------------

    @property
    def total_events(self) -> int:
        return sum(self._dataset["files"].values())

    @property
    def process_priority(self) -> int:
        return self._process_priority

    def in_flight_count(self) -> int:
        return len(self._in_flight)

    def owns(self, result_id: int) -> bool:
        return result_id in self._in_flight

    def refresh_finished(self) -> None:
        """Catches pipelines that are done without ever producing an outcome
        to react to, e.g. an empty dataset, or one fully covered by seeded
        checkpoints for every file but not yet marked finished at construction."""
        if (
            not self.finished
            and self.chunks_all_done
            and not self.pool
            and self.in_flight_count() == 0
        ):
            self.finished = True

    @property
    def chunks_all_done(self) -> bool:
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
            if self._retry_chunks:
                chunk = self._retry_chunks.pop()
                # A retry chunk may predate the last chunksize halving; re-split
                # it so we actually retry at the smaller size, not the size that
                # just failed.
                if self.chunksize is not None and chunk.num_events > self.chunksize:
                    split_point = chunk.start + self.chunksize
                    self._retry_chunks.append(Chunk(chunk.url, split_point, chunk.stop))
                    chunk = Chunk(chunk.url, chunk.start, split_point)
            elif not self._generator_exhausted:
                chunk = next(self._generator, None)
                if chunk is None:
                    self._generator_exhausted = True
                    continue
            else:
                break

            self._submit_chunk(chunk)
            submitted += 1
        return submitted

    def _submit_chunk(self, chunk: Chunk) -> None:
        self._files_in_progress.setdefault(
            chunk.url, _FileProgress(num_entries=self._dataset["files"][chunk.url])
        )
        result_id = self._distributor.submit(
            self._process_priority,
            self._process_category,
            "processor",
            self._executor_wrapper,
            self._processor,
            chunk,
            self._dataset_metadata,
            None,
            None,
            self._chunk_to_args,
            self._executor,
        )
        self._in_flight[result_id] = _ChunkTask(chunk=chunk)

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

        result_id = self._distributor.submit(
            self._reduce_priority,
            self._reduce_category,
            "reducer",
            self._reducer_wrapper,
            self._reducer,
            [item.file for item in group],
            is_final,
            self._result_postprocess,
        )
        self._in_flight[result_id] = _ReduceTask(group=group, is_final=is_final)

    # -- outcome handling ------------------------------------------------------

    def handle_outcome(self, result_id: int, outcome: Outcome) -> None:
        task = self._in_flight.pop(result_id)
        if isinstance(task, _ChunkTask):
            self._handle_chunk_outcome(task, outcome)
        else:
            self._handle_reduce_outcome(task, outcome)
        self.finished = self.finished or (
            self.chunks_all_done and not self.pool and self.in_flight_count() == 0
        )

    def _handle_chunk_outcome(self, task: _ChunkTask, outcome: Outcome) -> None:
        chunk = task.chunk
        if isinstance(outcome, RuntimeFailure):
            raise VineReduceError(
                f"processor {self.processor_name!r} failed on "
                f"{chunk.url}[{chunk.start}:{chunk.stop}]:\n{outcome.traceback}"
            )
        if isinstance(outcome, ResourceExhaustion):
            self.chunksize = max(1, (self.chunksize or chunk.num_events) // 2)
            self._retry_chunks.append(chunk)
            return

        assert isinstance(outcome, Success)
        progress = self._files_in_progress[chunk.url]
        progress.covered_events += chunk.num_events
        progress.staged_items.append(
            PoolItem(
                file=outcome.file,
                num_events=chunk.num_events,
                wall_time_s=outcome.resources.get("wall_time_s", 0.0),
                memory_mb=outcome.resources.get("memory_mb", 0.0),
                files=frozenset({chunk.url}),
                since_checkpoint_time=outcome.resources.get("wall_time_s", 0.0),
                since_checkpoint_memory=outcome.resources.get("memory_mb", 0.0),
                source_result_id=outcome.result_id,
            )
        )
        if progress.covered_events >= progress.num_entries:
            self.pool.extend(progress.staged_items)
            del self._files_in_progress[chunk.url]

    def _handle_reduce_outcome(self, task: _ReduceTask, outcome: Outcome) -> None:
        group, is_final = task.group, task.is_final
        if isinstance(outcome, RuntimeFailure):
            raise VineReduceError(
                f"reducer for {self.processor_name!r}/{self.dataset_name!r} failed:\n"
                f"{outcome.traceback}"
            )
        if isinstance(outcome, ResourceExhaustion):
            self.reduction_size = max(2, self.reduction_size // 2)
            self.pool[:0] = group  # retry with a (now smaller) reduction_size next cycle
            return

        assert isinstance(outcome, Success)
        for item in group:
            if item.source_result_id is not None:
                self._distributor.free_result(item.source_result_id)

        new_item = PoolItem(
            file=outcome.file,
            num_events=sum(item.num_events for item in group),
            wall_time_s=sum(item.wall_time_s for item in group)
            + outcome.resources.get("wall_time_s", 0.0),
            memory_mb=sum(item.memory_mb for item in group)
            + outcome.resources.get("memory_mb", 0.0),
            files=frozenset().union(*(item.files for item in group)),
            since_checkpoint_time=sum(item.since_checkpoint_time for item in group)
            + outcome.resources.get("wall_time_s", 0.0),
            since_checkpoint_memory=sum(item.since_checkpoint_memory for item in group)
            + outcome.resources.get("memory_mb", 0.0),
            source_result_id=outcome.result_id,
        )

        crosses_checkpoint = is_final or (
            (
                self._checkpoint_time is not None
                and new_item.since_checkpoint_time >= self._checkpoint_time
            )
            or (
                self._checkpoint_size is not None
                and new_item.since_checkpoint_memory >= self._checkpoint_size
            )
        )
        if crosses_checkpoint:
            self._checkpoint(new_item, group, is_final)

        if is_final:
            self.final_results.append(new_item)
        else:
            self.pool.append(new_item)

    def _checkpoint(self, new_item: PoolItem, inputs: list[PoolItem], is_final: bool) -> None:
        dest_dir = self._results_dir if is_final else self._checkpoint_dir
        should_retrieve = self._results_retrieve if is_final else self._checkpoint_retrieve

        if should_retrieve:
            dest_path = os.path.join(dest_dir, f"{self.processor_name}__{uuid4().hex}.pkl.zst")
            self._distributor.retrieve(new_item.source_result_id, dest_path)
            self._distributor.free_result(new_item.source_result_id)
            new_item.file = dest_path
            new_item.source_result_id = None
        # else: the distributor keeps its own permanent copy; nothing to move.

        row_id = self._db.add_checkpoint(
            self.processor_name,
            self.dataset_name,
            sorted(new_item.files),
            new_item.num_events,
            new_item.wall_time_s,
            new_item.memory_mb,
            is_final,
            new_item.file,
        )

        for item in inputs:
            if item.checkpoint_row_id is not None:
                self._db.delete_checkpoint(item.checkpoint_row_id)
                if item.file.startswith(self._checkpoint_dir + os.sep):
                    try:
                        os.remove(item.file)
                    except FileNotFoundError:
                        pass

        new_item.checkpoint_row_id = row_id
        new_item.since_checkpoint_time = 0
        new_item.since_checkpoint_memory = 0
