"""Per-(processor, dataset) state: chunk generation, the reduction pool, and
checkpoint bookkeeping. See "Implementation Clarifications" in PLAN.md for
the pooling and checkpointing rules this implements.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from uuid import uuid4

from .checkpoint_db import CheckpointDB, CheckpointRow
from .distributor import Distributor
from .types import Chunk, Outcome, ResourceExhaustion, RuntimeFailure, Success


class VineReduceError(RuntimeError):
    """Raised when a processing or reduction function fails remotely. Carries
    the remote traceback so the failure can be debugged from the local side."""


@dataclass
class PoolItem:
    """A not-yet-final result eligible for reduction: either a single chunk's
    output, or the output of a previous (non-final) reduction call.

    file: the distributor's handle for this item's result file - still a
        live, distributor-native handle even once checkpointed, unless this
        item is a final result (see Pipeline._checkpoint).
    num_events / wall_time_s / memory_mb: totals accumulated into this item.
    files: dataset file URLs whose data this item represents.
    since_checkpoint_time: wall time accumulated since this item (or one of
        its inputs) was last checkpointed - reset to 0 whenever a checkpoint
        is written (see Pipeline._checkpoint).
    since_checkpoint_distance: number of reductions folded into this item's
        lineage since it (or one of its inputs) was last checkpointed - 0 for
        a fresh chunk result, max(group)+1 whenever a group is folded, reset
        to 0 whenever a checkpoint is written (see Pipeline._checkpoint).
    checkpoint_row_id: id of the CheckpointDB row backing this item, if any.
    checkpoint_path: local on-disk path of this item's checkpoint file, once
        checkpointed - distinct from `file`, which stays the distributor's
        own handle so the item can still be folded into a later reduction
        without the manager re-sending it to the cluster. None if never
        checkpointed.
    source_result_id: this item's own distributor result_id, kept alive
        (not released) until it is itself folded into a later reduction.
    inputs: the items folded together to produce this one (empty for a raw
        chunk's output). Kept around so that, if this item is lost before
        anything checkpoints it, it can be recovered by re-folding its
        inputs instead of recomputing everything from scratch - see
        Pipeline._release_uncheckpointed, which walks this recursively to
        release every not-yet-checkpointed ancestor once this item finally
        is checkpointed. Cleared once that happens (see Pipeline._checkpoint)
        - a checkpointed item's own durable copy is backup enough, so there
        is nothing left in its lineage still worth holding onto.
    """

    file: str
    num_events: int
    wall_time_s: float
    memory_mb: float
    files: frozenset[str]
    since_checkpoint_time: float
    since_checkpoint_distance: int
    checkpoint_row_id: int | None = None
    checkpoint_path: str | None = None
    source_result_id: int | None = None
    inputs: list[PoolItem] = field(default_factory=list)


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
        checkpoint_distance: int | None,
        checkpoint_accumulations: bool,
        results_dir: str,
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
        self._checkpoint_distance = checkpoint_distance
        self._checkpoint_accumulations = checkpoint_accumulations
        self._results_dir = os.path.join(results_dir, dataset_name, processor_name)
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
        self._skip_files: set[str] = set()
        self.finished = False

        self._seed_from_checkpoints()
        if not self.finished:
            os.makedirs(self._results_dir, exist_ok=True)

    # -- restart -----------------------------------------------------------

    @staticmethod
    def _pool_item_from_checkpoint(row: CheckpointRow) -> PoolItem:
        # On restart there is no live distributor state to hand back, so the
        # on-disk file doubles as both the distributor-facing handle (to be
        # resubmitted) and the checkpoint's own on-disk record.
        return PoolItem(
            file=row.path,
            num_events=row.num_events,
            wall_time_s=row.wall_time_s,
            memory_mb=row.memory_mb,
            files=frozenset(row.covers_files),
            since_checkpoint_time=0,
            since_checkpoint_distance=0,
            checkpoint_row_id=row.id,
            checkpoint_path=row.path,
        )

    def _seed_from_checkpoints(self) -> None:
        """Restart support: replay the checkpoint rows on file as final results
        and pool items, and record which dataset files they already cover so
        chunk generation can skip them."""
        rows = self._db.checkpoints_for(self.processor_name, self.dataset_name)
        final_rows = [row for row in rows if row.is_final]
        partial_rows = [row for row in rows if not row.is_final]

        all_files = set(self._dataset["files"])
        finalized_files = {url for row in final_rows for url in row.covers_files}
        self.final_results = [self._pool_item_from_checkpoint(row) for row in final_rows]

        if all_files <= finalized_files:
            # Every file already has a final result, so there is nothing left
            # to run and any partial checkpoints are moot.
            self.finished = True
            self._skip_files = all_files
            return

        self.pool = [self._pool_item_from_checkpoint(row) for row in partial_rows]
        covered_by_partial = {url for row in partial_rows for url in row.covers_files}
        self._skip_files = finalized_files | covered_by_partial

    # -- chunk generation ----------------------------------------------------

    def in_flight_count(self) -> int:
        """How many chunk/reduce tasks this pipeline currently has submitted
        and not yet resolved."""
        return len(self._in_flight)

    def owns(self, result_id: int) -> bool:
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
            chunk = self._next_chunk()
            if chunk is None:
                break
            self._submit_chunk(chunk)
            submitted += 1
        return submitted

    def _next_chunk(self) -> Chunk | None:
        """The next chunk to submit - retries first, then freshly generated
        ones - or None when there is nothing left to submit right now."""
        if self._retry_chunks:
            chunk = self._retry_chunks.pop()
            # A retry chunk may predate the last chunksize halving; re-split it
            # so we actually retry at the smaller size, not the size that just
            # failed.
            if self.chunksize is not None and chunk.num_events > self.chunksize:
                split_point = chunk.start + self.chunksize
                self._retry_chunks.append(Chunk(chunk.url, split_point, chunk.stop))
                chunk = Chunk(chunk.url, chunk.start, split_point)
            return chunk

        if self._generator_exhausted:
            return None
        chunk = next(self._generator, None)
        if chunk is None:
            self._generator_exhausted = True
        return chunk

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
        is_checkpoint = is_final or self._checkpoint_due(
            sum(item.since_checkpoint_time for item in group),
            max(item.since_checkpoint_distance for item in group),
        )

        result_id = self._distributor.submit(
            self._reduce_priority,
            self._reduce_category,
            "reducer",
            self._reducer_wrapper,
            self._reducer,
            [item.file for item in group],
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

    # -- outcome handling ------------------------------------------------------

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
        wall_time_s = outcome.resources.get("wall_time_s", 0.0)
        memory_mb = outcome.resources.get("memory_mb", 0.0)

        progress = self._files_in_progress[chunk.url]
        progress.covered_events += chunk.num_events
        progress.staged_items.append(
            PoolItem(
                file=outcome.file,
                num_events=chunk.num_events,
                wall_time_s=wall_time_s,
                memory_mb=memory_mb,
                files=frozenset({chunk.url}),
                since_checkpoint_time=wall_time_s,
                since_checkpoint_distance=0,
                source_result_id=outcome.result_id,
            )
        )
        if progress.covered_events >= progress.num_entries:
            self.pool.extend(progress.staged_items)
            del self._files_in_progress[chunk.url]

    def _handle_reduce_outcome(self, task: _ReduceTask, outcome: Outcome) -> None:
        group = task.group
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
        # group's own source_result_ids are deliberately not released here:
        # new_item isn't durable yet, so if it's lost before something
        # checkpoints it, group (kept below as new_item.inputs) is the only
        # way to recover it without recomputing everything from scratch. See
        # _release_uncheckpointed, invoked from _checkpoint once new_item
        # eventually does become durable.
        wall_time_s = outcome.resources.get("wall_time_s", 0.0)
        memory_mb = outcome.resources.get("memory_mb", 0.0)
        new_item = PoolItem(
            file=outcome.file,
            num_events=task.num_events,
            wall_time_s=task.total_time + wall_time_s,
            memory_mb=task.total_memory + memory_mb,
            files=frozenset().union(*(item.files for item in group)),
            since_checkpoint_time=sum(item.since_checkpoint_time for item in group) + wall_time_s,
            since_checkpoint_distance=max(item.since_checkpoint_distance for item in group) + 1,
            source_result_id=outcome.result_id,
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
        if is_final:
            # A final result is vine_reduce's own deliverable, so it gets
            # vine_reduce's own naming/location (results_dir), independent of
            # whatever durable storage the distributor itself used. Nothing
            # will ever reduce it further, so once it is safely on disk here
            # the cluster-side copy - and the distributor's own durable copy
            # of it, if any - can go.
            dest_path = os.path.join(
                self._results_dir, f"{self.processor_name}__{uuid4().hex}.pkl.zst"
            )
            self._distributor.retrieve(new_item.source_result_id, dest_path)
            self._distributor.release_result(new_item.source_result_id)
            new_item.checkpoint_path = dest_path
            new_item.file = dest_path
            new_item.source_result_id = None
        else:
            # The distributor already made this durable at submit time
            # (is_checkpoint=True - see _submit_reduction), so there is
            # nothing to copy, just a path to learn. file/source_result_id
            # stay live and reusable as input to a later reduction, so the
            # manager never has to re-send a checkpoint it already generated;
            # they are released only once a further checkpoint covers this
            # item (via _release_uncheckpointed, below).
            new_item.checkpoint_path = self._distributor.checkpoint_path(new_item.source_result_id)

        # Batch the new row and the superseded rows' deletes into one
        # transaction, so this checkpoint event either fully lands or, on a
        # crash mid-way, fully doesn't (never leaves both old and new rows
        # covering the same files on disk).
        row_id = self._db.add_checkpoint(
            self.processor_name,
            self.dataset_name,
            sorted(new_item.files),
            new_item.num_events,
            new_item.wall_time_s,
            new_item.memory_mb,
            is_final,
            new_item.checkpoint_path,
            commit=False,
        )
        superseded = [item for item in inputs if item.checkpoint_row_id is not None]
        for item in superseded:
            self._db.delete_checkpoint(item.checkpoint_row_id, commit=False)
        self._db.commit()

        # Only now that the db no longer points at them: delete the
        # superseded checkpoints' own files. Pipeline removes them directly
        # rather than leaving it to release_result, since a restart-seeded
        # checkpoint has no source_result_id for release_result to be called
        # with; release_path is its counterpart, keyed on the path itself,
        # for whatever the distributor may have cached on this item's behalf
        # (see TaskVineDistributor._remap_files) - a harmless no-op for an
        # item that was never restart-seeded, since nothing is ever cached
        # under a this-run item's checkpoint_path (its token is the key
        # instead, already released via release_result elsewhere).
        for item in superseded:
            try:
                os.remove(item.checkpoint_path)
            except FileNotFoundError:
                pass
            self._distributor.release_path(item.checkpoint_path)

        # new_item is now durable (on disk, or the distributor's own
        # permanent copy), so its whole not-yet-checkpointed lineage is safe
        # to release - it was only ever kept alive as a fallback in case
        # new_item itself got lost before being checkpointed.
        for item in inputs:
            self._release_uncheckpointed(item)
        new_item.inputs = []

        new_item.checkpoint_row_id = row_id
        new_item.since_checkpoint_time = 0
        new_item.since_checkpoint_distance = 0

    def _release_uncheckpointed(self, item: PoolItem) -> None:
        """Release item's own cluster-side copy - safe now because the
        caller's own item is durable and covers it, whether item was itself
        a checkpoint being superseded or just an ordinary pooled result.
        Recurses into item's inputs to release the rest of its not-yet-
        checkpointed lineage too, unless item is itself a checkpoint - its
        inputs were already released back when item itself was checkpointed,
        so there is nothing left there to release."""
        if item.source_result_id is not None:
            self._distributor.release_result(item.source_result_id)
        if item.checkpoint_row_id is None:
            for child in item.inputs:
                self._release_uncheckpointed(child)
