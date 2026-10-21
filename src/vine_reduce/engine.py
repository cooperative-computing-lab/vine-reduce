"""VineReduce: the orchestration entry point. See PLAN.md for the design.

compute() builds one Pipeline per (processor, dataset) pair, then drives a
single loop: submit ready reductions, feed new chunks up to what the
distributor can take, wait for the next outcome, and let the owning
pipeline react to it. All of the per-pair bookkeeping lives in Pipeline;
this module is just the scheduling loop and priority/config wiring.
"""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console

from . import defaults
from .checkpoint_store import CheckpointStore, checksum_dataset
from .distributor import Distributor
from .executor import Executor, SimpleExecutor
from .failure_log import FailureLog
from .local_distributor import LocalDistributor
from .pipeline import Pipeline, VineReduceError
from .progress import NullProgressReporter, ProgressReporter
from .size_log import SizeLog

__all__ = ["VineReduce", "VineReduceError"]

# How long to sleep before re-checking state when nothing is in flight and the
# distributor has no capacity to accept more work right now (e.g. TaskVine
# before any worker has connected) - keeps that wait from busy-spinning.
_IDLE_POLL_INTERVAL_S = 0.5


@dataclass
class VineReduce:
    """Drives a dynamic data reduction over one or more datasets: for each
    (processor, dataset) pair, splits files into chunks, runs `processor` on
    each chunk remotely (map), and folds pooled outputs with `reducer` until
    final results cover the dataset. Call compute() to run it. See the README
    and PLAN.md.

    "Locally"/"remotely" below say where a callable runs.

    processors: {name: fn}; one Pipeline per (processor, dataset). fn receives
        whatever chunk_to_args returns. Remote.
    input: passed to input_to_datasets; by default a dict
        {dataset: {"metadata": {...}, "files": {url: num_entries}}} or a path
        to a json file holding one.
    input_to_datasets: parses `input` into that shape. Local.
    datasets_to_chunks: (dataset, current_chunksize) -> Chunks. Local.
    chunk_to_args: (chunk, dataset_metadata, distributor_metadata) -> the
        processor's argument; default passes the Chunk through. Remote.
    executor: runs processor(args) at the execution site (see executor.py);
        cloudpickled into every call. Remote.
    reducer: folds two outputs; must be commutative and associative.
        Default `a += b`. Remote.
    reduction_size: items folded per reduction call; an int or
        {"default": n, "processors": {name: n}, "datasets": {name: n}} (most
        specific wins). Must resolve to >= 2. Halved on resource exhaustion,
        down to minimum_reduction_size (clamped to [2, reduction_size]).
    is_result: (num_events, total_wall_time_s, total_memory_mb) of a group ->
        whether its reduction is a final result. Default: the group covers
        the whole dataset. Local.
    result_postprocess: applied to a final result inside the reduction that
        produces it. Remote.
    result_postprocess_offload: result_postprocess returns a JSON-serializable
        locator of data it stored elsewhere; a `<processor>__<uuid>.pointer.json`
        is written to results_dir instead of the result. Requires
        result_postprocess.
    checkpoint_time: checkpoint a reduction once its group carries at least
        this many seconds of wall time since the last checkpoint. None: off.
    checkpoint_distance: checkpoint a reduction once some item in its group
        has gone this many folds without a checkpoint; 0 checkpoints every
        reduction. None: off.
    checkpoint_accumulations: checkpoint every reduction; same as
        checkpoint_distance=0.
    results_dir: final results go to results_dir/<dataset>/<processor>/;
        size.jsonl (see size_log.py) also lives here.
    checkpoint_dir: where durable checkpoints are written (default
        results_dir/checkpoints); handed to Distributor.set_checkpoint_dir().
    distributor: where calls run. Default: a LocalDistributor created and
        shut down by compute(); a supplied one is left running.
    chunksize: events per chunk, same int-or-dict shape as reduction_size;
        None means one chunk per file. Halved on resource exhaustion, down
        to minimum_chunksize (at least 1); affects chunks not yet generated.
    max_chunks_active: cap on chunks in flight across all pipelines.
    max_chunks_cycle: cap on chunks submitted per scheduling cycle.
    db_path: sqlite checkpoint database; default results_dir/vine_reduce.db.
    extra_files / environment_variables: made available to every call (see
        Distributor.add_file/set_env_var).
    progress: show live status bars and one line per finished task.
    attempts: tries per chunk or reduction, counting failures and resource
        exhaustions alike. Halving the size gives the smaller unit a fresh
        budget; exhaustion at the minimum size gives up at once. A chunk that
        gives up is dropped (see failure_proportion); a reduction that gives
        up raises VineReduceError.
    failure_proportion: in [0, 1). Chunks that give up are logged to
        ./failed_files.log and left as holes (a restart retries them); the
        run aborts once a dataset's given-up events exceed this fraction of
        its total events. Default 0: the first failure aborts.
    """

    processors: dict[str, Callable[[Any], Any]]
    input: str | dict[str, Any]
    input_to_datasets: Callable[[str | dict[str, Any]], dict[str, Any]] = (
        defaults.default_input_to_datasets
    )
    datasets_to_chunks: Callable = defaults.default_datasets_to_chunks
    chunk_to_args: Callable = defaults.default_chunk_to_args
    executor: Executor = field(default_factory=SimpleExecutor)
    reducer: Callable = defaults.default_reducer
    reduction_size: int | dict = 10
    minimum_reduction_size: int = 2
    is_result: Callable[[int, float, float], bool] | None = None
    result_postprocess: Callable[[Any], Any] | None = None
    result_postprocess_offload: bool = False
    checkpoint_time: float | None = None
    checkpoint_distance: int | None = None
    checkpoint_accumulations: bool = False
    results_dir: str = "results"
    checkpoint_dir: str | None = None
    distributor: Distributor | None = None
    chunksize: int | dict | None = None
    minimum_chunksize: int = 1000
    max_chunks_active: int = 1000
    max_chunks_cycle: int = 100
    db_path: str | None = None
    extra_files: list[str] = field(default_factory=list)
    environment_variables: dict[str, str] = field(default_factory=dict)
    progress: bool = True
    attempts: int = 3
    failure_proportion: float = 0.0

    def __post_init__(self) -> None:
        if self.checkpoint_accumulations:
            self.checkpoint_distance = 0

    def compute(self) -> None:
        """Run to completion, resuming from any checkpoints on disk."""
        if not 0.0 <= self.failure_proportion < 1.0:
            raise ValueError(
                f"failure_proportion must be in [0, 1); got {self.failure_proportion!r}"
            )
        if self.result_postprocess_offload and self.result_postprocess is None:
            raise VineReduceError(
                "result_postprocess_offload=True requires result_postprocess to be set"
            )
        with ExitStack() as stack:
            distributor = (
                self.distributor
                if self.distributor is not None
                else stack.enter_context(LocalDistributor())
            )
            distributor.set_checkpoint_dir(
                self.checkpoint_dir or os.path.join(self.results_dir, "checkpoints")
            )
            for path in self.extra_files:
                distributor.add_file(path)
            for name, value in self.environment_variables.items():
                distributor.set_env_var(name, value)

            os.makedirs(self.results_dir, exist_ok=True)
            db = stack.enter_context(
                CheckpointStore(self.db_path or os.path.join(self.results_dir, "vine_reduce.db"))
            )

            datasets = self.input_to_datasets(self.input)
            for name, dataset in datasets.items():
                discarded_paths = db.reset_if_dataset_changed(name, checksum_dataset(dataset))
                for path in discarded_paths:
                    if os.path.exists(path):
                        os.remove(path)

            reporter: ProgressReporter | NullProgressReporter = stack.enter_context(
                ProgressReporter() if self.progress else NullProgressReporter()
            )
            failure_log_path = os.path.join(os.getcwd(), "failed_files.log")
            failure_log = FailureLog(failure_log_path)
            size_log = SizeLog(os.path.join(self.results_dir, "size.jsonl"))
            pipelines = self._build_pipelines(
                datasets, distributor, db, reporter, failure_log, size_log
            )
            self._run(pipelines, distributor, reporter)
            self._warn_unprocessed(pipelines, failure_log_path)

    @staticmethod
    def _warn_unprocessed(pipelines: list[Pipeline], failure_log_path: str) -> None:
        """Warn in red if any chunk was given up on this run."""
        failed = [chunk for p in pipelines for chunk in p.failed_chunks]
        if failed:
            events = sum(chunk.num_events for chunk in failed)
            Console().print(
                f"WARNING: {len(failed)} chunk(s) ({events} event(s)) were permanently "
                f"skipped and left unprocessed; see {failure_log_path}",
                style="bold red",
            )

    def _build_pipelines(
        self,
        datasets: dict[str, Any],
        distributor: Distributor,
        db: CheckpointStore,
        reporter: ProgressReporter | NullProgressReporter,
        failure_log: FailureLog,
        size_log: SizeLog,
    ) -> list[Pipeline]:
        """One Pipeline per (processor, dataset), in descending priority:
        earlier processors, then earlier datasets, rank higher, and every
        reduction outranks every processing call (see PLAN.md's
        "Priorities")."""
        total_slots = len(self.processors) * len(datasets)
        pairs = [(p, d) for p in self.processors for d in datasets]
        return [
            Pipeline(
                self,
                processor_name=proc_name,
                dataset_name=dataset_name,
                dataset=datasets[dataset_name],
                distributor=distributor,
                db=db,
                process_priority=total_slots - rank,
                reduce_priority=2 * total_slots - rank,
                failure_log=failure_log,
                size_log=size_log,
                task_reporter=reporter,
            )
            for rank, (proc_name, dataset_name) in enumerate(pairs)
        ]

    def _run(
        self,
        pipelines: list[Pipeline],
        distributor: Distributor,
        reporter: ProgressReporter | NullProgressReporter,
    ) -> None:
        while True:
            for pipeline in pipelines:
                if pipeline.finished:
                    continue
                pipeline.submit_ready_reductions()
                pipeline.maybe_drain_final_group()
                pipeline.refresh_finished()

            reporter.refresh(pipelines)

            remaining = [p for p in pipelines if not p.finished]
            if not remaining:
                reporter.refresh(pipelines, force=True)
                break

            in_flight_total = sum(p.in_flight_count for p in pipelines)
            budget = max(
                0,
                min(
                    distributor.capacity(),
                    self.max_chunks_active - in_flight_total,
                    self.max_chunks_cycle,
                ),
            )
            # pipelines are already in descending priority (see _build_pipelines).
            for pipeline in remaining:
                if budget <= 0:
                    break
                budget -= pipeline.feed(budget)

            if not any(p.in_flight_count for p in pipelines):
                # Nothing in flight, so wait() would block forever; e.g. no
                # TaskVine worker has connected yet. Poll without spinning.
                time.sleep(_IDLE_POLL_INTERVAL_S)
                continue

            outcome = distributor.wait(timeout=None)
            if outcome is None:
                continue
            pipeline = next(p for p in pipelines if p.owns(outcome.result_id))
            pipeline.handle_outcome(outcome)
