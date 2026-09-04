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

from . import defaults
from .checkpoint_store import CheckpointStore, checksum_dataset
from .distributor import Distributor
from .executor import Executor, SimpleExecutor
from .failure_log import FailureLog
from .pipeline import Pipeline, VineReduceError
from .progress import NullProgressReporter, ProgressReporter
from .size_log import SizeLog

__all__ = ["VineReduce", "VineReduceError"]

# How long to sleep before re-checking state when nothing is in flight and the
# distributor has no capacity to accept more work right now (e.g. TaskVine
# before any worker has connected) - keeps that wait from busy-spinning.
_IDLE_POLL_INTERVAL_S = 0.5


def _resolve_sized_config(
    config: int | dict | None, processor_name: str, dataset_name: str
) -> int | None:
    """Most-specific-wins lookup for chunksize/reduction_size: a per-dataset
    entry beats a per-processor entry, which beats the "default" entry."""
    if config is None or isinstance(config, int):
        return config
    if dataset_name in config.get("datasets", {}):
        return config["datasets"][dataset_name]
    if processor_name in config.get("processors", {}):
        return config["processors"][processor_name]
    return config.get("default")


def _resolve_reduction_size(config: int | dict, processor_name: str, dataset_name: str) -> int:
    """Like _resolve_sized_config, but for reduction_size specifically: unlike
    chunksize, reduction_size has no meaningful "unset" value, so a dict config
    missing a "default" (and no matching per-processor/per-dataset entry) is a
    configuration error, not a silent None - and the resolved size must be
    usable as a fold width (>= 2)."""
    resolved = _resolve_sized_config(config, processor_name, dataset_name)
    if resolved is None or resolved < 2:
        raise VineReduceError(
            f"reduction_size for processor {processor_name!r}, dataset {dataset_name!r} "
            f"resolved to {resolved!r}; it must be an int >= 2. Check the reduction_size "
            f'dict has a "default" entry or a matching "processors"/"datasets" override.'
        )
    return resolved


@dataclass
class VineReduce:
    """Drives a dynamic data reduction computation over one or more datasets: for
    each (processor, dataset) pair, splits every dataset file into chunks,
    runs `processor` over each chunk remotely (map), then repeatedly folds
    pooled outputs together with `reducer` (reduce) until a final result
    covers the whole dataset. Call compute() to run it. See the README's
    Quick Start and PLAN.md for the full design.

    processors: {name: processor_fn} - one Pipeline is built per
        (processor, dataset) pair. processor_fn receives whatever
        chunk_to_args returns for a chunk and runs remotely, at the
        execution site chosen by `executor`.
    input: the dataset description passed to input_to_datasets - by default
        (see defaults.default_input_to_datasets) either an already-parsed
        dict of shape {dataset_name: {"metadata": {...}, "files": {url:
        num_entries}}}, or a path to a json file holding that dict.
    input_to_datasets: parses `input` into that dataset dict shape; defaults
        to defaults.default_input_to_datasets. Runs locally.
    datasets_to_chunks: splits one dataset's files into Chunks; defaults to
        defaults.default_datasets_to_chunks. Runs locally.
    chunk_to_args: turns a Chunk into whatever argument `processor` expects
        (e.g. opening the file and reading events); defaults to
        defaults.default_chunk_to_args, which passes the Chunk through
        unchanged. Runs remotely.
    executor: an Executor instance (submit/map/shutdown, named after
        concurrent.futures.Executor) that runs `processor(args)` at the
        execution site - see executor.py for SimpleExecutor (default),
        CloudpickleExecutor, and DaskExecutor. Configured once here, then
        cloudpickled into every remote call. Runs remotely.
    reducer: folds two processor outputs (or two partial reductions)
        together; must be commutative, associative, and distributive over a
        dataset's chunks. Defaults to defaults.default_reducer (`a += b`).
        Runs remotely.
    reduction_size: how many pooled items a reduction call folds together at
        once. Either a plain int, or a dict of shape
        {"default": int, "processors": {name: int}, "datasets": {name: int}}
        for per-processor/per-dataset overrides (most specific wins).
        Halved automatically (down to a minimum of 2) on resource
        exhaustion.
    is_result: given (num_events, total_wall_time_s, total_memory_mb) for a
        pooled group, returns whether it counts as a final result for that
        (processor, dataset) pair. Defaults to a check that the group covers
        every event in the dataset (see defaults.make_default_is_result).
        Runs locally.
    result_postprocess: applied to a final result just before it's written
        out, e.g. to convert an accumulator into a plainer shape. Runs
        remotely, as part of the reduction call that produces the final
        result.
    checkpoint_time: checkpoint a non-final reduction once at least this
        many seconds of wall time have accumulated in it since its last
        checkpoint. None disables time-based checkpointing.
    checkpoint_distance: checkpoint a non-final reduction once some ancestor
        in the group being folded has gone at least this many accumulations
        (reductions) without being checkpointed. None disables
        distance-based checkpointing.
    checkpoint_accumulations: if True, checkpoint every non-final reduction
        result, regardless of checkpoint_time/checkpoint_distance.
    results_dir: directory final, per-(processor, dataset) results are
        written under, at results_dir/<dataset_name>/<processor_name>/.
        results_dir/size.jsonl also lives here - one line per (processor,
        dataset) pipeline, written as soon as it finishes; see size_log.py.
    distributor: where processor/reducer calls actually run - a Distributor
        implementation such as TaskVineDistributor. Defaults to a
        LocalDistributor (ProcessPoolExecutor-backed) that compute() creates
        and tears down itself; a distributor passed in here is left running
        for the caller to shut down.
    chunksize: target number of events per chunk. Either a plain int, or a
        dict of the same {"default"/"processors"/"datasets"} shape as
        reduction_size. None means one chunk per file. Halved automatically
        on resource exhaustion, taking effect for chunks not yet generated.
    max_chunks_active: cap on chunks in flight (submitted but not yet
        finished) across all pipelines at once.
    max_chunks_cycle: cap on new chunks submitted per scheduling cycle,
        across all pipelines.
    db_path: path to the sqlite checkpoint database; defaults to
        results_dir/vine_reduce.db. Non-final checkpoints are the
        responsibility of the distributor itself (e.g.
        TaskVineDistributor's own checkpoint_dir constructor argument), not
        VineReduce - see PLAN.md's "Temporary Results, Checkpoints, and
        Restart".
    extra_files: local paths made available, under their basename, wherever
        every processor/reducer call runs - see Distributor.add_file().
    environment_variables: environment variables set for every
        processor/reducer call - see Distributor.set_env_var().
    progress: whether to show the live status bars (events, processing,
        reductions, datasets - four per processor) and print one debug line
        per finished processor/reducer task, with its resource usage and
        success/failure status (plus its captured stdout, if any, on
        failure). See progress.py. Defaults to True; set False for a quiet
        run, e.g. under a test harness or a non-interactive batch log.
    attempts: how many times a single chunk (processor call) or reduction
        (reducer call) is tried, total, before giving up - counting both a
        RuntimeFailure (the call raised) and a ResourceExhaustion (the call
        was killed for exceeding its allocation) against the same budget.
        attempts=1 means no retries. A halving of chunksize/reduction_size
        resets the budget for the smaller unit it produces (a fresh start,
        not a strike against it); once a chunk/reduction is already at the
        minimum size (1 event / reduction_size 2) a further
        ResourceExhaustion raises immediately, since there is no smaller
        size left to retry at. Once the budget for a specific chunk is
        exhausted, the file it belongs to is given up on for good (see
        failure_proportion); once the budget for a reduction is exhausted,
        VineReduceError is raised and the whole run stops, unconditionally
        - see Pipeline._handle_chunk_outcome/_handle_reduce_outcome.
    failure_proportion: only applies to a processor permanent failure (a
        chunk that exhausted `attempts`) - a reducer permanent failure
        always stops the run, regardless of this setting, since a
        partially-folded reduction can't be trusted not to be corrupted.
        A file whose chunk permanently fails is logged to failed_files.log
        (dataset, filename, resources allocated/measured, last traceback)
        the moment it happens and removed from that (processor, dataset)
        pipeline for good - never retried, never pooled. The run only
        aborts once, for some dataset, `permanently_failed_files /
        max(files_concluded_so_far, 100) > failure_proportion` - the
        100-file floor means a lone early failure can't spuriously trip a
        nonzero threshold on a small dataset. Must be in [0, 1). The
        default, 0, reproduces the historical behavior: the very first
        permanent processor failure aborts the run (1/100 = 0.01 > 0).
        A run that finishes with any file left permanently unprocessed
        prints a warning in red naming failed_files.log.
    """

    processors: dict[str, Callable[[Any], Any]]
    input: str | dict[str, Any]
    input_to_datasets: Callable[[str | dict[str, Any]], dict[str, Any]] | None = None
    datasets_to_chunks: Callable | None = None
    chunk_to_args: Callable = defaults.default_chunk_to_args
    executor: Executor = field(default_factory=SimpleExecutor)
    reducer: Callable = defaults.default_reducer
    reduction_size: int | dict = 10
    is_result: Callable[[int, float, float], bool] | None = None
    result_postprocess: Callable[[Any], Any] | None = None
    checkpoint_time: float | None = None
    checkpoint_distance: int | None = None
    checkpoint_accumulations: bool = False
    results_dir: str = "results"
    distributor: Distributor | None = None
    chunksize: int | dict | None = None
    max_chunks_active: int = 1000
    max_chunks_cycle: int = 100
    db_path: str | None = None
    extra_files: list[str] = field(default_factory=list)
    environment_variables: dict[str, str] = field(default_factory=dict)
    progress: bool = True
    attempts: int = 3
    failure_proportion: float = 0.0

    def compute(self) -> None:
        """Run the computation to completion: build one Pipeline per
        (processor, dataset) pair (resuming from any checkpoints already on
        disk) and drive them until every pipeline has a final result. If
        `distributor` was not supplied, a LocalDistributor is created for
        this call and shut down again before returning."""
        if not 0.0 <= self.failure_proportion < 1.0:
            raise ValueError(
                f"failure_proportion must be in [0, 1); got {self.failure_proportion!r}"
            )
        with ExitStack() as stack:
            if self.distributor is not None:
                distributor = self.distributor
            else:
                from .local_distributor import LocalDistributor

                # Entered into the stack (unlike a caller-supplied one) so it
                # is shut down again on the way out, however this returns.
                # checkpoint_dir defaults alongside db_path, under results_dir,
                # rather than LocalDistributor's own bare "checkpoints" default
                # - so a checkpoint written by this run is found again next to
                # the checkpoint db that points at it, not wherever the
                # process happened to be started from.
                distributor = stack.enter_context(
                    LocalDistributor(checkpoint_dir=os.path.join(self.results_dir, "checkpoints"))
                )

            # Communicated to the distributor once, up front, so every
            # processor/reducer call it submits from here on has these files and
            # environment variables available - see Distributor.add_file/
            # set_env_var (distributor.py) for what each implementation does
            # with them.
            for path in self.extra_files:
                distributor.add_file(path)
            for name, value in self.environment_variables.items():
                distributor.set_env_var(name, value)

            input_to_datasets = self.input_to_datasets or defaults.default_input_to_datasets
            datasets_to_chunks = self.datasets_to_chunks or defaults.default_datasets_to_chunks

            os.makedirs(self.results_dir, exist_ok=True)
            db = stack.enter_context(
                CheckpointStore(self.db_path or os.path.join(self.results_dir, "vine_reduce.db"))
            )

            datasets = input_to_datasets(self.input)
            for name, dataset in datasets.items():
                db.dataset_changed(name, checksum_dataset(dataset))

            reporter: ProgressReporter | NullProgressReporter = stack.enter_context(
                ProgressReporter() if self.progress else NullProgressReporter()
            )
            # failed_files.log lives at the directory the run was started
            # from, shared across every (processor, dataset) pipeline of
            # this run, and appended to immediately as each permanent
            # processor/reducer failure is found - see failure_log.py and
            # failure_proportion's docstring above.
            failure_log_path = os.path.join(os.getcwd(), "failed_files.log")
            failure_log = FailureLog(failure_log_path)
            # size.jsonl lives at the top of results_dir, shared across every
            # (processor, dataset) pipeline of this run - see size_log.py and
            # Pipeline._log_size_once, the only place it's written to.
            size_log = SizeLog(os.path.join(self.results_dir, "size.jsonl"))
            pipelines = self._build_pipelines(
                datasets, distributor, db, datasets_to_chunks, reporter, failure_log, size_log
            )
            self._run(pipelines, distributor, reporter)

            unprocessed = sum(len(p.failed_files) for p in pipelines)
            if unprocessed:
                from rich.console import Console

                Console().print(
                    f"WARNING: {unprocessed} file(s) were permanently skipped and left "
                    f"unprocessed; see {failure_log_path}",
                    style="bold red",
                )

    def _build_pipelines(
        self,
        datasets: dict[str, Any],
        distributor: Distributor,
        db: CheckpointStore,
        datasets_to_chunks: Callable,
        task_reporter: ProgressReporter | NullProgressReporter,
        failure_log: FailureLog,
        size_log: SizeLog,
    ) -> list[Pipeline]:
        num_processors = len(self.processors)
        pipelines: list[Pipeline] = []
        for index, (proc_name, processor) in enumerate(self.processors.items()):
            # Earlier processors get better (larger) priority; reductions always
            # outrank every processing call, at any processor's priority level.
            process_priority = num_processors - index
            reduce_priority = process_priority + num_processors
            for dataset_name, dataset in datasets.items():
                is_result = self.is_result or defaults.make_default_is_result(
                    sum(dataset["files"].values())
                )
                pipelines.append(
                    Pipeline(
                        processor_name=proc_name,
                        processor=processor,
                        dataset_name=dataset_name,
                        dataset=dataset,
                        distributor=distributor,
                        db=db,
                        datasets_to_chunks=datasets_to_chunks,
                        chunk_to_args=self.chunk_to_args,
                        executor=self.executor,
                        executor_wrapper=defaults.executor_wrapper,
                        reducer=self.reducer,
                        reducer_wrapper=defaults.reducer_wrapper,
                        is_result=is_result,
                        result_postprocess=self.result_postprocess,
                        chunksize=_resolve_sized_config(self.chunksize, proc_name, dataset_name),
                        reduction_size=_resolve_reduction_size(
                            self.reduction_size, proc_name, dataset_name
                        ),
                        checkpoint_time=self.checkpoint_time,
                        checkpoint_distance=self.checkpoint_distance,
                        checkpoint_accumulations=self.checkpoint_accumulations,
                        results_dir=self.results_dir,
                        process_priority=process_priority,
                        reduce_priority=reduce_priority,
                        attempts=self.attempts,
                        failure_proportion=self.failure_proportion,
                        failure_log=failure_log,
                        size_log=size_log,
                        task_reporter=task_reporter,
                    )
                )
        return pipelines

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

            in_flight_total = sum(p.in_flight_count() for p in pipelines)
            budget = max(
                0,
                min(
                    distributor.capacity(),
                    self.max_chunks_active - in_flight_total,
                    self.max_chunks_cycle,
                ),
            )
            # _build_pipelines emits pipelines in descending process_priority
            # order already (outer loop over processors, highest first), and
            # `remaining` is a priority-order-preserving filter of that list,
            # so no re-sort is needed here.
            for pipeline in remaining:
                if budget <= 0:
                    break
                budget -= pipeline.feed(budget)

            if not any(p.in_flight_count() for p in pipelines):
                # Nothing submitted this cycle and nothing pending from before;
                # waiting now would block forever. Sleep briefly rather than
                # busy-spinning re-checking distributor.capacity() (e.g. while
                # waiting for a TaskVine worker to connect), then re-check state.
                time.sleep(_IDLE_POLL_INTERVAL_S)
                continue

            outcome = distributor.wait(timeout=None)
            if outcome is None:
                continue
            pipeline = next(p for p in pipelines if p.owns(outcome.result_id))
            pipeline.handle_outcome(outcome)
            reporter.refresh(pipelines)
