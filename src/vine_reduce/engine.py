"""VineReduce: the orchestration entry point. See PLAN.md for the design.

compute() builds one Pipeline per (processor, dataset) pair, then drives a
single loop: submit ready reductions, feed new chunks up to what the
distributor can take, wait for the next outcome, and let the owning
pipeline react to it. All of the per-pair bookkeeping lives in Pipeline;
this module is just the scheduling loop and priority/config wiring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from . import defaults
from .checkpoint_db import CheckpointDB, checksum_dataset
from .distributor import Distributor
from .executor import simple_executor
from .pipeline import Pipeline, VineReduceError

__all__ = ["VineReduce", "VineReduceError"]


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
    executor: runs `processor(args)` at the execution site - see
        executor.py for simple_executor (default), cloudpickle_executor, and
        dask_executor. Runs remotely.
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
    checkpoint_size: checkpoint a non-final reduction once at least this
        many MB of memory usage have accumulated in it since its last
        checkpoint. None disables size-based checkpointing.
    checkpoint_accumulations: if True, checkpoint every non-final reduction
        result, regardless of checkpoint_time/checkpoint_size.
    checkpoint_dir: directory intermediate (non-final) checkpoints are
        written under.
    checkpoint_retrieve: if True, pull each checkpoint back to this process
        via Distributor.retrieve(); if False, leave it wherever the
        distributor produced it (only meaningful for a distributor that
        keeps its own permanent copy).
    results_dir: directory final, per-(processor, dataset) results are
        written under, at results_dir/<dataset_name>/<processor_name>/.
    results_retrieve: like checkpoint_retrieve, but for final results.
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
        checkpoint_dir/vine_reduce.db.
    extra_files: local paths made available, under their basename, wherever
        every processor/reducer call runs - see Distributor.add_file().
    environment_variables: environment variables set for every
        processor/reducer call - see Distributor.set_env_var().
    """

    processors: dict[str, Callable[[Any], Any]]
    input: str | dict[str, Any]
    input_to_datasets: Callable[[str | dict[str, Any]], dict[str, Any]] | None = None
    datasets_to_chunks: Callable | None = None
    chunk_to_args: Callable = defaults.default_chunk_to_args
    executor: Callable = simple_executor
    reducer: Callable = defaults.default_reducer
    reduction_size: int | dict = 10
    is_result: Callable[[int, float, float], bool] | None = None
    result_postprocess: Callable[[Any], Any] | None = None
    checkpoint_time: float | None = None
    checkpoint_size: float | None = None
    checkpoint_accumulations: bool = False
    checkpoint_dir: str = "checkpoints"
    checkpoint_retrieve: bool = True
    results_dir: str = "results"
    results_retrieve: bool = True
    distributor: Distributor | None = None
    chunksize: int | dict | None = None
    max_chunks_active: int = 1000
    max_chunks_cycle: int = 100
    db_path: str | None = None
    extra_files: list[str] = field(default_factory=list)
    environment_variables: dict[str, str] = field(default_factory=dict)

    def compute(self) -> None:
        """Run the computation to completion: build one Pipeline per
        (processor, dataset) pair (resuming from any checkpoints already on
        disk) and drive them until every pipeline has a final result. If
        `distributor` was not supplied, a LocalDistributor is created for
        this call and shut down again before returning."""
        distributor = self.distributor
        owns_distributor = distributor is None
        if owns_distributor:
            from .local_distributor import LocalDistributor

            distributor = LocalDistributor()

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

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        db = CheckpointDB(self.db_path or os.path.join(self.checkpoint_dir, "vine_reduce.db"))

        try:
            datasets = input_to_datasets(self.input)
            for name, dataset in datasets.items():
                db.dataset_changed(name, checksum_dataset(dataset))

            pipelines = self._build_pipelines(datasets, distributor, db, datasets_to_chunks)
            self._run(pipelines, distributor)
        finally:
            db.close()
            if owns_distributor:
                distributor.shutdown()

    def _build_pipelines(
        self,
        datasets: dict[str, Any],
        distributor: Distributor,
        db: CheckpointDB,
        datasets_to_chunks: Callable,
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
                        reduction_size=_resolve_sized_config(
                            self.reduction_size, proc_name, dataset_name
                        ),
                        checkpoint_time=self.checkpoint_time,
                        checkpoint_size=self.checkpoint_size,
                        checkpoint_accumulations=self.checkpoint_accumulations,
                        checkpoint_dir=self.checkpoint_dir,
                        checkpoint_retrieve=self.checkpoint_retrieve,
                        results_dir=self.results_dir,
                        results_retrieve=self.results_retrieve,
                        process_priority=process_priority,
                        reduce_priority=reduce_priority,
                    )
                )
        return pipelines

    def _run(self, pipelines: list[Pipeline], distributor: Distributor) -> None:
        while True:
            for pipeline in pipelines:
                if pipeline.finished:
                    continue
                pipeline.submit_ready_reductions()
                pipeline.maybe_drain_final_group()
                pipeline.refresh_finished()

            remaining = [p for p in pipelines if not p.finished]
            if not remaining:
                break

            in_flight_total = sum(p.in_flight_count() for p in pipelines)
            capacity = max(
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
                if capacity <= 0:
                    break
                capacity -= pipeline.feed(capacity)

            if sum(p.in_flight_count() for p in pipelines) == 0:
                # Nothing submitted this cycle and nothing pending from before;
                # waiting now would block forever. Loop back and re-check state.
                continue

            outcome = distributor.wait(timeout=None)
            if outcome is None:
                continue
            pipeline = next(p for p in pipelines if p.owns(outcome.result_id))
            pipeline.handle_outcome(outcome.result_id, outcome)
