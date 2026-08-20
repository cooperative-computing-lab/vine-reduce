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
        def active() -> list[Pipeline]:
            return [p for p in pipelines if not p.finished]

        while True:
            remaining = active()
            for pipeline in remaining:
                pipeline.submit_ready_reductions()
                pipeline.maybe_drain_final_group()
                pipeline.refresh_finished()
            remaining = active()
            if not remaining:
                break

            in_flight_total = sum(p.in_flight_count() for p in pipelines)
            capacity = max(
                0,
                min(
                    distributor.hungry(),
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
