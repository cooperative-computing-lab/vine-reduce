from __future__ import annotations

import os

from vine_reduce import defaults, serialization
from vine_reduce.checkpoint_db import CheckpointDB
from vine_reduce.executor import simple_executor
from vine_reduce.pipeline import Pipeline

from helpers import count_events, sum_reducer


def flaky_processor(chunk):
    """Exhausts unless the chunk has shrunk to <= 2 events. Only ever run
    in-process via FakeDistributor, so it doesn't need to be picklable."""
    if chunk.num_events > 2:
        raise MemoryError("too big")
    return chunk.num_events


def make_pipeline(
    fake_distributor,
    tmp_path,
    dataset,
    *,
    processor=count_events,
    reducer=sum_reducer,
    reduction_size=10,
    chunksize=None,
    checkpoint_time=None,
    checkpoint_size=None,
    checkpoint_accumulations=False,
    db=None,
    dataset_name="ds",
):
    db = db or CheckpointDB(str(tmp_path / "db.sqlite"))
    total_events = sum(dataset["files"].values())
    return (
        Pipeline(
            processor_name="proc",
            processor=processor,
            dataset_name=dataset_name,
            dataset=dataset,
            distributor=fake_distributor,
            db=db,
            datasets_to_chunks=defaults.default_datasets_to_chunks,
            chunk_to_args=defaults.default_chunk_to_args,
            executor=simple_executor,
            executor_wrapper=defaults.executor_wrapper,
            reducer=reducer,
            reducer_wrapper=defaults.reducer_wrapper,
            is_result=defaults.make_default_is_result(total_events),
            result_postprocess=None,
            chunksize=chunksize,
            reduction_size=reduction_size,
            checkpoint_time=checkpoint_time,
            checkpoint_size=checkpoint_size,
            checkpoint_accumulations=checkpoint_accumulations,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            checkpoint_retrieve=True,
            results_dir=str(tmp_path / "results"),
            results_retrieve=True,
            process_priority=1,
            reduce_priority=2,
        ),
        db,
    )


def run_to_completion(pipeline, distributor, max_cycles=1000):
    for _ in range(max_cycles):
        if pipeline.finished:
            return
        pipeline.submit_ready_reductions()
        pipeline.maybe_drain_final_group()
        pipeline.refresh_finished()
        if pipeline.finished:
            return
        pipeline.feed(100)
        outcome = distributor.wait()
        if outcome is not None:
            pipeline.handle_outcome(outcome.result_id, outcome)
    raise AssertionError("pipeline did not finish within max_cycles")


def final_value(pipeline):
    assert len(pipeline.final_results) == 1
    return serialization.load(pipeline.final_results[0].file)


def test_pools_across_files_and_produces_one_final_result(fake_distributor, tmp_path):
    dataset = {"files": {"a.root": 5, "b.root": 5}}
    pipeline, db = make_pipeline(fake_distributor, tmp_path, dataset, reduction_size=10)

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 10
    assert pipeline.final_results[0].num_events == 10
    db.close()


def test_small_reduction_size_forces_intermediate_reductions(fake_distributor, tmp_path):
    dataset = {"files": {"a.root": 1, "b.root": 1, "c.root": 1}}
    pipeline, db = make_pipeline(fake_distributor, tmp_path, dataset, reduction_size=2)

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 3
    assert pipeline.final_results[0].num_events == 3
    db.close()


def test_single_chunk_dataset_reduces_as_final_checkpoint(fake_distributor, tmp_path):
    dataset = {"files": {"a.root": 7}}
    pipeline, db = make_pipeline(fake_distributor, tmp_path, dataset, reduction_size=10)

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 7
    db.close()


def test_checkpoint_time_threshold_persists_and_supersedes_intermediates(
    fake_distributor, tmp_path
):
    dataset = {"files": {"a.root": 1, "b.root": 1, "c.root": 1}}
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, reduction_size=2, checkpoint_time=0
    )

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 3
    rows = db.checkpoints_for("proc", "ds")
    # every intermediate checkpoint should have been superseded and deleted,
    # leaving only the final one.
    assert len(rows) == 1
    assert rows[0].is_final is True
    # and its file, plus every superseded checkpoint file, should be cleaned
    # off disk (final results live in results_dir, not checkpoint_dir).
    assert os.listdir(str(tmp_path / "checkpoints")) == []
    db.close()


def test_checkpoint_accumulations_checkpoints_every_intermediate_reduction(
    fake_distributor, tmp_path
):
    dataset = {"files": {"a.root": 1, "b.root": 1, "c.root": 1, "d.root": 1}}
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, reduction_size=2, checkpoint_accumulations=True
    )
    checkpoint_calls = []
    original_checkpoint = pipeline._checkpoint

    def spy_checkpoint(new_item, inputs, is_final):
        checkpoint_calls.append(is_final)
        original_checkpoint(new_item, inputs, is_final)

    pipeline._checkpoint = spy_checkpoint

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 4
    # a.root+b.root and c.root+d.root each reduce to a non-final checkpoint,
    # then those two reduce to the final one - three reductions, all checkpointed.
    assert checkpoint_calls == [False, False, True]
    db.close()


def test_restart_skips_files_covered_by_a_non_final_checkpoint(fake_distributor, tmp_path):
    dataset = {"files": {"a.root": 5, "b.root": 5}}
    db = CheckpointDB(str(tmp_path / "db.sqlite"))

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    seeded_file = checkpoint_dir / "seeded.pkl.zst"
    serialization.dump(100, str(seeded_file))  # stands in for a's "already processed" result
    db.add_checkpoint("proc", "ds", ["a.root"], 5, 1.0, 1.0, False, str(seeded_file))

    pipeline, _ = make_pipeline(fake_distributor, tmp_path, dataset, reduction_size=10, db=db)

    assert pipeline._skip_files == {"a.root"}
    assert len(pipeline.pool) == 1
    assert pipeline.pool[0].file == str(seeded_file)

    run_to_completion(pipeline, fake_distributor)

    # 100 (seeded, standing in for a.root) + 5 (b.root, actually processed)
    assert final_value(pipeline) == 105
    db.close()


def test_restart_with_final_checkpoint_for_all_files_skips_pipeline_entirely(
    fake_distributor, tmp_path
):
    dataset = {"files": {"a.root": 5, "b.root": 5}}
    db = CheckpointDB(str(tmp_path / "db.sqlite"))
    db.add_checkpoint("proc", "ds", ["a.root", "b.root"], 10, 1.0, 1.0, True, "/tmp/final.pkl")

    pipeline, _ = make_pipeline(fake_distributor, tmp_path, dataset, db=db)

    assert pipeline.finished is True
    assert pipeline.pool == []
    assert pipeline.in_flight_count() == 0
    assert len(pipeline.final_results) == 1
    db.close()


def test_resource_exhaustion_halves_chunksize_and_eventually_succeeds(fake_distributor, tmp_path):
    dataset = {"files": {"a.root": 8}}
    pipeline, db = make_pipeline(
        fake_distributor,
        tmp_path,
        dataset,
        processor=flaky_processor,
        chunksize=8,
        reduction_size=10,
    )

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 8
    assert pipeline.chunksize < 8
    db.close()
