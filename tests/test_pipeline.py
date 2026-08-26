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


def test_intermediate_checkpoint_keeps_its_cluster_copy_until_superseded(
    fake_distributor, tmp_path
):
    """A non-final checkpoint must stay usable as a reduction input without the
    manager re-sending it: its distributor handle/source_result_id must not be
    released at checkpoint time, only later once a further reduction actually
    folds it in and supersedes it - the same point any other pooled item's
    source_result_id is released."""
    dataset = {"files": {"a.root": 1, "b.root": 1, "c.root": 1, "d.root": 1}}
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, reduction_size=2, checkpoint_accumulations=True
    )

    released: list[int] = []
    original_release = fake_distributor.release_result

    def spy_release(result_id):
        released.append(result_id)
        original_release(result_id)

    fake_distributor.release_result = spy_release

    intermediate_items = []
    original_checkpoint = pipeline._checkpoint

    def spy_checkpoint(new_item, inputs, is_final):
        before = len(released)
        original_checkpoint(new_item, inputs, is_final)
        newly_released = released[before:]
        if is_final:
            # nothing reduces a final result further, so it's fully detached
            # from the distributor once safely on disk.
            assert new_item.source_result_id is None
            assert new_item.file == new_item.checkpoint_path
        else:
            # checkpointing must not release the cluster-side copy, or a
            # later reduction folding this item in would need the manager
            # to re-send it - it must still be a live, distributor-native
            # handle, distinct from the on-disk checkpoint file.
            assert new_item.source_result_id is not None
            assert new_item.source_result_id not in newly_released
            assert new_item.checkpoint_path is not None
            assert new_item.file != new_item.checkpoint_path
            assert os.path.exists(new_item.checkpoint_path)
            intermediate_items.append(new_item)

    pipeline._checkpoint = spy_checkpoint

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 4
    assert len(intermediate_items) == 2  # a+b, c+d
    # each intermediate checkpoint's cluster copy is only released once a
    # later reduction (here, the final one) actually folds it in.
    for item in intermediate_items:
        assert item.source_result_id in released
    db.close()


def test_uncheckpointed_fold_keeps_every_ancestor_until_a_later_checkpoint_covers_them(
    fake_distributor, tmp_path
):
    """With no checkpoint thresholds set at all, only the final reduction is a
    checkpoint - a.root+b.root and c.root+d.root each fold into a non-final,
    non-checkpointed intermediate result first. None of the four raw chunk
    results (nor the two intermediate fold results) may be released at those
    intermediate folds: if the intermediate result is lost before anything
    checkpoints it, its inputs are the only way to recover it without
    recomputing everything. They should all be released together, in one
    shot, once the final reduction folds them in and checkpoints - covering
    the whole lineage recursively, not just its immediate inputs."""
    dataset = {"files": {"a.root": 1, "b.root": 1, "c.root": 1, "d.root": 1}}
    pipeline, db = make_pipeline(fake_distributor, tmp_path, dataset, reduction_size=2)

    released: list[int] = []
    original_release = fake_distributor.release_result

    def spy_release(result_id):
        released.append(result_id)
        original_release(result_id)

    fake_distributor.release_result = spy_release

    def count_lineage(item):
        return len(item.inputs) + sum(count_lineage(child) for child in item.inputs)

    checkpoint_calls = []
    original_checkpoint = pipeline._checkpoint

    def spy_checkpoint(new_item, inputs, is_final):
        # nothing should have been released before this - the sole -
        # checkpoint call fires.
        assert released == []
        # the final fold's lineage must cover the whole tree: the two
        # intermediate folds' own inputs, plus the four raw chunk results
        # each of them was still carrying - none released until now.
        assert count_lineage(new_item) == 6
        checkpoint_calls.append(is_final)
        original_checkpoint(new_item, inputs, is_final)

    pipeline._checkpoint = spy_checkpoint

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 4
    # only the final reduction ever checkpoints - the two intermediate folds
    # (a+b, c+d) do not cross any threshold.
    assert checkpoint_calls == [True]
    # everything is released in that one checkpoint: the 6 ancestors plus the
    # final result's own cluster copy.
    assert len(released) == 7
    assert len(set(released)) == 7
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
