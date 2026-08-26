from __future__ import annotations

import os

import pytest

from vine_reduce import defaults, serialization
from vine_reduce.checkpoint_db import CheckpointDB
from vine_reduce.executor import simple_executor
from vine_reduce.pipeline import Pipeline, VineReduceError

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
    checkpoint_distance=None,
    checkpoint_accumulations=False,
    db=None,
    dataset_name="ds",
    is_result=None,
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
            is_result=is_result or defaults.make_default_is_result(total_events),
            result_postprocess=None,
            chunksize=chunksize,
            reduction_size=reduction_size,
            checkpoint_time=checkpoint_time,
            checkpoint_distance=checkpoint_distance,
            checkpoint_accumulations=checkpoint_accumulations,
            results_dir=str(tmp_path / "results"),
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
            pipeline.handle_outcome(outcome)
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
    # every superseded checkpoint file, plus the final result's own
    # originating cluster-side copy, should be cleaned off disk (the final
    # result itself lives in results_dir, a separate copy - see
    # final_value()).
    assert os.listdir(str(tmp_path / "cluster")) == []
    db.close()


def test_checkpoint_distance_threshold_persists_and_supersedes_intermediates(
    fake_distributor, tmp_path
):
    dataset = {"files": {"a.root": 1, "b.root": 1, "c.root": 1}}
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, reduction_size=2, checkpoint_distance=1
    )

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 3
    rows = db.checkpoints_for("proc", "ds")
    # every intermediate checkpoint should have been superseded and deleted,
    # leaving only the final one.
    assert len(rows) == 1
    assert rows[0].is_final is True
    assert os.listdir(str(tmp_path / "cluster")) == []
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

    retrieved: list[int] = []
    original_retrieve = fake_distributor.retrieve

    def spy_retrieve(result_id, dest_path):
        retrieved.append(result_id)
        original_retrieve(result_id, dest_path)

    fake_distributor.retrieve = spy_retrieve

    intermediate_items = []
    original_checkpoint = pipeline._checkpoint

    def spy_checkpoint(new_item, inputs, is_final):
        before = len(released)
        before_retrieved = len(retrieved)
        original_checkpoint(new_item, inputs, is_final)
        newly_released = released[before:]
        if is_final:
            # nothing reduces a final result further, so it's fully detached
            # from the distributor once safely on disk. Its durable copy is
            # vine_reduce's own (results_dir), fetched via retrieve() -
            # unlike a non-final checkpoint, whose durable copy is entirely
            # the distributor's own doing (see the else branch below).
            assert new_item.source_result_id is None
            assert new_item.file == new_item.checkpoint_path
            assert len(retrieved) == before_retrieved + 1
        else:
            # checkpointing must not release the cluster-side copy, or a
            # later reduction folding this item in would need the manager
            # to re-send it - it must still be a live, distributor-native
            # handle. A non-final checkpoint is durable because the
            # distributor made it so at submit time (is_checkpoint=True),
            # not because vine_reduce copied it out via retrieve() - only a
            # final result does that.
            assert new_item.source_result_id is not None
            assert new_item.source_result_id not in newly_released
            assert len(retrieved) == before_retrieved  # no retrieve() copy for a non-final
            assert new_item.checkpoint_path is not None
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

    released_paths: list[str] = []
    original_release_path = fake_distributor.release_path

    def spy_release_path(path):
        released_paths.append(path)
        original_release_path(path)

    fake_distributor.release_path = spy_release_path

    run_to_completion(pipeline, fake_distributor)

    # 100 (seeded, standing in for a.root) + 5 (b.root, actually processed)
    assert final_value(pipeline) == 105
    # The seeded item has no source_result_id for release_result to release
    # it by (see PoolItem docstring) - once the final reduction folds it in
    # and supersedes it, release_path is how its distributor-side handle (if
    # any) gets dropped instead.
    assert str(seeded_file) in released_paths
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


def test_unsatisfiable_is_result_raises_instead_of_looping_forever(fake_distributor, tmp_path):
    """A custom is_result that the dataset's total can never reach must raise,
    not silently keep re-submitting a no-op fold of the one remaining item
    forever - see maybe_drain_final_group/_handle_reduce_outcome."""
    dataset = {"files": {"a.root": 3, "b.root": 3}}
    never_satisfied = lambda num_events, total_time, total_memory: False  # noqa: E731
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, reduction_size=10, is_result=never_satisfied
    )

    with pytest.raises(VineReduceError, match="is_result"):
        run_to_completion(pipeline, fake_distributor)
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
