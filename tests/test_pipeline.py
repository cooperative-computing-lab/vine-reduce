from __future__ import annotations

import os

import pytest

from vine_reduce import defaults, serialization
from vine_reduce.checkpoint_store import CheckpointRecord, CheckpointStore
from vine_reduce.executor import SimpleExecutor
from vine_reduce.failure_log import FailureLog
from vine_reduce.pipeline import (
    Pipeline,
    PoolItem,
    VineReduceError,
    _ChunkTask,
    _ReduceTask,
    plan_restart,
)
from vine_reduce.types import Chunk, ResourceExhaustion, ResultHandle

from helpers import (
    count_events,
    exhausting_processor,
    failing_processor,
    make_flaky_n_times,
    make_flaky_reducer_n_times,
    sum_reducer,
)


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
    attempts=3,
    failure_proportion=0.0,
    failure_log=None,
):
    db = db or CheckpointStore(str(tmp_path / "db.sqlite"))
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
            executor=SimpleExecutor(),
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
            attempts=attempts,
            failure_proportion=failure_proportion,
            failure_log=failure_log,
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
    # A final result's durable copy is its checkpoint's path (in results_dir);
    # its distributor handle is gone by then (see Pipeline._checkpoint).
    assert len(pipeline.final_results) == 1
    return serialization.load(pipeline.final_results[0].checkpoint.path)


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
    manager re-sending it: its distributor handle must not be released at
    checkpoint time, only later once a further reduction actually folds it in
    and supersedes it - the same point any other pooled item's handle is
    released (invariant #2)."""
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
            assert new_item.handle is None
            assert new_item.checkpoint is not None
            assert os.path.exists(new_item.checkpoint.path)
            assert len(retrieved) == before_retrieved + 1
        else:
            # checkpointing must not release the cluster-side copy, or a
            # later reduction folding this item in would need the manager
            # to re-send it - it must still be a live, distributor-native
            # handle. A non-final checkpoint is durable because the
            # distributor made it so at submit time (is_checkpoint=True),
            # not because vine_reduce copied it out via retrieve() - only a
            # final result does that.
            assert new_item.handle is not None
            assert new_item.handle.result_id not in newly_released
            assert len(retrieved) == before_retrieved  # no retrieve() copy for a non-final
            assert new_item.checkpoint is not None
            assert os.path.exists(new_item.checkpoint.path)
            # the handle is cleared when this item is later released, so
            # remember its result_id now.
            intermediate_items.append((new_item, new_item.handle.result_id))

    pipeline._checkpoint = spy_checkpoint

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 4
    assert len(intermediate_items) == 2  # a+b, c+d
    # each intermediate checkpoint's cluster copy is only released once a
    # later reduction (here, the final one) actually folds it in.
    for item, result_id in intermediate_items:
        assert result_id in released
        assert item.handle is None
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


def test_restart_seeded_checkpoint_is_adopted_used_and_released_when_superseded(
    fake_distributor, tmp_path
):
    """Seeded-checkpoint lifecycle: a partial checkpoint row from a previous
    run is adopted (adopt_checkpoint) at Pipeline construction, giving it a
    real distributor handle indistinguishable from a this-run result's; once
    a superseding checkpoint lands, that handle is released via
    release_result - which, per its contract, also removes the seeded file
    from disk (the distributor is the single owner of non-final checkpoint
    files)."""
    dataset = {"files": {"a.root": 5, "b.root": 5}}
    db = CheckpointStore(str(tmp_path / "db.sqlite"))

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    seeded_file = checkpoint_dir / "seeded.pkl.zst"
    serialization.dump(100, str(seeded_file))  # stands in for a's "already processed" result
    db.record(
        processor="proc",
        dataset="ds",
        covers_files=["a.root"],
        num_events=5,
        wall_time_s=1.0,
        memory_mb=1.0,
        is_final=False,
        path=str(seeded_file),
    )

    adopted_paths: list[str] = []
    original_adopt = fake_distributor.adopt_checkpoint

    def spy_adopt(result_id, path):
        adopted_paths.append(path)
        return original_adopt(result_id, path)

    fake_distributor.adopt_checkpoint = spy_adopt

    pipeline, _ = make_pipeline(fake_distributor, tmp_path, dataset, reduction_size=10, db=db)

    assert adopted_paths == [str(seeded_file)]
    assert pipeline._skip_files == {"a.root"}
    assert len(pipeline.pool) == 1
    seeded_item = pipeline.pool[0]
    assert seeded_item.handle is not None
    assert seeded_item.checkpoint.path == str(seeded_file)
    adopted_result_id = seeded_item.handle.result_id

    released: list[int] = []
    original_release = fake_distributor.release_result

    def spy_release(result_id):
        released.append(result_id)
        original_release(result_id)

    fake_distributor.release_result = spy_release

    run_to_completion(pipeline, fake_distributor)

    # 100 (seeded, standing in for a.root) + 5 (b.root, actually processed)
    assert final_value(pipeline) == 105
    # the final reduction folded the seeded item in and superseded its row,
    # so its adopted handle was released - exactly once - and release_result
    # removed the seeded file from disk.
    assert released.count(adopted_result_id) == 1
    assert not os.path.exists(str(seeded_file))
    db.close()


def test_restart_with_final_checkpoint_for_all_files_skips_pipeline_entirely(
    fake_distributor, tmp_path
):
    dataset = {"files": {"a.root": 5, "b.root": 5}}
    db = CheckpointStore(str(tmp_path / "db.sqlite"))
    db.record(
        processor="proc",
        dataset="ds",
        covers_files=["a.root", "b.root"],
        num_events=10,
        wall_time_s=1.0,
        memory_mb=1.0,
        is_final=True,
        path="/tmp/final.pkl",
    )

    pipeline, _ = make_pipeline(fake_distributor, tmp_path, dataset, db=db)

    assert pipeline.finished is True
    assert pipeline.pool == []
    assert pipeline.in_flight_count() == 0
    assert len(pipeline.final_results) == 1
    db.close()


def test_pool_admits_a_file_only_once_every_chunk_of_it_succeeded(fake_distributor, tmp_path):
    """Invariant #1: only completely processed files are accumulated into the
    pool. With chunksize=2, a 4-event file processes as two chunks; the first
    Success alone must stage its result, not pool it - both chunk results
    enter the pool together only once the whole file is covered."""
    dataset = {"files": {"a.root": 4}}
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, chunksize=2, reduction_size=10
    )

    assert pipeline.feed(100) == 2  # both chunks of a.root

    pipeline.handle_outcome(fake_distributor.wait())
    assert pipeline.pool == []  # a.root only half processed - nothing pooled

    pipeline.handle_outcome(fake_distributor.wait())
    assert len(pipeline.pool) == 2  # fully processed - both results, together
    assert all(item.files == frozenset({"a.root"}) for item in pipeline.pool)
    db.close()


def test_release_covered_frees_deep_lineage_without_recursion(fake_distributor, tmp_path):
    """Invariant #3's walk must survive an arbitrarily long uncheckpointed
    chain: a lineage thousands of levels deep is released exactly once per
    item, with no RecursionError (pins the iterative, explicit-stack walk)."""
    dataset = {"files": {"a.root": 1}}
    pipeline, db = make_pipeline(fake_distributor, tmp_path, dataset)

    released: list[int] = []
    fake_distributor.release_result = released.append

    depth = 5000  # far beyond Python's default recursion limit
    item = None
    for result_id in range(1, depth + 1):
        item = PoolItem(
            handle=ResultHandle(result_id, f"file_{result_id}"),
            num_events=1,
            wall_time_s=0.0,
            memory_mb=0.0,
            files=frozenset({"a.root"}),
            since_checkpoint_time=0.0,
            since_checkpoint_distance=0,
            inputs=[item] if item is not None else [],
        )

    pipeline._release_covered(item)

    assert sorted(released) == list(range(1, depth + 1))  # each exactly once
    assert item.handle is None
    assert item.inputs == []
    db.close()


def _restart_row(row_id, covers, *, is_final, path="/tmp/x.pkl"):
    return CheckpointRecord(
        id=row_id,
        processor="proc",
        dataset="ds",
        covers_files=frozenset(covers),
        num_events=len(covers),
        wall_time_s=1.0,
        memory_mb=1.0,
        is_final=is_final,
        path=path,
    )


def test_plan_restart_all_final_rows_means_finished_with_no_pool():
    rows = [
        _restart_row(1, ["a.root"], is_final=True),
        _restart_row(2, ["b.root"], is_final=True),
    ]
    plan = plan_restart(rows, {"a.root", "b.root"})
    assert plan.finished is True
    assert plan.final_rows == rows
    assert plan.pool_rows == []
    assert plan.skip_files == {"a.root", "b.root"}


def test_plan_restart_mixed_rows_skip_the_union_of_what_they_cover():
    final = _restart_row(1, ["a.root"], is_final=True)
    partial = _restart_row(2, ["b.root", "c.root"], is_final=False)
    plan = plan_restart([final, partial], {"a.root", "b.root", "c.root", "d.root"})
    assert plan.finished is False
    assert plan.final_rows == [final]
    assert plan.pool_rows == [partial]
    assert plan.skip_files == {"a.root", "b.root", "c.root"}  # d.root still to do


def test_plan_restart_partials_are_moot_when_finals_cover_everything():
    final = _restart_row(1, ["a.root", "b.root"], is_final=True)
    partial = _restart_row(2, ["a.root"], is_final=False)
    plan = plan_restart([final, partial], {"a.root", "b.root"})
    assert plan.finished is True
    assert plan.final_rows == [final]
    assert plan.pool_rows == []
    assert plan.skip_files == {"a.root", "b.root"}


def test_two_run_restart_never_resubmits_checkpoint_covered_files(fake_distributor, tmp_path):
    dataset = {"files": {"a.root": 1, "b.root": 1, "c.root": 1, "d.root": 1}}
    db = CheckpointStore(str(tmp_path / "db.sqlite"))

    # Run 1: checkpoint every accumulation, and "crash" (stop driving) right
    # after the first non-final checkpoint lands.
    pipeline, _ = make_pipeline(
        fake_distributor,
        tmp_path,
        dataset,
        reduction_size=2,
        checkpoint_accumulations=True,
        db=db,
    )
    for _ in range(1000):
        if any(not row.is_final for row in db.checkpoints_for("proc", "ds")):
            break
        pipeline.submit_ready_reductions()
        pipeline.maybe_drain_final_group()
        pipeline.feed(100)
        outcome = fake_distributor.wait()
        if outcome is not None:
            pipeline.handle_outcome(outcome)
    else:
        raise AssertionError("no non-final checkpoint ever landed")
    covered = {url for row in db.checkpoints_for("proc", "ds") for url in row.covers_files}
    assert covered
    assert not pipeline.finished

    # Run 2: fresh Pipeline + fresh distributor, same store and cluster dir.
    distributor2 = type(fake_distributor)(str(tmp_path / "cluster"))
    submitted_chunk_urls: set[str] = set()
    original_submit = distributor2.submit

    def spy_submit(result_id, priority, category, kind, func, *args, **kwargs):
        if kind == "processor":
            submitted_chunk_urls.update(a.url for a in args if isinstance(a, Chunk))
        return original_submit(result_id, priority, category, kind, func, *args, **kwargs)

    distributor2.submit = spy_submit

    pipeline2, _ = make_pipeline(
        distributor2,
        tmp_path,
        dataset,
        reduction_size=2,
        checkpoint_accumulations=True,
        db=db,
    )
    run_to_completion(pipeline2, distributor2)

    # checkpoint-covered files were never re-submitted as chunks, and the
    # final value still accounts for every file exactly once.
    assert submitted_chunk_urls == set(dataset["files"]) - covered
    assert final_value(pipeline2) == 4
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


def test_chunk_runtime_failure_retries_within_attempts_then_succeeds(fake_distributor, tmp_path):
    dataset = {"files": {"a.root": 5}}
    processor = make_flaky_n_times(2)  # fails twice, then succeeds
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, processor=processor, reduction_size=10, attempts=3
    )

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 5
    assert processor.calls["count"] == 3  # 2 failures + 1 success, all counted
    db.close()


def test_chunk_runtime_failure_exhausts_attempts_and_raises(fake_distributor, tmp_path):
    dataset = {"files": {"a.root": 5}}
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, processor=failing_processor, attempts=2
    )

    with pytest.raises(VineReduceError, match=r"after 2 attempts \(attempts=2\)"):
        run_to_completion(pipeline, fake_distributor)
    db.close()


def test_chunk_resource_exhaustion_at_minimum_size_raises_immediately(fake_distributor, tmp_path):
    """Once a chunk is down to a single event, ResourceExhaustion has no
    smaller size left to retry at - it must raise right away, regardless of
    how much of the attempts budget remains."""
    dataset = {"files": {"a.root": 1}}
    pipeline, db = make_pipeline(
        fake_distributor,
        tmp_path,
        dataset,
        processor=exhausting_processor,
        chunksize=1,
        attempts=5,
    )

    with pytest.raises(VineReduceError, match="minimum chunk size"):
        run_to_completion(pipeline, fake_distributor)
    db.close()


def test_chunk_attempts_budget_resets_after_a_productive_split(fake_distributor, tmp_path):
    """A ResourceExhaustion-driven halving is a fresh start for the smaller
    chunks it produces, not a strike against the attempts already used."""
    dataset = {"files": {"a.root": 4}}
    pipeline, db = make_pipeline(fake_distributor, tmp_path, dataset, chunksize=4, attempts=2)

    chunk = Chunk("a.root", 0, 4)
    pipeline._in_flight["r1"] = _ChunkTask(chunk=chunk, attempts_used=1)  # 1 of 2 already used
    pipeline._handle_chunk_outcome(
        pipeline._in_flight.pop("r1"),
        ResourceExhaustion(result_id="r1", resources={}, std_output=None),
    )
    assert pipeline.chunksize == 2
    # not yet split - attempts_used still carried as-is until it actually splits
    assert pipeline._retry_chunks == [(chunk, 1)]

    next_chunk = pipeline._next_chunk()
    assert next_chunk == (Chunk("a.root", 0, 2), 0)  # fresh budget for both halves
    assert pipeline._retry_chunks == [(Chunk("a.root", 2, 4), 0)]
    db.close()


def test_reduction_runtime_failure_retries_within_attempts_then_succeeds(
    fake_distributor, tmp_path
):
    dataset = {"files": {"a.root": 1, "b.root": 1}}
    reducer = make_flaky_reducer_n_times(2)  # fails twice, then folds
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, reducer=reducer, reduction_size=2, attempts=3
    )

    run_to_completion(pipeline, fake_distributor)

    assert final_value(pipeline) == 2
    assert reducer.calls["count"] == 3
    db.close()


def test_reduction_runtime_failure_exhausts_attempts_and_raises(fake_distributor, tmp_path):
    def always_fails(a, b):
        raise ValueError("boom")

    dataset = {"files": {"a.root": 1, "b.root": 1}}
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, reducer=always_fails, reduction_size=2, attempts=2
    )

    with pytest.raises(VineReduceError, match=r"after 2 attempts \(attempts=2\)"):
        run_to_completion(pipeline, fake_distributor)
    db.close()


def test_reduction_resource_exhaustion_at_minimum_size_raises_immediately(
    fake_distributor, tmp_path
):
    """reduction_size=2 is already the floor - a ResourceExhaustion there
    must raise right away, regardless of how much of the attempts budget
    remains."""

    def always_exhausts(a, b):
        raise MemoryError("simulated resource exhaustion")

    dataset = {"files": {"a.root": 1, "b.root": 1}}
    pipeline, db = make_pipeline(
        fake_distributor, tmp_path, dataset, reducer=always_exhausts, reduction_size=2, attempts=5
    )

    with pytest.raises(VineReduceError, match="minimum reduction_size"):
        run_to_completion(pipeline, fake_distributor)
    db.close()


def test_reduction_attempts_budget_resets_after_resource_exhaustion(fake_distributor, tmp_path):
    """A ResourceExhaustion-driven halving of reduction_size is a fresh
    start for the disbanded items, not a strike against attempts already
    used - see PoolItem.attempts."""
    dataset = {"files": {f"{c}.root": 1 for c in "abcd"}}
    pipeline, db = make_pipeline(fake_distributor, tmp_path, dataset, reduction_size=4, attempts=2)

    items = [
        PoolItem(
            handle=ResultHandle(f"r{i}", f"f{i}"),
            num_events=1,
            wall_time_s=0.0,
            memory_mb=0.0,
            files=frozenset({f"{c}.root"}),
            since_checkpoint_time=0.0,
            since_checkpoint_distance=0,
            attempts=1,  # 1 of 2 already used
        )
        for i, c in enumerate("abcd")
    ]
    pipeline._in_flight["r"] = _ReduceTask(
        group=items,
        is_final=False,
        is_checkpoint=False,
        num_events=4,
        total_time=0.0,
        total_memory=0.0,
    )
    pipeline._handle_reduce_outcome(
        pipeline._in_flight.pop("r"),
        ResourceExhaustion(result_id="r", resources={}, std_output=None),
    )

    assert pipeline.reduction_size == 2
    assert all(item.attempts == 0 for item in items)  # reset, not incremented
    assert pipeline.pool[:4] == items
    db.close()


def test_processor_permanent_failure_is_skipped_within_failure_proportion(
    fake_distributor, tmp_path
):
    """A permanently-failed file is removed from the pool for good, but the
    rest of the dataset keeps going - and the pipeline still reaches a
    final result over what's left - as long as failure_proportion tolerates
    it. is_result is custom here since the default ("every event") could
    never be satisfied once a file is permanently missing - see PLAN.md."""
    dataset = {"files": {"a.root": 5, "b.root": 5, "c.root": 5}}

    def processor(chunk):
        if chunk.url == "b.root":
            raise ValueError("boom")
        return chunk.stop - chunk.start

    pipeline, db = make_pipeline(
        fake_distributor,
        tmp_path,
        dataset,
        processor=processor,
        reduction_size=10,
        attempts=1,
        failure_proportion=0.5,
        is_result=lambda num_events, total_time, total_memory: num_events >= 10,
    )

    run_to_completion(pipeline, fake_distributor)

    assert pipeline.failed_files == frozenset({"b.root"})
    assert final_value(pipeline) == 10
    db.close()


def test_processor_permanent_failures_abort_once_proportion_exceeded(fake_distributor, tmp_path):
    """failure_proportion tolerates the first permanent failure but not the
    second, for the same dataset."""
    dataset = {"files": {"a.root": 1, "b.root": 1, "c.root": 1, "d.root": 1}}

    def processor(chunk):
        if chunk.url in ("b.root", "d.root"):
            raise ValueError("boom")
        return chunk.stop - chunk.start

    pipeline, db = make_pipeline(
        fake_distributor,
        tmp_path,
        dataset,
        processor=processor,
        reduction_size=10,
        attempts=1,
        failure_proportion=0.015,  # 1/100 tolerated, 2/100 is not
        is_result=lambda num_events, total_time, total_memory: num_events >= 2,
    )

    with pytest.raises(VineReduceError, match=r"after 1 attempt \(attempts=1\)"):
        run_to_completion(pipeline, fake_distributor)
    assert pipeline.failed_files == frozenset({"b.root", "d.root"})
    db.close()


def test_reducer_permanent_failure_logs_group_files_and_ignores_failure_proportion(
    fake_distributor, tmp_path
):
    """Unlike a processor failure, a reducer permanent failure always aborts
    the whole run - failure_proportion is never consulted for it - but every
    file folded into the failing group is still logged first."""

    def always_fails(a, b):
        raise ValueError("boom")

    dataset = {"files": {"a.root": 1, "b.root": 1}}
    log_path = str(tmp_path / "failed_files.log")
    failure_log = FailureLog(log_path)
    pipeline, db = make_pipeline(
        fake_distributor,
        tmp_path,
        dataset,
        reducer=always_fails,
        reduction_size=2,
        attempts=1,
        failure_proportion=0.99,  # would tolerate nearly anything for a processor failure
        failure_log=failure_log,
    )

    with pytest.raises(VineReduceError, match=r"after 1 attempt \(attempts=1\)"):
        run_to_completion(pipeline, fake_distributor)

    contents = open(log_path).read()
    assert "a.root" in contents
    assert "b.root" in contents
    assert "reducer" in contents
    db.close()
