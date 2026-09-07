from __future__ import annotations

import json
import os

import pytest

from vine_reduce import VineReduce, VineReduceError, defaults, serialization
from vine_reduce.checkpoint_store import CheckpointStore, checksum_dataset
from vine_reduce.engine import (
    _resolve_minimum_chunksize,
    _resolve_minimum_reduction_size,
    _resolve_reduction_size,
    _resolve_sized_config,
)
from vine_reduce.failure_log import FailureLog
from vine_reduce.local_distributor import LocalDistributor
from vine_reduce.progress import NullProgressReporter
from vine_reduce.size_log import SizeLog

from helpers import count_events, read_env_var, sum_reducer


def double_count_events(chunk):
    return 2 * (chunk.stop - chunk.start)


@pytest.fixture
def distributor(tmp_path):
    dist = LocalDistributor(
        max_workers=2,
        work_dir=str(tmp_path / "cluster"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    yield dist
    dist.shutdown()


def _read_only_result(results_dir, dataset_name, processor_name="count"):
    dataset_dir = os.path.join(results_dir, dataset_name, processor_name)
    files = os.listdir(dataset_dir)
    assert len(files) == 1
    return serialization.load(os.path.join(dataset_dir, files[0]))


def test_resolve_sized_config_passes_through_plain_int():
    assert _resolve_sized_config(5, "proc", "ds") == 5


def test_resolve_sized_config_passes_through_none():
    assert _resolve_sized_config(None, "proc", "ds") is None


def test_resolve_sized_config_dataset_beats_processor_beats_default():
    config = {"default": 1, "processors": {"proc": 2}, "datasets": {"ds": 3}}
    assert _resolve_sized_config(config, "proc", "ds") == 3
    assert _resolve_sized_config(config, "proc", "other_ds") == 2
    assert _resolve_sized_config(config, "other_proc", "other_ds") == 1


def test_resolve_sized_config_missing_keys_fall_back_to_default():
    assert _resolve_sized_config({"default": 7}, "proc", "ds") == 7
    assert _resolve_sized_config({}, "proc", "ds") is None


def test_resolve_reduction_size_passes_through_valid_int():
    assert _resolve_reduction_size(5, "proc", "ds") == 5
    assert _resolve_reduction_size({"default": 5}, "proc", "ds") == 5


def test_resolve_reduction_size_raises_on_missing_default():
    with pytest.raises(VineReduceError):
        _resolve_reduction_size({}, "proc", "ds")


def test_resolve_reduction_size_raises_on_too_small_value():
    with pytest.raises(VineReduceError):
        _resolve_reduction_size(1, "proc", "ds")
    with pytest.raises(VineReduceError):
        _resolve_reduction_size({"default": 1}, "proc", "ds")


def test_resolve_minimum_reduction_size_defaults_to_two_when_not_given():
    assert _resolve_minimum_reduction_size(None, 10) == 2


def test_resolve_minimum_reduction_size_raises_values_below_two_to_two():
    assert _resolve_minimum_reduction_size(0, 10) == 2
    assert _resolve_minimum_reduction_size(1, 10) == 2


def test_resolve_minimum_reduction_size_caps_at_reduction_size():
    assert _resolve_minimum_reduction_size(20, 10) == 10


def test_resolve_minimum_reduction_size_passes_through_valid_value():
    assert _resolve_minimum_reduction_size(4, 10) == 4


def test_resolve_minimum_chunksize_defaults_to_1000_when_not_given():
    assert _resolve_minimum_chunksize(None) == 1000


def test_resolve_minimum_chunksize_raises_values_below_one_to_one():
    assert _resolve_minimum_chunksize(0) == 1
    assert _resolve_minimum_chunksize(-5) == 1


def test_resolve_minimum_chunksize_passes_through_valid_value():
    assert _resolve_minimum_chunksize(50) == 50


def test_reduction_size_dict_missing_default_raises_clearly(tmp_path, dataset_input, distributor):
    input_path = dataset_input({"numbers": {"metadata": {}, "files": {"a.root": 7}}})

    vr = VineReduce(
        processors={"count": count_events},
        input=input_path,
        reducer=sum_reducer,
        reduction_size={"processors": {"other_proc": 2}},
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    with pytest.raises(VineReduceError):
        vr.compute()


def test_end_to_end_two_processors_two_datasets(tmp_path, dataset_input, distributor):
    input_path = dataset_input(
        {
            "numbers": {"metadata": {}, "files": {"a.root": 7, "b.root": 3}},
            "more_numbers": {"metadata": {}, "files": {"c.root": 4}},
        }
    )

    vr = VineReduce(
        processors={"count": count_events, "double_count": double_count_events},
        input=input_path,
        reducer=sum_reducer,
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    vr.compute()

    # each (processor, dataset) pair gets its own pipeline, and its own
    # results_dir/dataset/processor subdirectory, so results never collide.
    assert _read_only_result(vr.results_dir, "numbers", "count") == 10
    assert _read_only_result(vr.results_dir, "more_numbers", "count") == 4
    assert _read_only_result(vr.results_dir, "numbers", "double_count") == 20
    assert _read_only_result(vr.results_dir, "more_numbers", "double_count") == 8

    # size.jsonl gets one row per (processor, dataset) pair too, at the top
    # of results_dir rather than nested per-pipeline.
    rows = [
        json.loads(line)
        for line in open(os.path.join(vr.results_dir, "size.jsonl")).read().splitlines()
    ]
    pairs = {(row["processor_name"], row["dataset_name"]) for row in rows}
    assert pairs == {
        ("count", "numbers"),
        ("count", "more_numbers"),
        ("double_count", "numbers"),
        ("double_count", "more_numbers"),
    }
    for row in rows:
        assert row["processing"]["cores"] == 1
        assert row["processing"]["memory"] > 0


def test_per_dataset_reduction_size_config_is_respected(tmp_path, dataset_input, distributor):
    input_path = dataset_input(
        {
            "small_groups": {"metadata": {}, "files": {"a.root": 1, "b.root": 1, "c.root": 1}},
            "one_group": {"metadata": {}, "files": {"d.root": 1, "e.root": 1, "f.root": 1}},
        }
    )

    vr = VineReduce(
        processors={"count": count_events},
        input=input_path,
        reducer=sum_reducer,
        reduction_size={"datasets": {"small_groups": 2}, "default": 10},
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    vr.compute()

    assert _read_only_result(vr.results_dir, "small_groups") == 3
    assert _read_only_result(vr.results_dir, "one_group") == 3


def test_end_to_end_two_datasets_two_files_each(tmp_path, dataset_input, distributor):
    input_path = dataset_input(
        {
            "numbers": {"metadata": {}, "files": {"a.root": 7, "b.root": 3}},
            "more_numbers": {"metadata": {}, "files": {"c.root": 4}},
        }
    )

    vr = VineReduce(
        processors={"count": count_events},
        input=input_path,
        reducer=sum_reducer,
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    vr.compute()

    assert _read_only_result(vr.results_dir, "numbers") == 10
    assert _read_only_result(vr.results_dir, "more_numbers") == 4


def test_restart_skips_already_finalized_dataset(tmp_path, dataset_input, distributor):
    datasets = {"numbers": {"metadata": {}, "files": {"a.root": 7, "b.root": 3}}}
    input_path = dataset_input(datasets)

    db_path = tmp_path / "vine_reduce.db"
    results_dir = tmp_path / "results" / "numbers" / "count"
    results_dir.mkdir(parents=True)
    final_file = results_dir / "already_done.pkl.zst"
    serialization.dump(999, str(final_file))

    db = CheckpointStore(str(db_path))
    db.reset_if_dataset_changed("numbers", checksum_dataset(datasets["numbers"]))
    db.record(
        processor="count",
        dataset="numbers",
        covers_files=["a.root", "b.root"],
        num_events=10,
        wall_time_s=1.0,
        memory_mb=1.0,
        is_final=True,
        path=str(final_file),
    )
    db.close()

    def explode(chunk):
        raise AssertionError("processor should not run: dataset already finalized")

    vr = VineReduce(
        processors={"count": explode},
        input=input_path,
        reducer=sum_reducer,
        db_path=str(db_path),
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    vr.compute()

    # unchanged: still just the pre-seeded final result, processor never ran
    assert os.listdir(str(results_dir)) == ["already_done.pkl.zst"]


def test_dataset_change_deletes_the_stale_final_result_file(tmp_path, dataset_input, distributor):
    """reset_if_dataset_changed() drops the DB row for a changed dataset's old final
    result, but that alone leaves the file itself sitting in results_dir
    forever (Correctness #4) - compute() must unlink it too, so a re-run
    after editing a dataset's definition doesn't leave the old final result
    beside the new one."""
    old_datasets = {"numbers": {"metadata": {}, "files": {"a.root": 7, "b.root": 3}}}
    new_datasets = {"numbers": {"metadata": {}, "files": {"a.root": 7, "b.root": 3, "c.root": 5}}}
    input_path = dataset_input(new_datasets)

    db_path = tmp_path / "vine_reduce.db"
    results_dir = tmp_path / "results" / "numbers" / "count"
    results_dir.mkdir(parents=True)
    stale_file = results_dir / "stale.pkl.zst"
    serialization.dump(999, str(stale_file))

    db = CheckpointStore(str(db_path))
    db.reset_if_dataset_changed("numbers", checksum_dataset(old_datasets["numbers"]))
    db.record(
        processor="count",
        dataset="numbers",
        covers_files=["a.root", "b.root"],
        num_events=10,
        wall_time_s=1.0,
        memory_mb=1.0,
        is_final=True,
        path=str(stale_file),
    )
    db.close()

    vr = VineReduce(
        processors={"count": count_events},
        input=input_path,
        reducer=sum_reducer,
        db_path=str(db_path),
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    vr.compute()

    assert not os.path.exists(stale_file)
    assert _read_only_result(vr.results_dir, "numbers") == 15


def test_environment_variables_reach_the_processor(tmp_path, dataset_input, distributor):
    input_path = dataset_input({"numbers": {"metadata": {}, "files": {"a.root": 1}}})

    vr = VineReduce(
        processors={"env": read_env_var},
        input=input_path,
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
        environment_variables={"VINE_REDUCE_TEST_VAR": "xyz"},
    )
    vr.compute()

    assert _read_only_result(vr.results_dir, "numbers", "env") == "xyz"


def test_zero_capacity_before_any_worker_is_available_does_not_hang(tmp_path, dataset_input):
    """Before engine.py slept in this state, a distributor reporting zero
    capacity with nothing in flight yet (e.g. TaskVine before a worker
    connects) made the scheduling loop spin on distributor.capacity() with no
    wait at all. This forces exactly that state a few times in a row and
    checks compute() still reaches completion instead of hanging."""

    class SlowToStartDistributor(LocalDistributor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._zero_capacity_calls_left = 3

        def capacity(self):
            if self._zero_capacity_calls_left > 0:
                self._zero_capacity_calls_left -= 1
                return 0
            return super().capacity()

    input_path = dataset_input({"numbers": {"metadata": {}, "files": {"a.root": 3}}})
    dist = SlowToStartDistributor(
        max_workers=1,
        work_dir=str(tmp_path / "cluster"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    try:
        vr = VineReduce(
            processors={"count": count_events},
            input=input_path,
            reducer=sum_reducer,
            results_dir=str(tmp_path / "results"),
            distributor=dist,
        )
        vr.compute()
    finally:
        dist.shutdown()

    assert _read_only_result(vr.results_dir, "numbers") == 3


def test_extra_files_and_environment_variables_are_passed_to_the_distributor(
    tmp_path, dataset_input
):
    """VineReduce itself is distributor-agnostic - it just forwards
    extra_files/environment_variables to distributor.add_file/set_env_var
    once, before compute() submits anything. This checks that forwarding
    directly, independent of what a given distributor does with them (see
    test_local_distributor.py/test_taskvine_distributor.py for that)."""

    class RecordingDistributor(LocalDistributor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.added_files = []
            self.env_vars = {}

        def add_file(self, local_path):
            self.added_files.append(local_path)
            super().add_file(local_path)

        def set_env_var(self, name, value):
            self.env_vars[name] = value
            super().set_env_var(name, value)

    input_path = dataset_input({"numbers": {"metadata": {}, "files": {"a.root": 1}}})
    shipped = tmp_path / "shipped.txt"
    shipped.write_text("hi")

    dist = RecordingDistributor(
        max_workers=2,
        work_dir=str(tmp_path / "cluster"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    try:
        vr = VineReduce(
            processors={"count": count_events},
            input=input_path,
            results_dir=str(tmp_path / "results"),
            distributor=dist,
            extra_files=[str(shipped)],
            environment_variables={"VINE_REDUCE_TEST_VAR": "xyz"},
        )
        vr.compute()
    finally:
        dist.shutdown()

    assert dist.added_files == [str(shipped)]
    assert dist.env_vars == {"VINE_REDUCE_TEST_VAR": "xyz"}


def test_build_pipelines_priority_orders_by_processor_then_dataset(tmp_path, distributor):
    """_build_pipelines must give each (processor, dataset) pair its own
    priority: earlier processors beat later ones, and - within the same
    processor - earlier datasets beat later ones, while every reduce
    priority still outranks every process priority (PLAN.md's
    "Priorities")."""
    # dict iteration order is insertion order, so this pins down both the
    # processor order (count, then double_count) and the dataset order
    # (numbers, then more_numbers, then last_numbers).
    datasets = {
        "numbers": {"metadata": {}, "files": {"a.root": 7}},
        "more_numbers": {"metadata": {}, "files": {"b.root": 3}},
        "last_numbers": {"metadata": {}, "files": {"c.root": 1}},
    }

    vr = VineReduce(
        processors={"count": count_events, "double_count": double_count_events},
        input="unused",
        reducer=sum_reducer,
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )

    db = CheckpointStore(str(tmp_path / "db.sqlite"))
    try:
        pipelines = vr._build_pipelines(
            datasets,
            distributor,
            db,
            defaults.default_datasets_to_chunks,
            NullProgressReporter(),
            FailureLog(str(tmp_path / "failed_files.log")),
            SizeLog(str(tmp_path / "size.jsonl")),
        )
    finally:
        db.close()

    by_key = {(p.processor_name, p.dataset_name): p for p in pipelines}
    assert set(by_key) == {
        ("count", "numbers"),
        ("count", "more_numbers"),
        ("count", "last_numbers"),
        ("double_count", "numbers"),
        ("double_count", "more_numbers"),
        ("double_count", "last_numbers"),
    }

    # 1. Within the same processor, earlier datasets get strictly better
    # (larger) process_priority than later ones.
    for proc_name in ("count", "double_count"):
        priorities = [
            by_key[(proc_name, ds_name)]._process_priority
            for ds_name in ("numbers", "more_numbers", "last_numbers")
        ]
        assert priorities == sorted(priorities, reverse=True)
        assert len(set(priorities)) == len(priorities)

    # 2. An earlier processor's pipelines all outrank a later processor's,
    # regardless of dataset.
    count_priorities = [
        by_key[("count", ds_name)]._process_priority
        for ds_name in ("numbers", "more_numbers", "last_numbers")
    ]
    double_count_priorities = [
        by_key[("double_count", ds_name)]._process_priority
        for ds_name in ("numbers", "more_numbers", "last_numbers")
    ]
    assert min(count_priorities) > max(double_count_priorities)

    # 3. Every reduce_priority exceeds every process_priority, even with
    # multiple datasets in play.
    all_process_priorities = [p._process_priority for p in pipelines]
    all_reduce_priorities = [p._reduce_priority for p in pipelines]
    assert min(all_reduce_priorities) > max(all_process_priorities)
