from __future__ import annotations

import os

import pytest

from vine_reduce import VineReduce, serialization
from vine_reduce.checkpoint_db import CheckpointDB, checksum_dataset
from vine_reduce.engine import _resolve_sized_config
from vine_reduce.local_distributor import LocalDistributor

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

    db = CheckpointDB(str(db_path))
    db.dataset_changed("numbers", checksum_dataset(datasets["numbers"]))
    db.add_checkpoint("count", "numbers", ["a.root", "b.root"], 10, 1.0, 1.0, True, str(final_file))
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
