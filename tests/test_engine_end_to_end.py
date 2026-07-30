from __future__ import annotations

import os

import pytest

from vine_reduce import VineReduce, serialization
from vine_reduce.checkpoint_db import CheckpointDB, checksum_dataset
from vine_reduce.local_distributor import LocalDistributor

from helpers import count_events, sum_reducer


@pytest.fixture
def distributor(tmp_path):
    dist = LocalDistributor(max_workers=2, work_dir=str(tmp_path / "cluster"))
    yield dist
    dist.shutdown()


def _read_only_result(results_dir, dataset_name):
    dataset_dir = os.path.join(results_dir, dataset_name)
    files = os.listdir(dataset_dir)
    assert len(files) == 1
    return serialization.load(os.path.join(dataset_dir, files[0]))


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
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    vr.compute()

    assert _read_only_result(vr.results_dir, "numbers") == 10
    assert _read_only_result(vr.results_dir, "more_numbers") == 4


def test_restart_skips_already_finalized_dataset(tmp_path, dataset_input, distributor):
    datasets = {"numbers": {"metadata": {}, "files": {"a.root": 7, "b.root": 3}}}
    input_path = dataset_input(datasets)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    results_dir = tmp_path / "results" / "numbers"
    results_dir.mkdir(parents=True)
    final_file = results_dir / "already_done.pkl.zst"
    serialization.dump(999, str(final_file))

    db = CheckpointDB(str(checkpoint_dir / "vine_reduce.db"))
    db.dataset_changed("numbers", checksum_dataset(datasets["numbers"]))
    db.add_checkpoint("count", "numbers", ["a.root", "b.root"], 10, 1.0, 1.0, True, str(final_file))
    db.close()

    def explode(chunk):
        raise AssertionError("processor should not run: dataset already finalized")

    vr = VineReduce(
        processors={"count": explode},
        input=input_path,
        reducer=sum_reducer,
        checkpoint_dir=str(checkpoint_dir),
        results_dir=str(tmp_path / "results"),
        distributor=distributor,
    )
    vr.compute()

    # unchanged: still just the pre-seeded final result, processor never ran
    assert os.listdir(str(results_dir)) == ["already_done.pkl.zst"]
