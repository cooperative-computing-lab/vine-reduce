"""Quick-start example: vine_reduce over real files via the TaskVine executor.

write_test_data.py generates two datasets of three binary files each, under
examples/quick_start/data/, freshly on every run. Each file holds 50-100
random positive 4-byte integers packed with struct. Datasets are built with
each file's *absolute* path (computed at run time) as its "files" key, since
the TaskVine workers below run as separate processes that only share a
filesystem with this one - not its working directory or relative paths.

Three processors run over both datasets: one sums only even numbers, one
sums only odd numbers, one sums everything. quick_start.py starts its own
TaskVine manager and a single local worker (via vine.Factory) for the
duration of the run, so `python quick_start.py` works standalone. At the
end, the six result files (2 datasets x 3 processors) are loaded back into
memory and checked for internal consistency: odd + even must equal all.
"""

from __future__ import annotations

import glob
import os
import random
import shutil
import struct

import ndcctools.taskvine as vine

import write_test_data
from vine_reduce import VineReduce, serialization
from vine_reduce.taskvine_distributor import TaskVineDistributor

CHUNKSIZE = 10
DATASET_NAMES = ("dataset_a", "dataset_b")
FILES_PER_DATASET = 3

# Duplicates write_test_data.INT_SIZE rather than referencing that module:
# numbers_chunk_to_args below runs on the TaskVine worker, and cloudpickle
# would otherwise capture a by-reference import of write_test_data - which
# isn't necessarily on the worker's PYTHONPATH. A plain int global pickles
# by value with the rest of the function, so no import is needed remotely.
INT_SIZE = struct.calcsize("<I")


def numbers_chunk_to_args(chunk, dataset_metadata, distributor_metadata=None):
    """Reads the [chunk.start, chunk.stop) integers packed into chunk.url by
    write_test_data.py."""
    count = chunk.stop - chunk.start
    with open(chunk.url, "rb") as f:
        f.seek(chunk.start * INT_SIZE)
        data = f.read(count * INT_SIZE)
    return list(struct.unpack(f"<{count}I", data))


def sum_even_processor(numbers):
    return sum(n for n in numbers if n % 2 == 0)


def sum_odd_processor(numbers):
    return sum(n for n in numbers if n % 2 == 1)


def sum_all_processor(numbers):
    return sum(numbers)


def load_result(results_dir, dataset_name, processor_name):
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


def build_datasets(data_dir):
    rng = random.Random()
    return {
        dataset_name: {
            "metadata": {},
            "files": write_test_data.generate_dataset_files(
                data_dir, dataset_name, FILES_PER_DATASET, rng
            ),
        }
        for dataset_name in DATASET_NAMES
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    results_dir = os.path.join(here, "results")
    checkpoint_dir = os.path.join(here, "checkpoints")

    # Fresh data every run, so stale results/checkpoints from a previous run
    # (over different random files) must not linger either.
    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    datasets = build_datasets(data_dir)

    distributor = TaskVineDistributor(
        port=0, resources_processor={"cores": 1}, resources_reducer={"cores": 1}
    )
    workers = vine.Factory(manager_host_port=f"localhost:{distributor.port}")
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1

    with workers:
        vr = VineReduce(
            processors={
                "sum_even": sum_even_processor,
                "sum_odd": sum_odd_processor,
                "sum_all": sum_all_processor,
            },
            input=datasets,
            chunk_to_args=numbers_chunk_to_args,
            chunksize=CHUNKSIZE,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            distributor=distributor,
        )
        vr.compute()
    distributor.shutdown()

    for dataset_name in datasets:
        results = {
            processor_name: load_result(results_dir, dataset_name, processor_name)
            for processor_name in vr.processors
        }
        print(f"{dataset_name}: {results}")
        assert (
            results["sum_odd"] + results["sum_even"] == results["sum_all"]
        ), f"{dataset_name}: odd + even != all"

    print("OK: odd + even == all for every dataset")


if __name__ == "__main__":
    main()
