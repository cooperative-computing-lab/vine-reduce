"""Quick-start tutorial: vine_reduce over real files via the TaskVine executor.

vine_reduce runs a MapReduce-style computation over one or more named
"datasets" (see build_datasets below for their shape). For each (processor,
dataset) pair, it:

  1. Splits every file in the dataset into contiguous chunks of entries and
     runs the processor (the "map" step) on each chunk, remotely, as a
     TaskVine task. numbers_chunk_to_args below is what turns a chunk into
     the actual arguments a processor receives.
  2. Collects the processor outputs into a pool, then repeatedly folds
     batches of pooled results together with a "reducer" (the "reduce"
     step) - also remotely - producing fewer, larger partial results each
     round, until one final result covers the whole dataset. This example
     doesn't pass a custom reducer, so vine_reduce uses its default: `a +=
     b`, i.e. plain addition, which is exactly what you want when a
     processor's output (here, an int) already knows how to combine with
     another of its own kind.
  3. Writes that final result to results_dir, one file per (dataset,
     processor) pair (see load_result below), and along the way
     checkpoints intermediate reduce outputs to the TaskVineDistributor's
     own checkpoint_dir so a crashed run can resume without recomputing
     chunks it already finished.

None of this requires knowing TaskVine's own vocabulary (managers, workers,
tasks) beyond what it takes to start one: main() below constructs a
TaskVineDistributor (vine_reduce's TaskVine-backed executor, wrapping a
TaskVine manager) and a single local worker via vine.Factory, so `python
quick_start.py` runs standalone - no cluster, no separate worker process to
launch by hand.

Concretely, in this example:

- write_test_data.py generates two datasets of three binary files each,
  under examples/quick_start/data/, freshly on every run. Each file holds
  50-100 random positive 4-byte integers packed with struct. Datasets are
  built with each file's *absolute* path (computed at run time) as its
  "files" key, since the TaskVine workers below run as separate processes
  that only share a filesystem with this one - not its working directory
  or relative paths.
- Three processors act as the map step, run over both datasets: one sums
  only even numbers, one sums only odd numbers, one sums everything.
- At the end, the six result files (2 datasets x 3 processors) are loaded
  back into memory and checked for internal consistency: odd + even must
  equal all.
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
    """chunk_to_args runs remotely, once per chunk, right before the processor
    is called with whatever it returns. vine_reduce hands it a
    vine_reduce.Chunk - a plain (url, start, stop) triple identifying a
    range of entries in one file, with no knowledge of that file's format -
    so this is the one place that knows how to turn a chunk into the actual
    numbers a processor works with: seek to chunk.start within chunk.url
    (the packed integers written by write_test_data.py) and unpack
    [chunk.start, chunk.stop). dataset_metadata (the dataset's "metadata"
    dict, see build_datasets) and distributor_metadata are unused here, but
    are how a real workflow would pass e.g. a branch list or per-worker
    config into this step."""
    count = chunk.stop - chunk.start
    with open(chunk.url, "rb") as f:
        f.seek(chunk.start * INT_SIZE)
        data = f.read(count * INT_SIZE)
    return list(struct.unpack(f"<{count}I", data))


# The three "processors" below are the map step: each one is applied,
# independently and remotely, to the list of numbers a single chunk decodes
# to. Their outputs (plain ints) are what the default reducer (`a += b`)
# then combines pairwise, chunk by chunk, into each dataset's final sum.


def sum_even_processor(numbers):
    return sum(n for n in numbers if n % 2 == 0)


def sum_odd_processor(numbers):
    return sum(n for n in numbers if n % 2 == 1)


def sum_all_processor(numbers):
    return sum(numbers)


def load_result(results_dir, dataset_name, processor_name):
    """Final results land under results_dir/<dataset_name>/<processor_name>/
    as a single compressed, pickled file (name includes a random uuid, hence
    the glob). serialization.load reverses what the reducer wrote."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


def build_datasets(data_dir):
    """Builds the `input` dict VineReduce expects: one entry per dataset,
    each with a "metadata" dict (passed verbatim to chunk_to_args - unused
    here) and a "files" dict mapping each file's path to its entry count.
    vine_reduce uses those counts to decide where chunk boundaries fall
    without opening any file itself."""
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

    # TaskVineDistributor is vine_reduce's executor for running chunk/reduce
    # tasks on a TaskVine cluster: it owns a vine.Manager (port=0 picks a
    # free port) and declares, once per "category" (map tasks and reduce
    # tasks get separate categories), how many cores each task in that
    # category needs. This is TaskVine's own resource-provisioning model -
    # vine_reduce just wires it up here rather than replacing it.
    distributor = TaskVineDistributor(
        port=0,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1},
        checkpoint_dir=checkpoint_dir,
    )
    # A manager alone runs nothing - it needs workers to connect and execute
    # tasks. vine.Factory manages the lifecycle of local worker processes for
    # us (min/max_workers pins it to exactly one here), so this example needs
    # no separate `vine_worker` process started by hand. In production, workers
    # normally run as their own long-lived processes across a cluster, entirely
    # outside of vine_reduce's or this script's control.
    workers = vine.Factory(manager_host_port=f"localhost:{distributor.port}")
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1

    with workers, distributor:
        # VineReduce ties everything together: which datasets to process
        # (input), what map function to run per processor (processors), how
        # to turn a chunk into that function's arguments (chunk_to_args), how
        # large a chunk should be in entries (chunksize), where to write
        # final results (results_dir - intermediate checkpoints are the
        # distributor's own concern, see checkpoint_dir above), and which
        # executor backend runs the actual tasks (distributor). reducer and
        # is_result are left at their defaults: plain `a += b` addition, and
        # "a group is final once it covers every entry of the dataset" (see
        # defaults.py for both).
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
            distributor=distributor,
        )
        # Runs the full map/reduce loop to completion: submits chunk and
        # reduce tasks to the distributor, waits for outcomes, and folds
        # results together until every (processor, dataset) pair has a
        # final result on disk under results_dir.
        vr.compute()

    # Sanity-check the results: for each dataset, the file written by the
    # sum_odd and sum_even processors should add up to the one written by
    # sum_all, confirming the chunk/reduce pipeline recombined every entry
    # of every file exactly once.
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
