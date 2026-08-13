"""HEP skim tutorial: vine_reduce.VineReduceCoffea over synthetic NanoAOD-like
data, via the TaskVine executor.

Adapted from the "cortado" example in
https://github.com/cooperative-computing-lab/dynamic_data_reduction (the
predecessor this project's dynamic map-reduce loop is based on): a coffea
processor skims events down to the ones with at least four leptons, and a
custom reducer concatenates the surviving events from every chunk into one
growing awkward array per dataset, exactly like DynamicDataReduction's own
"accumulator" hook. Dropped relative to the original: ROOT output (this
example writes plain parquet, see load_skim below), on-site condor/xrootd
config (samples here are local synthetic files, not a real CMS dataset), and
periodic checkpoint-triggered writes to disk (VineReduce already checkpoints
intermediate reduce results under checkpoint_dir for restart, so nothing
extra is needed for that).

VineReduceCoffea (src/vine_reduce/coffea.py) is a VineReduce specialization
for coffea: given a dataset in coffea's own preprocessed-file shape (name ->
metadata + files, each file carrying object_path/num_entries - see
build_datasets), it takes care of reading a Chunk as NanoEvents and
materializing a processor's awkward-array output before it's sent back over
the wire. Chunking, checkpointing, and restart are otherwise inherited
unchanged from VineReduce (see examples/quick_start/quick_start.py for a
line-by-line tour of those mechanics with plain Python types instead of
awkward arrays).

Concretely, in this example:

- write_test_data.py generates two datasets ("signal" and "background") of
  three NanoAOD-shaped ROOT files each, under examples/coffea_skim/data/,
  freshly on every run. "signal" files average more leptons per event than
  "background" ones (see DATASET_LEPTON_MEANS), so the skim below should
  keep a noticeably larger fraction of "signal" events.
- skimmer (the map step) keeps only events with >=4 reconstructed leptons
  (electrons + muons combined) - a placeholder for a real analysis'
  selection.
- accumulate_skims (the reduce step) concatenates the surviving events from
  two chunks/groups into one awkward array, replacing VineReduce's default
  `a += b` reducer (which doesn't know how to concatenate awkward arrays).
- At the end, the final skim for each dataset is loaded back into memory,
  written out as a parquet file, and checked for the signal-vs-background
  asymmetry the input data was built to produce.
"""

from __future__ import annotations

import glob
import os
import shutil

import awkward as ak
import ndcctools.taskvine as vine
import numpy as np

import write_test_data
from vine_reduce import serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor

CHUNKSIZE = 150
FILES_PER_DATASET = 3
DATASET_LEPTON_MEANS = {"signal": 3.0, "background": 1.0}


def skimmer(events):
    """Runs remotely, once per Chunk of NanoEvents (VineReduceCoffea's
    chunk_to_args + executor take care of turning a Chunk into `events`
    and materializing this function's return value). Placeholder ">=4
    leptons" selection, echoing a typical multi-lepton search skim."""
    num_leptons = ak.num(events.Electron) + ak.num(events.Muon)
    return events[num_leptons >= 4]


def accumulate_skims(a, b):
    """Reducer for the skimmer processor: two chunks' (or groups')
    surviving events are just concatenated into one, larger, awkward
    array. Runs remotely, like the base reducer it replaces."""
    return ak.concatenate([a, b], axis=0)


def load_skim(results_dir, dataset_name, processor_name):
    """Final results land under results_dir/<dataset_name>/<processor_name>/
    as a single compressed, pickled file (name includes a random uuid, hence
    the glob). serialization.load reverses what the reducer wrote."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


def build_datasets(data_dir):
    """Builds the `input` dict VineReduceCoffea expects: coffea's own
    preprocessed-dataset shape, one entry per dataset, each with a
    "metadata" dict and a "files" dict mapping each file's path to
    {"object_path": ..., "num_entries": ...} - what
    coffea.dataset_tools.preprocess() itself produces, and what
    coffea_input_to_datasets (VineReduceCoffea's default input_to_datasets)
    knows how to read."""
    rng = np.random.default_rng()
    datasets = {}
    for dataset_name, lepton_mean in DATASET_LEPTON_MEANS.items():
        files = write_test_data.generate_dataset_files(
            data_dir, dataset_name, FILES_PER_DATASET, lepton_mean, rng
        )
        datasets[dataset_name] = {
            "metadata": {},
            "files": {
                path: {"object_path": "Events", "num_entries": num_events}
                for path, num_events in files.items()
            },
        }
    return datasets


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

    # Same TaskVineDistributor + vine.Factory setup as quick_start.py: one
    # local worker process, no cluster or separate vine_worker needed to run
    # this example standalone.
    distributor = TaskVineDistributor(
        port=0, resources_processor={"cores": 1}, resources_reducer={"cores": 1}
    )
    workers = vine.Factory(manager_host_port=f"localhost:{distributor.port}")
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1

    with workers:
        vr = VineReduceCoffea(
            processors={"skim_4lep": skimmer},
            input=datasets,
            reducer=accumulate_skims,
            chunksize=CHUNKSIZE,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            distributor=distributor,
        )
        vr.compute()
    distributor.shutdown()

    # Load each dataset's final skim, write it out as parquet (the form a
    # downstream analysis step would actually want), and report how many
    # events survived.
    skims = {}
    for dataset_name in datasets:
        skim = load_skim(results_dir, dataset_name, "skim_4lep")
        skims[dataset_name] = skim
        ak.to_parquet(skim, os.path.join(results_dir, f"{dataset_name}.parquet"))
        print(f"{dataset_name}: {len(skim)} events pass the >=4-lepton skim")

    # Sanity check tied to how the input data was generated (see
    # DATASET_LEPTON_MEANS): "signal" has a higher mean lepton count than
    # "background", so it should pass the skim more often.
    assert len(skims["signal"]) > len(
        skims["background"]
    ), "expected signal to pass the skim more often than background"
    print("OK: signal passes the >=4-lepton skim more often than background")


if __name__ == "__main__":
    main()
