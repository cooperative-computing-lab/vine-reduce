"""Generates synthetic NanoAOD-like ROOT files for the coffea_skim example.

vine_reduce.VineReduceCoffea reads events with coffea's NanoAODSchema, which
expects a specific on-disk layout: a jagged collection like "Electron" is
stored as flat content branches ("Electron_pt", "Electron_eta", ...) plus a
matching integer counter branch ("nElectron") - NanoAOD's own "ragged array
via counter branch" format, distinct from ROOT's std::vector branches.
write_root_file below builds exactly that layout with uproot, via the
counter_name/field_name hooks TTree.mktree exposes for this purpose, so this
file needs no real CMS data or a coffea preprocessing step to produce
something NanoAODSchema can read.

Each event also gets Electron and Muon collections whose per-event lepton
count is Poisson-distributed, with "signal" datasets given a higher mean
than "background" (see coffea_skim.py's DATASET_LEPTON_MEANS) - enough that
the >=4-lepton skim in coffea_skim.py selects a very different fraction of
events per dataset, which is the point of running it at all.
"""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import uproot


def _field_name(outer: str, inner: str) -> str:
    return inner if outer == "" else f"{outer}_{inner}"


def _counter_name(counted: str) -> str:
    return f"n{counted}"


def _make_leptons(num_events: int, lepton_mean: float, rng: np.random.Generator) -> ak.Array:
    """A jagged {pt, eta, phi, mass, charge} collection, `lepton_mean`
    leptons per event on average (Poisson), each with plausible-looking
    kinematics."""
    counts = rng.poisson(lepton_mean, size=num_events)
    total = int(counts.sum())
    flat = ak.Array(
        {
            "pt": rng.uniform(10.0, 100.0, size=total).astype(np.float32),
            "eta": rng.uniform(-2.5, 2.5, size=total).astype(np.float32),
            "phi": rng.uniform(-np.pi, np.pi, size=total).astype(np.float32),
            "mass": np.full(total, 0.10566, dtype=np.float32),
            "charge": rng.choice(np.array([-1, 1], dtype=np.int32), size=total),
        }
    )
    return ak.unflatten(flat, counts)


def write_root_file(
    path: str, num_events: int, lepton_mean: float, rng: np.random.Generator
) -> None:
    """Writes one NanoAOD-shaped ROOT file with num_events events, an
    Electron and a Muon collection (see _make_leptons), under an "Events"
    tree - the layout coffea's NanoAODSchema (and this example's
    VineReduceCoffea) expects."""
    # NanoAODSchema requires these three run-identification branches even
    # though nothing downstream in this example actually reads them.
    events = ak.Array(
        {
            "run": np.ones(num_events, dtype=np.uint32),
            "luminosityBlock": np.ones(num_events, dtype=np.uint32),
            "event": np.arange(num_events, dtype=np.int64),
            "Electron": _make_leptons(num_events, lepton_mean, rng),
            "Muon": _make_leptons(num_events, lepton_mean, rng),
        }
    )
    with uproot.recreate(path) as f:
        branch_types = {name: events[name].type for name in events.fields}
        f.mktree("Events", branch_types, counter_name=_counter_name, field_name=_field_name)
        f["Events"].extend({name: events[name] for name in events.fields})


def generate_dataset_files(
    data_dir: str,
    dataset_name: str,
    num_files: int,
    lepton_mean: float,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Writes num_files ROOT files for dataset_name under data_dir, each with
    a random event count in [300, 500). Returns {absolute_path:
    num_events} - build_datasets in coffea_skim.py wraps this into the
    "files" shape a coffea-preprocessed dataset expects."""
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    files = {}
    for i in range(num_files):
        num_events = int(rng.integers(300, 500))
        path = os.path.abspath(os.path.join(dataset_dir, f"file_{i}.root"))
        write_root_file(path, num_events, lepton_mean, rng)
        files[path] = num_events
    return files
