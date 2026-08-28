"""VineReduceCoffea: a VineReduce specialization for coffea-based HEP
workflows.

It supplies the coffea-specific pieces of the pipeline - reading NanoEvents
out of a Chunk, materializing awkward arrays after the processor runs, and
folding coffea-style accumulators together - while chunking, checkpointing,
and restart are inherited unchanged from VineReduce. See PLAN.md for the
overall design.
"""

from __future__ import annotations

import copy
import hashlib
import json
import operator
import os
from collections.abc import Mapping, MutableMapping, MutableSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from coffea.nanoevents import NanoAODSchema

from .engine import VineReduce
from .types import Chunk

T = TypeVar("T")


@runtime_checkable
class Addable(Protocol):
    """Anything supporting `a + b`, e.g. a histogram or a plain number."""

    def __add__(self: T, other: T) -> T: ...


Accumulatable = Addable | MutableSet | MutableMapping
"""A value default_reducer knows how to merge: an Addable, a mutable set
(merged via union), or a mutable mapping (merged key-by-key, recursively)."""


def default_reducer(a: Accumulatable, b: Accumulatable) -> Accumulatable:
    """Add two accumulatables together, assuming the first is mutable.
    Handles plain addables (histograms, numbers), sets, and nested mappings -
    the shapes coffea processors typically return. Lifted from coffea's own
    accumulate() helper, since base VineReduce's default reducer (`a += b`)
    does not know how to merge dicts."""
    if isinstance(a, Addable) and isinstance(b, Addable):
        return operator.add(a, b)
    if isinstance(a, MutableSet) and isinstance(b, MutableSet):
        return operator.or_(a, b)
    if isinstance(a, MutableMapping) and isinstance(b, MutableMapping):
        if not isinstance(b, type(a)):
            raise ValueError(
                f"Cannot add two mappings of incompatible type ({type(a)} vs. {type(b)})"
            )
        # Snapshot both key sets up front, since the loops below mutate `a`.
        a_keys, b_keys = set(a), set(b)
        for key in a_keys & b_keys:
            a[key] = default_reducer(a[key], b[key])
        for key in b_keys - a_keys:
            a[key] = copy.deepcopy(b[key])
        return a
    raise ValueError(f"Cannot add accumulators of incompatible type ({type(a)} vs. {type(b)})")


def coffea_input_to_datasets(input_data: str | dict[str, Any]) -> dict[str, Any]:
    """Converts coffea's own preprocess() output into vine_reduce's dataset
    shape. coffea describes each file with a dict carrying num_entries (plus
    steps/uuid/object_path); vine_reduce only needs the event count per file.
    input_data may be that dict directly, or a path to a json file holding it.

    Raises ValueError if any file is missing a concrete num_entries - this
    means the fileset hasn't been preprocessed yet. Run
    VineReduceCoffea.preprocess_cache(fileset, cache_file=...) first and pass
    its result (or the cache_file path) in as input_data."""
    if isinstance(input_data, dict):
        raw = input_data
    else:
        with open(input_data) as f:
            raw = json.load(f)

    datasets = {}
    for name, spec in raw.items():
        files = {}
        for url, file_info in spec["files"].items():
            num_entries = file_info.get("num_entries") if isinstance(file_info, Mapping) else None
            if num_entries is None:
                raise ValueError(
                    f"File {url!r} in dataset {name!r} has no num_entries; the fileset "
                    "has not been preprocessed. Run "
                    "VineReduceCoffea.preprocess_cache(fileset, cache_file=...) first "
                    "and pass its result as input."
                )
            files[url] = num_entries
        datasets[name] = {"metadata": spec.get("metadata", {}), "files": files}
    return datasets


def _materialize(obj: Any) -> Any:
    """Recursively force any virtual awkward arrays in a processor's result
    to materialize, so the result is fully computed before it gets pickled
    and sent back over the wire."""
    import awkward as ak

    if isinstance(obj, ak.Array):
        return ak.materialize(obj)
    if isinstance(obj, dict):
        return {key: _materialize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_materialize(value) for value in obj)
    return obj


def _make_chunk_to_args(
    schema: Any, mode: str, uproot_options: Mapping[str, Any] | None, object_path: str
) -> Callable[[Chunk, dict[str, Any], dict[str, Any] | None], Any]:
    """Builds a chunk_to_args that opens chunk.url at object_path and returns
    the NanoEvents for [chunk.start, chunk.stop). Runs remotely, at the
    worker node handling the chunk."""
    uproot_options = dict(uproot_options or {})

    def chunk_to_args(
        chunk: Chunk,
        dataset_metadata: dict[str, Any],
        distributor_metadata: dict[str, Any] | None = None,
    ) -> Any:
        from coffea.nanoevents import NanoEventsFactory

        return NanoEventsFactory.from_root(
            {chunk.url: object_path},
            entry_start=chunk.start,
            entry_stop=chunk.stop,
            metadata=dict(dataset_metadata),
            schemaclass=schema,
            uproot_options=uproot_options,
            mode=mode,
        ).events()

    return chunk_to_args


def _make_executor(processor_args: Mapping[str, Any] | None) -> Callable[..., Any]:
    """Builds an executor that calls the processor on the NanoEvents produced
    by chunk_to_args, then materializes any virtual arrays in its result."""
    processor_args = dict(processor_args or {})

    def executor(
        processor: Callable[..., Any],
        events: Any,
        dataset_metadata: dict[str, Any],
        distributor_metadata: dict[str, Any] | None = None,
        executor_metadata: dict[str, Any] | None = None,
    ) -> Any:
        result = processor(events, **processor_args)
        return _materialize(result)

    return executor


def _checksum_fileset(fileset: dict[str, Any]) -> str:
    """A stable hash of a fileset's contents, used to detect whether a
    preprocess_cache entry is still valid for the given input fileset."""
    encoded = json.dumps(fileset, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_preprocess_cache(cache_file: str | Path, checksum: str) -> dict[str, Any] | None:
    """Reads a preprocess_cache jsonl file and returns its cached
    preprocessed fileset if its stored checksum matches, else None (a cache
    miss - including on a missing, truncated, or otherwise corrupt file)."""
    try:
        with open(cache_file) as f:
            header = json.loads(f.readline())
            if header.get("checksum") != checksum:
                return None
            cached = json.loads(f.readline())
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(cached, dict):
        return None
    return cached


def _write_preprocess_cache(
    cache_file: str | Path, checksum: str, preprocessed: dict[str, Any]
) -> None:
    """Writes a preprocess_cache jsonl file: a header line with the fileset's
    checksum, followed by the preprocessed fileset. Writes to a temp file and
    renames into place so a crash mid-write can't leave a corrupt cache_file
    (a reader would just treat that as a cache miss anyway, but this avoids
    it in the common case)."""
    tmp_path = f"{cache_file}.tmp"
    with open(tmp_path, "w") as f:
        f.write(json.dumps({"checksum": checksum}) + "\n")
        f.write(json.dumps(preprocessed) + "\n")
    os.replace(tmp_path, cache_file)


@dataclass
class VineReduceCoffea(VineReduce):
    """A VineReduce specialization for coffea-based HEP analyses: supplies
    NanoEvents-reading (chunk_to_args), awkward-array materialization
    (executor), and coffea-style accumulator merging (reducer), while
    chunking, checkpointing, and restart are inherited unchanged from
    VineReduce. `processors` values here take one `events` NanoEvents array
    and return any picklable, accumulatable result (see default_reducer).
    See the README's "HEP / coffea workflows" section and PLAN.md.

    schema: the coffea NanoEvents schema class used to interpret each ROOT
        file, e.g. NanoAODSchema (default).
    mode: NanoEventsFactory.from_root's `mode` - "virtual" (default) for
        lazily-materialized awkward arrays, or "eager"/"dask" per coffea's
        own NanoEventsFactory docs.
    object_path: the ROOT TTree name to read events from, e.g. "Events"
        (default).
    uproot_options: extra keyword options forwarded to
        NanoEventsFactory.from_root's `uproot_options`.
    processor_args: extra keyword arguments passed to every processor call,
        in addition to its `events` argument.
    reducer: overrides VineReduce's default_reducer with this module's
        coffea-aware default_reducer, which also merges sets and mappings
        (dicts of histograms, as coffea processors commonly return).
    input_to_datasets: overrides VineReduce's default with
        coffea_input_to_datasets, which reads coffea's own preprocess()
        output (a dict, or a path to the json file holding it) instead of
        vine_reduce's plain dataset shape.
    """

    schema: Any = NanoAODSchema
    mode: str = "virtual"
    object_path: str = "Events"
    uproot_options: Mapping[str, Any] | None = None
    processor_args: Mapping[str, Any] | None = None
    reducer: Callable[[Any, Any], Any] = default_reducer
    input_to_datasets: Callable[[str | dict[str, Any]], dict[str, Any]] = coffea_input_to_datasets

    def __post_init__(self) -> None:
        """Builds chunk_to_args/executor from the fields above, the way a
        user of plain VineReduce would pass them in directly."""
        self.chunk_to_args = _make_chunk_to_args(
            self.schema, self.mode, self.uproot_options, self.object_path
        )
        self.executor = _make_executor(self.processor_args)

    @staticmethod
    def preprocess_cache(
        fileset: dict[str, Any],
        cache_file: str | Path,
        step_size: None | int = None,
        align_clusters: bool = False,
        recalculate_steps: bool = False,
        files_per_batch: int = 1,
        skip_bad_files: bool = False,
        file_exceptions: Any = (OSError,),
        save_form: bool = False,
        scheduler: None | Callable | str = None,
        uproot_options: dict[str, Any] | None = None,
        step_size_safety_factor: float = 0.5,
        allow_empty_datasets: bool = False,
    ) -> dict[str, Any]:
        """Runs coffea's own dataset_tools.preprocess() on fileset, caching
        the result at cache_file so unchanged filesets don't get
        re-preprocessed on every run.

        cache_file is a jsonl file: its first line is a header dict holding
        a checksum of fileset, and its second line is the preprocessed
        fileset (coffea's "available" return value - the files it was able
        to fully probe). If cache_file exists and its checksum matches
        fileset's, preprocess() is skipped and the cached result is
        returned as-is. Otherwise (no cache, stale checksum, or a corrupt
        cache file) preprocess() is run and cache_file is (re)written.

        All other arguments are forwarded to coffea.dataset_tools.preprocess
        unchanged - see its docs. Returns coffea's rich per-file dict shape
        (not vine_reduce's flat shape); pass the result as input to
        coffea_input_to_datasets (e.g. via VineReduceCoffea's default
        input_to_datasets)."""
        checksum = _checksum_fileset(fileset)
        cached = _read_preprocess_cache(cache_file, checksum)
        if cached is not None:
            return cached

        from coffea.dataset_tools import preprocess

        available, _updated = preprocess(
            fileset,
            step_size=step_size,
            align_clusters=align_clusters,
            recalculate_steps=recalculate_steps,
            files_per_batch=files_per_batch,
            skip_bad_files=skip_bad_files,
            file_exceptions=file_exceptions,
            save_form=save_form,
            scheduler=scheduler,
            uproot_options=uproot_options or {},
            step_size_safety_factor=step_size_safety_factor,
            allow_empty_datasets=allow_empty_datasets,
        )
        _write_preprocess_cache(cache_file, checksum, available)
        return available
