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
import json
import operator
from collections.abc import Mapping, MutableMapping, MutableSet
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from coffea.nanoevents import NanoAODSchema

from .engine import VineReduce
from .types import Chunk

T = TypeVar("T")


@runtime_checkable
class Addable(Protocol):
    def __add__(self: T, other: T) -> T: ...


Accumulatable = Addable | MutableSet | MutableMapping


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
        lhs, rhs = set(a), set(b)
        for key in lhs:
            if key in rhs:
                a[key] = default_reducer(a[key], b[key])
        for key in rhs - lhs:
            a[key] = copy.deepcopy(b[key])
        return a
    raise ValueError(f"Cannot add accumulators of incompatible type ({type(a)} vs. {type(b)})")


def coffea_input_to_datasets(input_data: str | dict[str, Any]) -> dict[str, Any]:
    """Converts coffea's own preprocess() output into vine_reduce's dataset
    shape. coffea describes each file with a dict carrying num_entries (plus
    steps/uuid/object_path); vine_reduce only needs the event count per file.
    input_data may be that dict directly, or a path to a json file holding it."""
    if isinstance(input_data, dict):
        raw = input_data
    else:
        with open(input_data) as f:
            raw = json.load(f)

    datasets = {}
    for name, spec in raw.items():
        files = {url: file_info["num_entries"] for url, file_info in spec["files"].items()}
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


@dataclass
class VineReduceCoffea(VineReduce):
    schema: Any = NanoAODSchema
    mode: str = "virtual"
    object_path: str = "Events"
    uproot_options: Mapping[str, Any] | None = None
    processor_args: Mapping[str, Any] | None = None
    reducer: Callable[[Any, Any], Any] = default_reducer
    input_to_datasets: Callable[[str | dict[str, Any]], dict[str, Any]] = coffea_input_to_datasets

    def __post_init__(self) -> None:
        self.chunk_to_args = _make_chunk_to_args(
            self.schema, self.mode, self.uproot_options, self.object_path
        )
        self.executor = _make_executor(self.processor_args)
