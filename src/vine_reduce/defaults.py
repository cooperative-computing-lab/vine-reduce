"""Default implementations for every user-overridable step of the pipeline.

Functions in this file run in two different places:
  - input_to_datasets, default_datasets_to_chunks, make_default_is_result:
    run locally, in the vine_reduce process.
  - default_chunk_to_args, executor_wrapper, default_reducer, reducer_wrapper:
    run remotely, on worker nodes. Results are written to dest_file with
    vine_reduce.serialization (cloudpickle + zstd), so processor/reducer/etc.
    may be closures or lambdas, and result files are compressed on disk.

The `executor` step itself (what executor_wrapper calls to actually run
processor(args)) lives in executor.py, not here - see SimpleExecutor,
CloudpickleExecutor, and DaskExecutor.
"""

from __future__ import annotations

import json
import resource
import time
import traceback as traceback_module
from typing import Any, Callable, Iterator

from . import serialization
from .executor import Executor
from .types import Chunk, RawOutcome


def default_input_to_datasets(input_data: str | dict[str, Any]) -> dict[str, Any]:
    """Loads the input description, either as an already-parsed dict or from
    a json file path. Expected shape in both cases:
    {dataset_name: {"metadata": {...}, "files": {url: num_entries, ...}}}."""
    if isinstance(input_data, dict):
        return input_data
    with open(input_data) as f:
        return json.load(f)


def default_datasets_to_chunks(
    dataset: dict[str, Any],
    current_chunksize: Callable[[], int | None],
    skip_files: set[str] | None = None,
) -> Iterator[Chunk]:
    """Yields Chunks per file, in file order. `current_chunksize` is polled
    once per file (not once per chunk), so a file already being split still
    finishes at the size it started with, but the next file picks up any
    chunksize change made since - this is how chunksize halving on resource
    exhaustion takes effect for chunks not yet generated. A chunksize of
    None means one chunk per file (all of its events). Files in skip_files
    (already covered by a checkpoint) are not chunked again."""
    skip_files = skip_files or set()
    for url, num_entries in dataset["files"].items():
        if url in skip_files:
            continue
        chunksize = current_chunksize()
        step = chunksize if chunksize is not None else num_entries
        start = 0
        while start < num_entries:
            stop = min(start + step, num_entries)
            yield Chunk(url=url, start=start, stop=stop)
            start = stop


def default_chunk_to_args(
    chunk: Chunk,
    dataset_metadata: dict[str, Any],
    distributor_metadata: dict[str, Any] | None = None,
) -> Chunk:
    """The chunk itself is the args; a real workflow would open chunk.url and
    read events [chunk.start, chunk.stop) into whatever shape the processor
    expects."""
    return chunk


def _measure(fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    start_time = time.monotonic()
    result = fn()
    wall_time_s = time.monotonic() - start_time
    # ru_maxrss is kilobytes on Linux, bytes on macOS/BSD; this module targets Linux.
    memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result, {"cores": 1, "memory_mb": memory_mb, "wall_time_s": wall_time_s}


def _unmeasured_resources() -> dict[str, Any]:
    """Placeholder usage for a call that failed before _measure could report
    anything real. A fresh dict each time, since it ends up on a RawOutcome."""
    return {"cores": 1, "memory_mb": 0, "wall_time_s": 0}


def _run_and_wrap(dest_file: str, run: Callable[[], Any]) -> RawOutcome:
    """Runs `run`, measuring resources, and serializes its result to dest_file
    on success. Shared tail for executor_wrapper and reducer_wrapper.

    dest_file is written unconditionally, even on failure/exhaustion: a
    distributor may declare it as a required task output before this
    function ever runs (see TaskVineDistributor.submit), so leaving it
    missing would make the *transport* report the failure instead of this
    function's own RawOutcome, discarding the real status and traceback."""
    try:
        result, resources = _measure(run)
        serialization.dump(result, dest_file)
    except MemoryError:
        serialization.dump(None, dest_file)
        return RawOutcome(status="exhausted", resources=_unmeasured_resources())
    except Exception:
        serialization.dump(None, dest_file)
        return RawOutcome(
            status="failure",
            resources=_unmeasured_resources(),
            traceback=traceback_module.format_exc(),
        )

    return RawOutcome(status="success", resources=resources, file=dest_file)


def executor_wrapper(
    dest_file: str,
    processor: Callable[[Any], Any],
    chunk: Chunk,
    dataset_metadata: dict[str, Any],
    distributor_metadata: dict[str, Any] | None,
    executor_metadata: dict[str, Any] | None,
    chunk_to_args: Callable[..., Any],
    executor: Executor,
) -> RawOutcome:
    """Runs remotely. Calls chunk_to_args then executor.submit, measures
    resources, and serializes the processing result to dest_file on
    success. executor is a fresh, just-deserialized copy for this call
    alone - shut down via `with executor:` once run() returns."""

    def run() -> Any:
        args = chunk_to_args(chunk, dataset_metadata, distributor_metadata)
        with executor:
            return executor.submit(
                processor,
                args,
                dataset_metadata=dataset_metadata,
                distributor_metadata=distributor_metadata,
                executor_metadata=executor_metadata,
            ).result()

    return _run_and_wrap(dest_file, run)


def default_reducer(a: Any, b: Any) -> Any:
    """Folds b into a via `a += b`. Works for anything supporting in-place
    addition (numbers, arrays, ...); does not know how to merge dicts or
    sets - see coffea.py's default_reducer for that."""
    a += b
    return a


def reducer_wrapper(
    dest_file: str,
    reducer: Callable[[Any, Any], Any],
    input_files: list[str],
    is_final: bool,
    result_postprocess: Callable[[Any], Any] | None,
) -> RawOutcome:
    """Runs remotely. Folds input_files (each a serialized result) together
    with reducer, applies result_postprocess if this is a final result, and
    serializes the outcome to dest_file on success."""

    def run() -> Any:
        acc = serialization.load(input_files[0])
        for path in input_files[1:]:
            other = serialization.load(path)
            acc = reducer(acc, other)
            del other
        if is_final and result_postprocess is not None:
            acc = result_postprocess(acc)
        return acc

    return _run_and_wrap(dest_file, run)


def make_default_is_result(total_events: int) -> Callable[[int, float, float], bool]:
    """A group is a final result once it covers every event of the dataset."""

    def is_result(num_events: int, total_time: float, total_memory: float) -> bool:
        return num_events >= total_events

    return is_result
