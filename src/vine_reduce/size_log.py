"""results_dir/size.jsonl: one line per finished (processor, dataset) pipeline, recording the
chunk/reduction sizes it settled on and the peak resources its own tasks measured this run.
Shared by all pipelines of a run.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SizeRecord:
    """One finished pipeline's settled sizes and peak measured resources.

    chunk_size / reduction_size: the values in effect for the last chunk/reduction submitted;
        may have shrunk from the configured value through resource-exhaustion halving.
        chunk_size is None when chunking was unbounded (one chunk per file).
    processing / reduction: peak {"cores", "memory", "disk"} across this pipeline's successful
        processor/reducer tasks, rounded to 2 decimals; only keys the distributor reported are
        present. None if no task of that kind succeeded in this process (e.g. a restart that
        resumed entirely from checkpoints).
    """

    dataset_name: str
    processor_name: str
    chunk_size: int | None
    reduction_size: int
    processing: dict[str, float] | None
    reduction: dict[str, float] | None


class SizeLog:
    """Appends one json line per SizeRecord to `path`, written immediately like FailureLog."""

    def __init__(self, path: str) -> None:
        self._path = path

    def log(self, record: SizeRecord) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(dataclasses.asdict(record)) + "\n")
