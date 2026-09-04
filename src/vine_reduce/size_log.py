"""size.jsonl: one line per finished (processor, dataset) pipeline, recording
the chunk/reduction sizes it settled on and the peak resources its own
processor/reducer tasks measured this run - see Pipeline._log_size_once, the
only caller. Lives at results_dir/size.jsonl - top-level, shared across every
pipeline of a run, unlike results_dir/<dataset>/<processor>/ which is
per-pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SizeRecord:
    """One finished pipeline's settled sizes and peak measured resources.

    chunk_size / reduction_size: the value in effect for the last chunk/
        reduction this pipeline submitted - may have shrunk from its
        originally configured value via resource-exhaustion halving (see
        Pipeline._handle_chunk_outcome/_handle_reduce_outcome). chunk_size is
        None when chunking was never bounded (one chunk per file).
    processing / reduction: peak {"cores", "memory", "disk"} measured across
        this pipeline's own successful processor/reducer tasks respectively,
        rounded to 2 decimals - only keys a distributor actually reported are
        present (e.g. "disk" is absent unless some distributor starts
        reporting disk usage). Either dict is None if this pipeline produced
        no successful task of that kind itself in this process (e.g. a
        restart that resumed entirely from checkpoints, running no fresh
        tasks at all).
    """

    dataset_name: str
    processor_name: str
    chunk_size: int | None
    reduction_size: int
    processing: dict[str, float] | None
    reduction: dict[str, float] | None


class SizeLog:
    """Appends one json line per SizeRecord to `path`, flushing immediately -
    same durability convention as failure_log.py's FailureLog."""

    def __init__(self, path: str) -> None:
        self._path = path

    def log(self, record: SizeRecord) -> None:
        row = {
            "dataset_name": record.dataset_name,
            "processor_name": record.processor_name,
            "chunk_size": record.chunk_size,
            "reduction_size": record.reduction_size,
            "processing": record.processing,
            "reduction": record.reduction,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(row) + "\n")
