"""failed_files.log: a durable, human-readable record of every chunk or reduction vine_reduce
gave up on. One FailureLog is shared by all Pipelines in a VineReduce.compute() call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .distributor import TaskKind
from .types import ResourceUsage


@dataclass(frozen=True)
class FailureRecord:
    """One permanently-failed file's events, as written to failed_files.log.

    ranges: the [start, stop) event ranges of filename that were given up on.
    kind: which stage gave up.
    attempts: how many tries were made.
    resources_allocated / resources_measured: the configured cap (Distributor.resources(kind))
        and the last attempt's usage; either may be None.
    traceback: the last attempt's traceback, if any (a ResourceExhaustion carries none).
    """

    dataset_name: str
    filename: str
    ranges: tuple[tuple[int, int], ...]
    kind: TaskKind
    attempts: int
    resources_allocated: dict[str, Any] | None
    resources_measured: ResourceUsage | None
    traceback: str | None


class FailureLog:
    """Appends one text block per FailureRecord to `path`, written immediately rather than
    batched so the record survives a run abort (see PLAN.md's "Attempts and Retries")."""

    def __init__(self, path: str) -> None:
        self._path = path

    def log(self, record: FailureRecord) -> None:
        lines = [
            f"[{datetime.now(timezone.utc).isoformat()}] permanent failure",
            f"  dataset:    {record.dataset_name}",
            f"  file:       {record.filename}",
            "  events:     " + " ".join(f"[{start}:{stop}]" for start, stop in record.ranges),
            f"  stage:      {record.kind}",
            f"  attempts:   {record.attempts}",
            f"  resources allocated: {record.resources_allocated}",
            f"  resources measured:  {record.resources_measured}",
        ]
        if record.traceback:
            lines.append("  last traceback:")
            lines.extend(f"    {line}" for line in record.traceback.splitlines())
        lines.append("")
        with open(self._path, "a") as f:
            f.write("\n".join(lines) + "\n")
