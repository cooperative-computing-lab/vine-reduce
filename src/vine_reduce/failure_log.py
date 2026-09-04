"""failed_files.log: a durable, human-readable record of every file
vine_reduce gave up on. Shared across every Pipeline in a run (one
FailureLog per VineReduce.compute() call) - see Pipeline._give_up_on_file
and Pipeline._handle_reduce_outcome's failure branches, the only callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class FailureRecord:
    """One permanently-failed file, as written to failed_files.log.

    kind: "processor" | "reducer" - which stage gave up on this file.
    attempts: how many tries were made before giving up.
    resources_allocated / resources_measured: the configured cap
        (Distributor.resources(kind)) and the last attempt's actual usage,
        if the distributor reported one - either may be None.
    traceback: the last attempt's captured traceback, if any (a
        ResourceExhaustion carries none).
    """

    dataset_name: str
    filename: str
    kind: str
    attempts: int
    resources_allocated: dict[str, Any] | None
    resources_measured: dict[str, Any] | None
    traceback: str | None


class FailureLog:
    """Appends one text block per FailureRecord to `path`, flushing
    immediately - a permanent failure is written the moment it is found,
    not batched, so the record survives even if the run aborts right
    after (see PLAN.md's "Attempts and Retries")."""

    def __init__(self, path: str) -> None:
        self._path = path

    def log(self, record: FailureRecord) -> None:
        lines = [
            f"[{datetime.now(timezone.utc).isoformat()}] permanent failure",
            f"  dataset:    {record.dataset_name}",
            f"  file:       {record.filename}",
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
