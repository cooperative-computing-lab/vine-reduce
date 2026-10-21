"""Shared data types passed between vine_reduce and a distributor.

Worker-side wrappers build an `Outcome` with result_id="" and std_output=None; the distributor
fills those in (and resources_allocated, if known) with dataclasses.replace() before returning it
from wait().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A contiguous range of events [start, stop) from a single file.

    url: the dataset file this chunk belongs to.
    start: index of the first event in the chunk.
    stop: index one past the last event in the chunk.
    """

    url: str
    start: int
    stop: int

    @property
    def num_events(self) -> int:
        """Number of events covered by this chunk (stop - start)."""
        return self.stop - self.start


@dataclass(frozen=True)
class ResourceUsage:
    """Usage measured (or allocated) for one call, in vine_reduce's units. A field is None when
    not reported (e.g. disk_mb); wall_time_s is only ever measured, never a cap."""

    cores: float | None = None
    memory_mb: float | None = None
    wall_time_s: float | None = None
    disk_mb: float | None = None

    def to_rmsummary(self) -> dict[str, float]:
        """Non-None fields under TaskVine's RMSummary names (cores/memory/disk)."""
        names = {"cores": self.cores, "memory": self.memory_mb, "disk": self.disk_mb}
        return {name: value for name, value in names.items() if value is not None}


UNMEASURED = ResourceUsage(cores=1, memory_mb=0, wall_time_s=0)
"""Placeholder usage for a synthesized outcome of a call that failed before measurement."""


@dataclass(frozen=True)
class Outcome:
    """Base class for the result of a submitted call, as reported by a distributor.

    result_id: the id passed to Distributor.submit().
    resources: usage reported for the call (which fields are filled depends on the distributor).
    resources_allocated: what the distributor actually gave the call, if it reports that; it can
        differ from the static per-category cap. None means the caller falls back to that cap.
    std_output: the call's stdout if the distributor can capture it, else None; used for debug
        printing on failure.
    """

    result_id: str
    resources: ResourceUsage
    std_output: str | None
    resources_allocated: dict[str, Any] | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class Success(Outcome):
    """The call finished normally. `file` is the distributor's own handle for the result (a
    local path or an opaque token)."""

    file: str


@dataclass(frozen=True)
class RuntimeFailure(Outcome):
    """The call raised an exception. `traceback` is the formatted remote traceback."""

    traceback: str


@dataclass(frozen=True)
class ResourceExhaustion(Outcome):
    """The call was killed for exceeding its resource allocation (memory, wall time, disk)."""


@dataclass(frozen=True)
class ResultHandle:
    """A completed result: result_id for release_result/retrieve/checkpoint_path, and the
    distributor's handle `file` (Outcome.file) for use in a later submit()'s args."""

    result_id: str
    file: str
