"""Shared data types passed between vine_reduce and a distributor.

See PLAN.md for the full design. `Outcome` and its variants are the public,
distributor-facing result of a submitted call. `RawOutcome` is the internal,
distributor-agnostic value returned by executor_wrapper/reducer_wrapper on
the worker side; a distributor is responsible for attaching the result_id the
caller gave it at submit() time to produce a proper Outcome (see
distributor.py).
"""

from __future__ import annotations

from dataclasses import dataclass
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
class Outcome:
    """Base class for the result of a submitted call, as reported by a distributor.

    result_id: the id passed to Distributor.submit() for this call.
    resources: usage reported for the call, e.g. {"cores", "memory_mb",
        "wall_time_s"} - see each Distributor implementation for exactly
        which keys it fills in.
    """

    result_id: str
    resources: dict[str, Any]


@dataclass(frozen=True)
class Success(Outcome):
    """The call finished normally. `file` is the distributor's own handle
    for the result (a local path, or an opaque token - see distributor.py's
    module docstring), to be passed to Distributor.retrieve()."""

    file: str


@dataclass(frozen=True)
class RuntimeFailure(Outcome):
    """The call raised an exception. `traceback` is the remote traceback,
    formatted as text, for surfacing in a local VineReduceError."""

    traceback: str


@dataclass(frozen=True)
class ResourceExhaustion(Outcome):
    """The call was killed for exceeding its resource allocation (memory,
    wall time, disk). Carries no extra fields beyond result_id/resources."""


@dataclass(frozen=True)
class ResultHandle:
    """A completed result as the distributor knows it: result_id for
    release_result/retrieve/checkpoint_path, and file - the distributor's own
    opaque handle (Outcome.file) - for use inside a later submit()'s args."""

    result_id: str
    file: str


@dataclass(frozen=True)
class RawOutcome:
    """What executor_wrapper/reducer_wrapper return on the worker side, before a
    distributor attaches the result_id and turns it into a proper Outcome."""

    status: str  # "success" | "failure" | "exhausted"
    resources: dict[str, Any]
    file: str | None = None
    traceback: str | None = None

    def to_outcome(self, result_id: str) -> Outcome:
        """Attach result_id and convert to the matching Outcome subclass."""
        if self.status == "success":
            return Success(result_id=result_id, resources=self.resources, file=self.file)
        if self.status == "failure":
            return RuntimeFailure(
                result_id=result_id, resources=self.resources, traceback=self.traceback
            )
        if self.status == "exhausted":
            return ResourceExhaustion(result_id=result_id, resources=self.resources)
        raise ValueError(f"unknown RawOutcome status: {self.status!r}")
