"""Shared data types passed between vine_reduce and a distributor.

See PLAN.md for the full design. `Outcome` and its variants are the public,
distributor-facing result of a submitted call. `RawOutcome` is the internal,
distributor-agnostic value returned by executor_wrapper/reducer_wrapper on
the worker side; a distributor is responsible for attaching the result_id it
assigned at submit() time to produce a proper Outcome (see distributor.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A contiguous range of events [start, stop) from a single file."""

    url: str
    start: int
    stop: int

    @property
    def num_events(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class Outcome:
    """Base class for the result of a submitted call, as reported by a distributor."""

    result_id: int
    resources: dict[str, Any]


@dataclass(frozen=True)
class Success(Outcome):
    file: str


@dataclass(frozen=True)
class RuntimeFailure(Outcome):
    traceback: str


@dataclass(frozen=True)
class ResourceExhaustion(Outcome):
    pass


@dataclass(frozen=True)
class RawOutcome:
    """What executor_wrapper/reducer_wrapper return on the worker side, before a
    distributor attaches the result_id and turns it into a proper Outcome."""

    status: str  # "success" | "failure" | "exhausted"
    resources: dict[str, Any]
    file: str | None = None
    traceback: str | None = None

    def to_outcome(self, result_id: int) -> Outcome:
        if self.status == "success":
            return Success(result_id=result_id, resources=self.resources, file=self.file)
        if self.status == "failure":
            return RuntimeFailure(
                result_id=result_id, resources=self.resources, traceback=self.traceback
            )
        if self.status == "exhausted":
            return ResourceExhaustion(result_id=result_id, resources=self.resources)
        raise ValueError(f"unknown RawOutcome status: {self.status!r}")
