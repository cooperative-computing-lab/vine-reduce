from typing import TYPE_CHECKING

from .distributor import Distributor
from .engine import VineReduce
from .local_distributor import LocalDistributor
from .pipeline import VineReduceError
from .types import Chunk, Outcome, RawOutcome, ResourceExhaustion, RuntimeFailure, Success

if TYPE_CHECKING:
    from .taskvine_distributor import TaskVineDistributor

__all__ = [
    "Chunk",
    "Distributor",
    "LocalDistributor",
    "Outcome",
    "RawOutcome",
    "ResourceExhaustion",
    "RuntimeFailure",
    "Success",
    "TaskVineDistributor",
    "VineReduce",
    "VineReduceError",
]


def __getattr__(name: str):
    """TaskVineDistributor pulls in ndcctools (a heavy, optional dependency
    not required for LocalDistributor-based use), so it's imported lazily
    here rather than at module load time - `import vine_reduce` must not
    require ndcctools to be installed."""
    if name == "TaskVineDistributor":
        from .taskvine_distributor import TaskVineDistributor

        return TaskVineDistributor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
