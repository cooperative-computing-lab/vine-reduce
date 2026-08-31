"""VineReduce: a dynamic data reduction framework for data processing. See
PLAN.md for the full design and the README for usage.

Re-exports the package's public API: VineReduce/VineReduceError (engine.py),
Distributor (the interface a distributor implements), LocalDistributor and
TaskVineDistributor (the two Distributor implementations shipped here),
Executor (the interface an executor implements) and its SimpleExecutor/
CloudpickleExecutor/DaskExecutor implementations, get_environment/
UnstagedChanges (remote_environment.py), and the Chunk/Outcome family of
types shared between vine_reduce and a distributor.
"""

from typing import TYPE_CHECKING

from .distributor import Distributor
from .engine import VineReduce
from .executor import CloudpickleExecutor, DaskExecutor, Executor, SimpleExecutor
from .local_distributor import LocalDistributor
from .pipeline import VineReduceError
from .remote_environment import UnstagedChanges, get_environment
from .types import Chunk, Outcome, RawOutcome, ResourceExhaustion, RuntimeFailure, Success

if TYPE_CHECKING:
    from .taskvine_distributor import TaskVineDistributor

__all__ = [
    "Chunk",
    "CloudpickleExecutor",
    "DaskExecutor",
    "Distributor",
    "Executor",
    "LocalDistributor",
    "Outcome",
    "RawOutcome",
    "ResourceExhaustion",
    "RuntimeFailure",
    "SimpleExecutor",
    "Success",
    "TaskVineDistributor",
    "UnstagedChanges",
    "VineReduce",
    "VineReduceError",
    "get_environment",
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
