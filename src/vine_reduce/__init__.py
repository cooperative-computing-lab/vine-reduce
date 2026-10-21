"""VineReduce: a dynamic data reduction framework. See PLAN.md for the design and the README for
usage. Re-exports the public API.
"""

import importlib.util
from typing import TYPE_CHECKING

from .distributor import Distributor
from .engine import VineReduce
from .executor import CloudpickleExecutor, DaskExecutor, Executor, SimpleExecutor
from .local_distributor import LocalDistributor
from .pipeline import VineReduceError
from .remote_environment import UnstagedChanges, get_environment
from .types import Chunk, Outcome, ResourceExhaustion, RuntimeFailure, Success

if TYPE_CHECKING:
    from .taskvine_distributor import TaskVineDistributor  # noqa: F401

__all__ = [
    "Chunk",
    "CloudpickleExecutor",
    "DaskExecutor",
    "Distributor",
    "Executor",
    "LocalDistributor",
    "Outcome",
    "ResourceExhaustion",
    "RuntimeFailure",
    "SimpleExecutor",
    "Success",
    "UnstagedChanges",
    "VineReduce",
    "VineReduceError",
    "get_environment",
]

if TYPE_CHECKING or importlib.util.find_spec("ndcctools") is not None:
    __all__.append("TaskVineDistributor")


def __getattr__(name: str):
    """Lazily import TaskVineDistributor so `import vine_reduce` doesn't require the heavy,
    optional ndcctools."""
    if name == "TaskVineDistributor":
        from .taskvine_distributor import TaskVineDistributor  # noqa: F811

        return TaskVineDistributor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
