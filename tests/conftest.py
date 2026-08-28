from __future__ import annotations

import heapq
import itertools
import json
import os
import shutil
from typing import Any, Callable

import pytest

from vine_reduce.types import RawOutcome


class FakeDistributor:
    """A synchronous, in-process stand-in for a Distributor: submit() runs
    the call immediately, and wait() hands back outcomes one at a time, in
    priority order, exactly as a real distributor would. Only suitable for
    testing vine_reduce's own logic in isolation - it executes closures
    directly rather than pickling them to a subprocess."""

    def __init__(self, work_dir: str, capacity_amount: int = 1000):
        self._work_dir = work_dir
        self._capacity_amount = capacity_amount
        self._seq = itertools.count()
        self._ready: list[tuple[int, int, str, RawOutcome]] = []
        self._files: dict[str, str] = {}

    def submit(
        self,
        result_id: str,
        priority: int,
        category: str,
        kind: str,
        func: Callable[..., Any],
        *args: Any,
        is_checkpoint: bool = False,
    ) -> None:
        dest_file = os.path.join(self._work_dir, f"{result_id}.pkl.zst")
        raw: RawOutcome = func(dest_file, *args)
        heapq.heappush(self._ready, (-priority, next(self._seq), result_id, raw))

    def wait(self, timeout: float | None = None):
        if not self._ready:
            return None
        _, _, result_id, raw = heapq.heappop(self._ready)
        if raw.status == "success":
            self._files[result_id] = raw.file
        return raw.to_outcome(result_id)

    def release_result(self, result_id: str) -> None:
        path = self._files.pop(result_id, None)
        if path is not None and os.path.exists(path):
            os.remove(path)

    def adopt_checkpoint(self, result_id: str, path: str) -> str:
        """Mirror LocalDistributor.adopt_checkpoint: register an existing
        on-disk checkpoint file under result_id, so it can be released/
        retrieved/checkpointed exactly like a this-run result."""
        self._files[result_id] = path
        return path

    def capacity(self) -> int:
        return self._capacity_amount

    def retrieve(self, result_id: str, dest_path: str) -> None:
        shutil.copy(self._files[result_id], dest_path)

    def checkpoint_path(self, result_id: str) -> str:
        return self._files[result_id]


@pytest.fixture
def fake_distributor(tmp_path):
    work_dir = tmp_path / "cluster"
    work_dir.mkdir()
    return FakeDistributor(str(work_dir))


def write_dataset_input(path: str, datasets: dict[str, Any]) -> str:
    with open(path, "w") as f:
        json.dump(datasets, f)
    return path


@pytest.fixture
def dataset_input(tmp_path):
    """Returns a function that writes a {dataset_name: {metadata, files}}
    dict to a json file under tmp_path and returns its path."""

    def _write(datasets: dict[str, Any], name: str = "input.json") -> str:
        return write_dataset_input(str(tmp_path / name), datasets)

    return _write
