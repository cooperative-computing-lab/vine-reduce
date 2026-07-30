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

    def __init__(self, work_dir: str, hungry_amount: int = 1000):
        self._work_dir = work_dir
        self._hungry_amount = hungry_amount
        self._next_id = itertools.count(1)
        self._seq = itertools.count()
        self._ready: list[tuple[int, int, int, RawOutcome]] = []
        self._files: dict[int, str] = {}

    def submit(
        self, priority: int, category: str, kind: str, func: Callable[..., Any], *args: Any
    ) -> int:
        result_id = next(self._next_id)
        dest_file = os.path.join(self._work_dir, f"{result_id}.pkl.zst")
        raw: RawOutcome = func(dest_file, *args)
        heapq.heappush(self._ready, (-priority, next(self._seq), result_id, raw))
        return result_id

    def wait(self, timeout: float | None = None):
        if not self._ready:
            return None
        _, _, result_id, raw = heapq.heappop(self._ready)
        if raw.status == "success":
            self._files[result_id] = raw.file
        return raw.to_outcome(result_id)

    def free_result(self, result_id: int) -> None:
        self._files.pop(result_id, None)

    def hungry(self) -> int:
        return self._hungry_amount

    def retrieve(self, result_id: int, dest_path: str) -> None:
        shutil.copy(self._files[result_id], dest_path)


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
