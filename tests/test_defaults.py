from __future__ import annotations

import os

from vine_reduce import serialization
from vine_reduce.defaults import (
    default_datasets_to_chunks,
    executor_wrapper,
    make_default_is_result,
    reducer_wrapper,
)
from vine_reduce.types import Chunk

from helpers import (
    count_events,
    double_postprocess,
    exhausting_processor,
    failing_processor,
    sum_reducer,
)


def test_default_datasets_to_chunks_splits_by_chunksize():
    dataset = {"files": {"a.root": 10, "b.root": 5}}
    chunks = list(
        default_datasets_to_chunks(dataset, current_chunksize=lambda: 4, skip_files=set())
    )
    assert chunks == [
        Chunk("a.root", 0, 4),
        Chunk("a.root", 4, 8),
        Chunk("a.root", 8, 10),
        Chunk("b.root", 0, 4),
        Chunk("b.root", 4, 5),
    ]


def test_default_datasets_to_chunks_whole_file_when_chunksize_none():
    dataset = {"files": {"a.root": 10, "b.root": 5}}
    chunks = list(
        default_datasets_to_chunks(dataset, current_chunksize=lambda: None, skip_files=set())
    )
    assert chunks == [Chunk("a.root", 0, 10), Chunk("b.root", 0, 5)]


def test_default_datasets_to_chunks_skips_files():
    dataset = {"files": {"a.root": 10, "b.root": 5}}
    chunks = list(
        default_datasets_to_chunks(dataset, current_chunksize=lambda: None, skip_files={"a.root"})
    )
    assert chunks == [Chunk("b.root", 0, 5)]


def test_default_datasets_to_chunks_reads_chunksize_fresh_per_file():
    dataset = {"files": {"a.root": 10, "b.root": 10}}
    sizes = iter([5, 2])
    chunks = list(
        default_datasets_to_chunks(dataset, current_chunksize=lambda: next(sizes), skip_files=set())
    )
    a_chunks = [c for c in chunks if c.url == "a.root"]
    b_chunks = [c for c in chunks if c.url == "b.root"]
    assert [c.num_events for c in a_chunks] == [5, 5]
    assert [c.num_events for c in b_chunks] == [2, 2, 2, 2, 2]


def test_executor_wrapper_success(tmp_path):
    dest = str(tmp_path / "out.pkl.zst")
    chunk = Chunk("a.root", 0, 5)
    outcome = executor_wrapper(
        dest,
        count_events,
        chunk,
        {},
        None,
        None,
        lambda c, dm, dmeta=None: c,
        lambda proc, args, dm, dmeta=None, emeta=None: proc(args),
    )
    assert outcome.status == "success"
    assert outcome.file == dest
    assert serialization.load(dest) == 5
    assert outcome.resources["wall_time_s"] >= 0


def test_executor_wrapper_failure_captures_traceback(tmp_path):
    dest = str(tmp_path / "out.pkl.zst")
    chunk = Chunk("a.root", 0, 5)
    outcome = executor_wrapper(
        dest,
        failing_processor,
        chunk,
        {},
        None,
        None,
        lambda c, dm, dmeta=None: c,
        lambda proc, args, dm, dmeta=None, emeta=None: proc(args),
    )
    assert outcome.status == "failure"
    assert "ValueError: boom" in outcome.traceback
    # dest_file must still be written on failure: a distributor (e.g.
    # TaskVineDistributor) may have already declared it as a required task
    # output, so leaving it missing would make the transport itself report
    # the failure and discard this RawOutcome (see taskvine_distributor.py).
    assert os.path.exists(dest)


def test_executor_wrapper_resource_exhaustion(tmp_path):
    dest = str(tmp_path / "out.pkl.zst")
    chunk = Chunk("a.root", 0, 5)
    outcome = executor_wrapper(
        dest,
        exhausting_processor,
        chunk,
        {},
        None,
        None,
        lambda c, dm, dmeta=None: c,
        lambda proc, args, dm, dmeta=None, emeta=None: proc(args),
    )
    assert outcome.status == "exhausted"
    assert os.path.exists(dest)


def test_reducer_wrapper_folds_inputs(tmp_path):
    inputs = []
    for i, value in enumerate([1, 2, 3]):
        path = str(tmp_path / f"in{i}.pkl.zst")
        serialization.dump(value, path)
        inputs.append(path)

    dest = str(tmp_path / "out.pkl.zst")
    outcome = reducer_wrapper(dest, sum_reducer, inputs, False, None)
    assert outcome.status == "success"
    assert serialization.load(dest) == 6


def test_reducer_wrapper_applies_postprocess_only_when_final(tmp_path):
    inputs = []
    for i, value in enumerate([1, 2]):
        path = str(tmp_path / f"in{i}.pkl.zst")
        serialization.dump(value, path)
        inputs.append(path)

    dest = str(tmp_path / "out.pkl.zst")
    outcome = reducer_wrapper(dest, sum_reducer, inputs, True, double_postprocess)
    assert outcome.status == "success"
    assert serialization.load(dest) == 6  # (1+2) * 2


def test_make_default_is_result_true_only_when_all_events_covered():
    is_result = make_default_is_result(total_events=100)
    assert is_result(50, 1.0, 1.0) is False
    assert is_result(100, 1.0, 1.0) is True
