"""Picklable, module-level toy functions shared across tests. They must be
importable by name so ProcessPoolExecutor can send them to worker
subprocesses (closures/lambdas can't be pickled)."""

from __future__ import annotations


def count_events(chunk):
    return chunk.stop - chunk.start


def sum_reducer(a, b):
    return a + b


def double_postprocess(x):
    return x * 2


def failing_processor(chunk):
    raise ValueError("boom")


def exhausting_processor(chunk):
    raise MemoryError("simulated resource exhaustion")
