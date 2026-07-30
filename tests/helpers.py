"""Toy functions shared across tests. Kept as plain, importable, module-level
callables (cloudpickle can serialize closures/lambdas too, but these are
reused across several test modules, so a shared name is simpler)."""

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
