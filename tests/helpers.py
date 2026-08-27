"""Toy functions shared across tests. Kept as plain, importable, module-level
callables (cloudpickle can serialize closures/lambdas too, but these are
reused across several test modules, so a shared name is simpler)."""

from __future__ import annotations

import os
import threading


def read_env_var(chunk):
    return os.environ.get("VINE_REDUCE_TEST_VAR", "")


def read_shipped_file(chunk):
    with open("shipped.txt") as f:
        return f.read()


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


def unpicklable_processor(chunk):
    return threading.Lock()  # cloudpickle cannot serialize a lock
