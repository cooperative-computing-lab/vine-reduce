"""Which event ranges of which dataset files a result represents.

A checkpoint row, final result or pool item may hold only part of a file (a
file's chunks can be folded into different reductions), so restart needs to
know exactly which events are already covered. See "Temporary Results,
Checkpoints, and Restart" in PLAN.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

from .types import Chunk

# Sorted, disjoint, non-adjacent [start, stop) event ranges of one file.
Ranges = tuple[tuple[int, int], ...]
# Per dataset file URL; files with nothing covered are absent.
Coverage = Mapping[str, Ranges]


def union_ranges(ranges: Iterable[tuple[int, int]]) -> Ranges:
    """Sort and merge overlapping or adjacent ranges."""
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def union_coverage(coverages: Iterable[Coverage]) -> dict[str, Ranges]:
    """The coverage holding everything any of coverages holds."""
    gathered: dict[str, list[tuple[int, int]]] = {}
    for coverage in coverages:
        for url, ranges in coverage.items():
            gathered.setdefault(url, []).extend(ranges)
    return {url: union_ranges(ranges) for url, ranges in gathered.items()}


def uncovered_chunks(chunk: Chunk, covered: Ranges) -> Iterator[Chunk]:
    """The sub-chunks of chunk not inside covered (chunk itself, if nothing
    in it is covered)."""
    start = chunk.start
    for covered_start, covered_stop in covered:
        if covered_stop <= start:
            continue
        if covered_start >= chunk.stop:
            break
        if covered_start > start:
            yield Chunk(chunk.url, start, covered_start)
        start = max(start, covered_stop)
    if start < chunk.stop:
        yield Chunk(chunk.url, start, chunk.stop)


def covers_all(coverage: Coverage, file_sizes: Mapping[str, int]) -> bool:
    """Whether coverage holds every event of every file ({url: num_entries})."""
    return all(size == 0 or coverage.get(url) == ((0, size),) for url, size in file_sizes.items())
