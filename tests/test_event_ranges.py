from __future__ import annotations

from vine_reduce.event_ranges import covers_all, uncovered_chunks, union_coverage, union_ranges
from vine_reduce.types import Chunk


def test_union_ranges_merges_overlapping_and_adjacent_ranges():
    assert union_ranges([(6, 8), (0, 2), (2, 4), (3, 5)]) == ((0, 5), (6, 8))
    assert union_ranges([]) == ()


def test_union_coverage_merges_per_file():
    merged = union_coverage(
        [
            {"a.root": ((0, 2),), "b.root": ((0, 1),)},
            {"a.root": ((2, 4),)},
        ]
    )
    assert merged == {"a.root": ((0, 4),), "b.root": ((0, 1),)}


def test_uncovered_chunks_with_nothing_covered_is_the_chunk_itself():
    chunk = Chunk("a.root", 0, 4)
    assert list(uncovered_chunks(chunk, ())) == [chunk]
    assert list(uncovered_chunks(chunk, ((4, 8),))) == [chunk]


def test_uncovered_chunks_cuts_out_the_covered_events():
    chunk = Chunk("a.root", 2, 10)
    covered = ((0, 3), (5, 6), (9, 12))
    assert list(uncovered_chunks(chunk, covered)) == [
        Chunk("a.root", 3, 5),
        Chunk("a.root", 6, 9),
    ]


def test_uncovered_chunks_fully_covered_yields_nothing():
    assert list(uncovered_chunks(Chunk("a.root", 2, 4), ((0, 8),))) == []


def test_covers_all_needs_every_event_of_every_nonempty_file():
    sizes = {"a.root": 4, "b.root": 2, "empty.root": 0}
    assert covers_all({"a.root": ((0, 4),), "b.root": ((0, 2),)}, sizes)
    assert not covers_all({"a.root": ((0, 4),), "b.root": ((0, 1),)}, sizes)
    assert not covers_all({"a.root": ((0, 2), (3, 4)), "b.root": ((0, 2),)}, sizes)
    assert not covers_all({"a.root": ((0, 4),)}, sizes)
