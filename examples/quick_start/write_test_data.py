"""Generates binary test data files for the quick_start example.

Each file holds a random number of positive 4-byte unsigned integers,
written with struct.pack. Run standalone to regenerate examples/quick_start/data/,
or import generate_dataset_files() to do it programmatically (quick_start.py does
the latter, on every run).
"""

from __future__ import annotations

import os
import random
import struct

INT_FORMAT = "<I"
INT_SIZE = struct.calcsize(INT_FORMAT)
MIN_INTS_PER_FILE = 50
MAX_INTS_PER_FILE = 100


def write_file(path: str, num_ints: int, rng: random.Random) -> None:
    with open(path, "wb") as f:
        for _ in range(num_ints):
            f.write(struct.pack(INT_FORMAT, rng.randint(1, 2**32 - 1)))


def generate_dataset_files(
    data_dir: str, dataset_name: str, num_files: int, rng: random.Random
) -> dict[str, int]:
    """Writes num_files binary files for dataset_name under data_dir, each
    holding a random count of integers in [MIN_INTS_PER_FILE,
    MAX_INTS_PER_FILE]. Returns {absolute_path: num_entries}, the shape
    vine_reduce expects for a dataset's "files" entry."""
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    files = {}
    for i in range(num_files):
        num_ints = rng.randint(MIN_INTS_PER_FILE, MAX_INTS_PER_FILE)
        path = os.path.abspath(os.path.join(dataset_dir, f"file_{i}.bin"))
        write_file(path, num_ints, rng)
        files[path] = num_ints
    return files


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    rng = random.Random()
    for dataset_name in ("dataset_a", "dataset_b"):
        files = generate_dataset_files(data_dir, dataset_name, num_files=3, rng=rng)
        for path, num_ints in files.items():
            print(f"{dataset_name}: {path} ({num_ints} ints)")
