"""Serialization for result files: cloudpickle wrapped in zstd compression.

cloudpickle (rather than stdlib pickle) is used because results, and the
functions that produce them, may be closures or lambdas. Compression is
streamed through zstandard's compressor/decompressor file objects, so the
pickled bytes are compressed and written (or read and decompressed) in
chunks rather than held in memory as one large buffer.
"""

from __future__ import annotations

from typing import Any

import cloudpickle
import zstandard


def dump(obj: Any, path: str) -> None:
    """Cloudpickles obj and zstd-compresses it, streaming straight to path."""
    cctx = zstandard.ZstdCompressor()
    with open(path, "wb") as f, cctx.stream_writer(f) as compressor:
        cloudpickle.dump(obj, compressor)


def load(path: str) -> Any:
    """Reverses dump: streams path through zstd decompression into cloudpickle."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f) as decompressor:
        return cloudpickle.load(decompressor)
