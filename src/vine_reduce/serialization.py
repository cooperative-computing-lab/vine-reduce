"""Result-file serialization: cloudpickle (results and functions may be closures or lambdas)
streamed through zstd compression, so large pickles aren't held in memory as one buffer.
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
