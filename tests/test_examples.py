"""End-to-end tests that actually run the example scripts under examples/,
each as its own subprocess - exactly `pixi run python <script>.py`, the same
way a user would - against a real local TaskVine worker. Slower and heavier
than the rest of the suite (each one starts a manager, a vine_factory, and a
worker process, and waits for real tasks to run): these are integration
smoke tests for the examples themselves, not unit tests of vine_reduce's
internals.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("vine_factory") is None, reason="vine_factory not on PATH"
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
TIMEOUT = 600  # generous: covers worker connect latency plus real task execution


def _run_example(
    tmp_path: Path, example_dir_name: str, script_name: str
) -> subprocess.CompletedProcess:
    """Copies examples/<example_dir_name>'s .py files into an isolated tmp
    directory and runs script_name there, exactly as a user would from
    within examples/<example_dir_name>. Isolating into tmp_path keeps
    generated data/results/checkpoints out of the source tree.
    """
    src_dir = EXAMPLES_DIR / example_dir_name
    run_dir = tmp_path / example_dir_name
    run_dir.mkdir()
    for py_file in src_dir.glob("*.py"):
        shutil.copy(py_file, run_dir / py_file.name)

    return subprocess.run(
        [sys.executable, script_name],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def test_quick_start_example(tmp_path):
    result = _run_example(tmp_path, "quick_start", "quick_start.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: odd + even == all for every dataset" in result.stdout
