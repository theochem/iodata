"""Test that the example scripts run without errors."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATHS = [
    path for path in Path(__file__).parent.glob("*.py") if not path.name.startswith("test_")
]


@pytest.mark.parametrize("path", SCRIPT_PATHS)
def test_something(path: Path):
    subprocess.run([sys.executable, path.name], check=True, cwd=path.parent)
