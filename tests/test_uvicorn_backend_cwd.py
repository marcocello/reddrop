from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_api_main_imports_from_backend_cwd() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib; importlib.import_module('api.main')",
        ],
        cwd=backend_dir,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
