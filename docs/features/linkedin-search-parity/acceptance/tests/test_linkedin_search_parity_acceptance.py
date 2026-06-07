from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "backend" / "requirements.txt").exists():
            return parent
    raise AssertionError("Repository root not found.")


def _pytest_command(repo: Path) -> list[str]:
    venv_pytest = repo / ".venv" / "bin" / "pytest"
    if venv_pytest.exists():
        return [str(venv_pytest)]
    return ["python3", "-m", "pytest"]


def _python_command(repo: Path) -> str:
    venv_python = repo / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python3"


def _run_repo_tests(*node_ids: str) -> None:
    repo = _repo_root()
    env = {**os.environ, "PYTHONPATH": str(repo)}
    result = subprocess.run(
        [*_pytest_command(repo), *node_ids, "-q"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_api_is_available_through_real_http_boundary() -> None:
    repo = _repo_root()
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": str(repo)}
    process = subprocess.Popen(
        [
            _python_command(repo),
            "-m",
            "uvicorn",
            "backend.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=1)
                raise AssertionError(f"API server exited early.\n{stdout}\n{stderr}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    payload = response.read().decode("utf-8")
                assert '"status":"ok"' in payload
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.2)
        raise AssertionError(f"API health endpoint was not reachable: {last_error}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_linkedin_discovery_behavior_contract() -> None:
    _run_repo_tests("tests/test_linkedin_search_parity_bridge.py")


def test_linkedin_api_runtime_control_contract() -> None:
    _run_repo_tests(
        "tests/test_social_platform_api_contract.py::test_linkedin_search_job_passes_scroll_rounds_to_discovery",
        "tests/test_jobs_api_contract.py::test_linkedin_search_job_payload_includes_time_filter_and_sort_by",
    )
