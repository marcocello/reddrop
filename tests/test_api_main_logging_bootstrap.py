from __future__ import annotations

import types

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.config.log import get_app_log_config


def test_run_api_server_uses_shared_log_config(monkeypatch) -> None:
    calls: list[dict] = []

    def _fake_run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setitem(__import__("sys").modules, "uvicorn", types.SimpleNamespace(run=_fake_run))

    api_main.run_api_server(host="127.0.0.1", port=8011, reload=True)

    assert len(calls) == 1
    call = calls[0]
    assert call["args"] == ("backend.api.main:app",)
    assert call["kwargs"]["host"] == "127.0.0.1"
    assert call["kwargs"]["port"] == 8011
    assert call["kwargs"]["reload"] is True
    assert call["kwargs"]["log_config"] == get_app_log_config()


def _setup_isolated_runtime(tmp_path, monkeypatch):
    home = tmp_path / ".reddrop"
    runs = home / "runs"
    jobs = home / "jobs_config.json"
    monkeypatch.setattr(api_main, "DEFAULT_HOME", home)
    monkeypatch.setattr(api_main, "DEFAULT_RUNS_DIR", runs)
    monkeypatch.setattr(api_main, "DEFAULT_JOBS_CONFIG", jobs)
    try:
        api_main.runtime_manager.shutdown()
    except SchedulerNotRunningError:
        pass
    monkeypatch.setattr(api_main, "runtime_manager", api_main.JobRuntimeManager())
    return home, runs, jobs


def test_api_http_errors_are_logged_to_terminal(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    error_messages: list[str] = []

    class _FakeLogger:
        def info(self, message: str, *args) -> None:
            _ = (message, args)

        def error(self, message: str, *args) -> None:
            rendered = message % args if args else message
            error_messages.append(rendered)

    monkeypatch.setattr(api_main, "logger", _FakeLogger())

    with TestClient(api_main.app) as client:
        response = client.post("/search", json={"name": "missing-job"})

    assert response.status_code == 400
    assert any("Job not found" in item for item in error_messages)
