from __future__ import annotations

import types

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
