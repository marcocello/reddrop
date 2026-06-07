from __future__ import annotations

from pathlib import Path

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main


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
    return home


def test_scenario_persona_api_persists_single_combined_objective_field(tmp_path, monkeypatch) -> None:
    home = _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        saved = client.put(
            "/personas",
            json={
                "personas": [
                    {
                        "name": "seller",
                        "objective": "Book intro calls. Direct, practical, and concise.",
                    }
                ]
            },
        )
        assert saved.status_code == 200
        payload = saved.json()
        assert payload["source"] == str(home / "personas.json")
        assert payload["personas"] == [
            {
                "name": "seller",
                "objective": "Book intro calls. Direct, practical, and concise.",
            }
        ]


def test_scenario_settings_and_ui_drop_reply_system_prompt() -> None:
    settings_ui = Path("frontend/src/features/settings/index.tsx").read_text(encoding="utf-8")
    assert "OpenRouter configuration" in settings_ui
    assert "Reply generation" not in settings_ui
    assert "System prompt" not in settings_ui

    personas_ui = Path("frontend/src/features/personas/index.tsx").read_text(encoding="utf-8")
    assert "Label htmlFor={`${prefix}-persona-description`}>Description</Label>" not in personas_ui
    assert "Label htmlFor={`${prefix}-persona-objective`}>Objective, tone & style</Label>" in personas_ui
    assert "## Objectives" in personas_ui
    assert "......" in personas_ui
    assert "# Tone & Style" in personas_ui
    assert "....." in personas_ui
    assert "Label htmlFor={`${prefix}-persona-tone-style`}>Tone & style</Label>" not in personas_ui
    assert "max-h-[80vh] space-y-4 overflow-y-auto pr-1" in personas_ui
