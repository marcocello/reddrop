import json
from pathlib import Path


def test_run_backend_and_frontend_task_is_executable_contract() -> None:
    tasks = json.loads(Path(".vscode/tasks.json").read_text(encoding="utf-8"))
    by_label = {task["label"]: task for task in tasks["tasks"]}

    combined = by_label["Run Backend and Frontend"]
    assert combined["dependsOn"] == ["Run Backend Server", "Run Frontend"]
    assert combined["dependsOrder"] == "parallel"

    backend_command = by_label["Run Backend Server"]["command"]
    assert backend_command.startswith("${workspaceFolder}/.venv/bin/python ")
    assert "backend.api.main:app" in backend_command
    assert by_label["Run Frontend"]["command"] == "pnpm run dev"
