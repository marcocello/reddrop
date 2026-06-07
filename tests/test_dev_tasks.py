import json
from pathlib import Path


def test_vscode_task_runs_backend_and_frontend() -> None:
    tasks = json.loads(Path(".vscode/tasks.json").read_text(encoding="utf-8"))

    labels = {task["label"]: task for task in tasks["tasks"]}
    combined = labels["Run Backend and Frontend"]

    assert combined["dependsOrder"] == "parallel"
    assert combined["dependsOn"] == ["Run Backend Server", "Run Frontend"]

    backend = labels["Run Backend Server"]
    assert backend["command"] == (
        "${workspaceFolder}/.venv/bin/python -m uvicorn backend.api.main:app --reload"
    )
    assert backend["options"]["cwd"] == "${workspaceFolder}"

    frontend = labels["Run Frontend"]
    assert frontend["command"] == "pnpm run dev"
    assert frontend["options"]["cwd"] == "${workspaceFolder}/frontend"
