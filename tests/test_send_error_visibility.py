from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from backend.cli.commands import send as send_command


class _FakeArtifact:
    def __init__(self) -> None:
        self.id = "job-123"
        self.name = "job-name"
        self.topic = "topic"
        self.created_at = datetime.now(timezone.utc)
        self.conversations: list[SimpleNamespace] = []


class _FakeStorage:
    def __init__(self, _runs_dir: Path) -> None:
        self.artifact = _FakeArtifact()

    def get_search_by_path(self, _path: Path):
        return self.artifact

    def save_search_at_path(self, _path: Path, _artifact) -> None:
        return None


def test_send_surfaces_job_store_lookup_failures() -> None:
    errors: list[str] = []

    def _error(message: str) -> int:
        errors.append(message)
        return 1

    def _broken_job_store_factory():
        raise RuntimeError("job store exploded")

    runs_dir = Path(".")
    search_file = Path("search_job123.json")
    search_file.write_text("{}", encoding="utf-8")
    try:
        deps = send_command.SendDependencies(
            logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
            runs_dir=runs_dir,
            search_storage_cls=_FakeStorage,
            reddit_service_cls=object,
            conversation_model=SimpleNamespace(model_validate=lambda payload: payload),
            search_artifact_model=lambda **kwargs: SimpleNamespace(**kwargs),
            now_fn=lambda: datetime.now(timezone.utc),
            job_store_factory=_broken_job_store_factory,
        )
        rc = send_command.handle(
            SimpleNamespace(search_file=search_file.name),
            lambda _data: None,
            _error,
            deps,
        )
    finally:
        search_file.unlink(missing_ok=True)

    assert rc == 1
    assert errors
    assert "job store exploded" in errors[0]
