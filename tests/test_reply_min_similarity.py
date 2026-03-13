from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.cli.commands import reply as reply_command
from backend.models import Conversation, PersonaProfile, SearchArtifact


@dataclass
class _FakeSearchStorage:
    artifact: SearchArtifact | None = None
    saved: SearchArtifact | None = None

    def __init__(self, _runs_dir: Path) -> None:
        pass

    def get_search_by_path(self, _path: Path) -> SearchArtifact | None:
        return self.__class__.artifact

    def save_search_at_path(self, _path: Path, artifact: SearchArtifact) -> None:
        self.__class__.saved = artifact


class _FakePersonaStore:
    def get_persona(self, _name: str) -> PersonaProfile | None:
        return PersonaProfile(name="seller", description="desc", objective="obj")


class _FakeReplyGenerator:
    def generate_reply(self, *, topic: str, conversation: Conversation, persona: PersonaProfile) -> str:
        _ = topic, persona
        return f"reply:{conversation.thread_id}"


def test_reply_command_uses_min_similarity_from_args(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    search_file = runs_dir / "search_test.json"
    search_file.write_text("{}", encoding="utf-8")

    _FakeSearchStorage.artifact = SearchArtifact(
        id="job-1",
        name="search-job",
        topic="AI launch",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        conversations=[
            Conversation(
                thread_id="high",
                title="High",
                subreddit="sales",
                url="https://reddit.com/high",
                score=10,
                num_comments=5,
                semantic_similarity=0.91,
            ),
            Conversation(
                thread_id="low",
                title="Low",
                subreddit="sales",
                url="https://reddit.com/low",
                score=8,
                num_comments=3,
                semantic_similarity=0.42,
            ),
        ],
    )
    _FakeSearchStorage.saved = None

    deps = reply_command.ReplyDependencies(
        logger=type("L", (), {"info": lambda *args, **kwargs: None})(),
        runs_dir=runs_dir,
        persona_store_cls=_FakePersonaStore,
        reply_generation_service_cls=_FakeReplyGenerator,
        search_storage_cls=_FakeSearchStorage,
        search_artifact_model=SearchArtifact,
        conversation_model=Conversation,
        now_fn=lambda: datetime.now(timezone.utc),
    )

    args = argparse.Namespace(
        search_file=search_file.name,
        personas="seller",
        replies=10,
        min_similarity=0.9,
    )

    emitted: dict = {}
    rc = reply_command.handle(args, lambda payload: emitted.update(payload), lambda _msg: 1, deps)
    assert rc == 0
    assert emitted["generated"] == 1
    assert emitted["min_similarity"] == 0.9

    assert _FakeSearchStorage.saved is not None
    saved_by_thread = {item.thread_id: item for item in _FakeSearchStorage.saved.conversations}
    assert saved_by_thread["high"].reply == "reply:high"
    assert saved_by_thread["low"].reply == ""
