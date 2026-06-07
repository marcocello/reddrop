from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from backend.cli.commands import search as search_command


class _FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args) -> None:
        rendered = message % args if args else message
        self.messages.append(rendered)


class _FakeArtifact:
    def __init__(self, **kwargs) -> None:
        self.id = kwargs["id"]
        self.name = kwargs["name"]
        self.topic = kwargs["topic"]
        self.created_at = kwargs["created_at"]
        self.updated_at = kwargs["updated_at"]
        self.conversations = kwargs["conversations"]

    def model_dump(self, mode: str = "json") -> dict:
        _ = mode
        return {
            "id": self.id,
            "name": self.name,
            "topic": self.topic,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "conversations": self.conversations,
        }


class _FakeStorage:
    def __init__(self) -> None:
        self._items: dict[str, _FakeArtifact] = {}

    def get_search(self, job_id: str):
        return self._items.get(job_id)

    def save_search(self, artifact: _FakeArtifact) -> None:
        self._items[artifact.id] = artifact


class _FakeRedditService:
    def discover_relevant_threads(self, *args, **kwargs):
        _ = args
        _ = kwargs
        return []


class _FakeLinkedinService:
    def discover_relevant_threads(self, *args, **kwargs):
        _ = args
        _ = kwargs
        return []


def _deps(logger: _FakeLogger) -> search_command.SearchDependencies:
    return search_command.SearchDependencies(
        logger=logger,
        search_storage_factory=_FakeStorage,
        reddit_service_cls=_FakeRedditService,
        linkedin_service_cls=_FakeLinkedinService,
        conversation_model=lambda **kwargs: kwargs,
        search_artifact_model=_FakeArtifact,
        now_fn=lambda: datetime.now(timezone.utc),
    )


def test_runtime_logs_linkedin_effective_query_and_search_path() -> None:
    logger = _FakeLogger()
    job = SimpleNamespace(
        id="job-ln-acceptance",
        name="ln-job",
        social="linkedin",
        topic="long topic",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
        scroll_rounds=12,
        sort_by="recency",
        search_query_id="path-ln-acceptance",
        search_query_version=3,
        search_plan=SimpleNamespace(model_dump=lambda mode="python": {"query": "pluto AND pizza"}),
    )

    _ = search_command._search_job(job, _FakeStorage(), _deps(logger))
    log_line = next((message for message in logger.messages if "searching linkedin conversations" in message), "")

    assert "search_query=path-ln-acceptance@v3" in log_line
    assert "query='pluto AND pizza'" in log_line


def test_runtime_logs_reddit_effective_plan_inputs_and_search_path() -> None:
    logger = _FakeLogger()
    job = SimpleNamespace(
        id="job-rd-acceptance",
        name="rd-job",
        social="reddit",
        topic="crm sales",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
        search_query_id="path-rd-acceptance",
        search_query_version=7,
        search_plan=SimpleNamespace(
            model_dump=lambda mode="python": {
                "topics": ["crm"],
                "subreddits": ["sales", "entrepreneur"],
                "queries": ["crm sales"],
            }
        ),
    )

    _ = search_command._search_job(job, _FakeStorage(), _deps(logger))
    log_line = next((message for message in logger.messages if "searching reddit conversations" in message), "")

    assert "search_query=path-rd-acceptance@v7" in log_line
    assert "topics=['crm']" in log_line
    assert "subreddits=['sales', 'entrepreneur']" in log_line
    assert "queries=['crm sales']" in log_line
