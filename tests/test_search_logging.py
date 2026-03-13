from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from backend.cli.commands import search as search_command
from backend.services import reddit_service as reddit_service_module


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
    last_kwargs: dict | None = None

    def discover_relevant_threads(self, *args, **kwargs):
        _ = args
        self.__class__.last_kwargs = dict(kwargs)
        return [
            {
                "id": "t1",
                "title": "title",
                "subreddit": "startups",
                "url": "https://reddit.com/x",
                "score": 10,
                "num_comments": 2,
                "created_utc": 0.0,
                "selftext": "body",
                "semantic_similarity": 0.9,
                "user_has_commented": False,
            }
        ]


def test_search_job_logs_starting_with_name_and_id() -> None:
    fake_logger = _FakeLogger()
    deps = search_command.SearchDependencies(
        logger=fake_logger,
        search_storage_factory=_FakeStorage,
        reddit_service_cls=_FakeRedditService,
        conversation_model=lambda **kwargs: kwargs,
        search_artifact_model=_FakeArtifact,
        now_fn=lambda: datetime.now(timezone.utc),
    )
    job = SimpleNamespace(
        id="job-123",
        name="meshify",
        topic="Meshify pipeline workflow",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
    )
    storage = _FakeStorage()

    _ = search_command._search_job(job, storage, deps)

    assert fake_logger.messages
    assert fake_logger.messages[0] == "starting: meshify - job-123"
    assert _FakeRedditService.last_kwargs is not None
    assert _FakeRedditService.last_kwargs["job_id"] == "job-123"


def test_discover_relevant_threads_logs_steps_with_job_prefix(monkeypatch) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr(reddit_service_module, "logger", fake_logger)

    service = reddit_service_module.RedditService.__new__(reddit_service_module.RedditService)
    service.reddit = object()
    service.initialization_error = ""
    service._raise_if_stopped = lambda should_stop: None
    service._analyze_content = lambda content: {
        "intent": "sharing",
        "topics": ["sales automation", "CRM integration"],
        "semantic_keywords": ["meshify", "pipeline"],
        "suggested_subreddits": ["startups"],
    }
    service._find_relevant_subreddits = lambda content_analysis, limit, stop_requested: [{"name": "startups"}]
    service._generate_search_queries = lambda content_analysis: ["meshify crm"]
    service._search_threads_directly = lambda **kwargs: [{"id": "t1"}]
    service._llm_filter_and_rank_threads = lambda **kwargs: []

    _ = service.discover_relevant_threads(
        content="Meshify as an execution layer",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
        job_id="job-123",
    )

    assert any(message.startswith("job-123 - step 1/5: Analyzing topic") for message in fake_logger.messages)
    assert any(message.startswith("job-123 - step 1/5 output: topic=") for message in fake_logger.messages)


def test_discover_relevant_threads_logs_step_5_even_with_zero_candidates(monkeypatch) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr(reddit_service_module, "logger", fake_logger)

    service = reddit_service_module.RedditService.__new__(reddit_service_module.RedditService)
    service.reddit = object()
    service.initialization_error = ""
    service._raise_if_stopped = lambda should_stop: None
    service._analyze_content = lambda content: {
        "intent": "sharing",
        "topics": ["sales automation"],
        "semantic_keywords": ["meshify"],
        "suggested_subreddits": ["startups"],
    }
    service._find_relevant_subreddits = lambda content_analysis, limit, stop_requested: [{"name": "startups"}]
    service._generate_search_queries = lambda content_analysis: ["meshify crm"]
    service._search_threads_directly = lambda **kwargs: []
    service._llm_filter_and_rank_threads = lambda **kwargs: []

    rows = service.discover_relevant_threads(
        content="Meshify as an execution layer",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
        job_id="job-123",
    )

    assert rows == []
    assert any(message.startswith("job-123 - step 5/5: Ranking 0 candidate thread(s).") for message in fake_logger.messages)
