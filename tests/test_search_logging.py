from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backend.cli.commands import search as search_command
from backend.models import Conversation, SearchArtifact
from backend.services import reddit_service as reddit_service_module
from backend.services.search_storage import SearchStorage


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


class _FakeLinkedinService:
    last_kwargs: dict | None = None

    def discover_relevant_threads(self, *args, **kwargs):
        _ = args
        self.__class__.last_kwargs = dict(kwargs)
        return []


class _FailingRedditService:
    def discover_relevant_threads(self, *args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("discovery failed")


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
        name="example",
        topic="Example pipeline workflow",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
        search_plan=SimpleNamespace(
            model_dump=lambda mode="python": {
                "topics": ["Example pipeline workflow"],
                "subreddits": ["startups"],
                "queries": ["Example pipeline workflow"],
            }
        ),
    )
    storage = _FakeStorage()

    _ = search_command._search_job(job, storage, deps)

    assert fake_logger.messages
    assert fake_logger.messages[0] == "starting: example - job-123"
    assert _FakeRedditService.last_kwargs is not None
    assert _FakeRedditService.last_kwargs["job_id"] == "job-123"


def test_search_job_logs_linkedin_query_and_search_path_context() -> None:
    fake_logger = _FakeLogger()
    deps = search_command.SearchDependencies(
        logger=fake_logger,
        search_storage_factory=_FakeStorage,
        reddit_service_cls=_FakeRedditService,
        linkedin_service_cls=_FakeLinkedinService,
        conversation_model=lambda **kwargs: kwargs,
        search_artifact_model=_FakeArtifact,
        now_fn=lambda: datetime.now(timezone.utc),
    )
    job = SimpleNamespace(
        id="job-ln-1",
        name="example-linkedin",
        social="linkedin",
        topic="very long topic that should not be the only runtime signal",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
        scroll_rounds=15,
        sort_by="recency",
        search_query_id="path-ln-1",
        search_query_version=2,
        search_plan=SimpleNamespace(model_dump=lambda mode="python": {"query": "pluto AND pizza"}),
    )
    storage = _FakeStorage()

    _ = search_command._search_job(job, storage, deps)

    log_line = next((message for message in fake_logger.messages if "searching linkedin conversations" in message), "")
    assert "search_query=path-ln-1@v2" in log_line
    assert "query='pluto AND pizza'" in log_line
    assert "sort_by=recency" in log_line


def test_search_job_logs_reddit_plan_topics_subreddits_and_queries() -> None:
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
        id="job-rd-1",
        name="example-reddit",
        social="reddit",
        topic="crm sales",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
        search_query_id="path-rd-1",
        search_query_version=4,
        search_plan=SimpleNamespace(
            model_dump=lambda mode="python": {
                "topics": ["crm", "sales execution"],
                "subreddits": ["sales", "smallbusiness"],
                "queries": ["crm sales", "sales process"],
            }
        ),
    )
    storage = _FakeStorage()

    _ = search_command._search_job(job, storage, deps)

    log_line = next((message for message in fake_logger.messages if "searching reddit conversations" in message), "")
    assert "search_query=path-rd-1@v4" in log_line
    assert "topics=['crm', 'sales execution']" in log_line
    assert "subreddits=['sales', 'smallbusiness']" in log_line
    assert "queries=['crm sales', 'sales process']" in log_line


def test_search_rerun_failure_keeps_existing_conversations(tmp_path) -> None:
    storage = SearchStorage(tmp_path / "runs")
    now = datetime.now(timezone.utc)
    storage.save_search(
        SearchArtifact(
            id="job-123",
            name="example",
            topic="crm sales",
            created_at=now,
            updated_at=now,
            conversations=[
                Conversation(
                    thread_id="t1",
                    title="Existing thread",
                    subreddit="sales",
                    url="https://reddit.com/r/sales/comments/t1",
                    score=12,
                    num_comments=4,
                    semantic_similarity=0.7,
                    user_has_commented=False,
                    reply="existing draft",
                )
            ],
        )
    )

    deps = search_command.SearchDependencies(
        logger=_FakeLogger(),
        search_storage_factory=lambda: storage,
        reddit_service_cls=_FailingRedditService,
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=lambda: datetime.now(timezone.utc),
    )
    job = SimpleNamespace(
        id="job-123",
        name="example",
        topic="crm sales",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
        search_plan=SimpleNamespace(
            model_dump=lambda mode="python": {
                "topics": ["crm sales"],
                "subreddits": ["sales"],
                "queries": ["crm sales"],
            }
        ),
    )

    with pytest.raises(RuntimeError, match="discovery failed"):
        search_command._search_job(job, storage, deps)

    persisted = storage.get_search("job-123")
    assert persisted is not None
    assert len(persisted.conversations) == 1
    assert persisted.conversations[0].thread_id == "t1"
    assert persisted.conversations[0].reply == "existing draft"


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
        "semantic_keywords": ["example", "pipeline"],
        "suggested_subreddits": ["startups"],
    }
    service._find_relevant_subreddits = lambda content_analysis, limit, stop_requested: [{"name": "startups"}]
    service._generate_search_queries = lambda content_analysis: ["example crm"]
    service._search_threads_directly = lambda **kwargs: [{"id": "t1"}]
    service._llm_filter_and_rank_threads = lambda **kwargs: []

    _ = service.discover_relevant_threads(
        content="Example as an execution layer",
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
        "semantic_keywords": ["example"],
        "suggested_subreddits": ["startups"],
    }
    service._find_relevant_subreddits = lambda content_analysis, limit, stop_requested: [{"name": "startups"}]
    service._generate_search_queries = lambda content_analysis: ["example crm"]
    service._search_threads_directly = lambda **kwargs: []
    service._llm_filter_and_rank_threads = lambda **kwargs: []

    rows = service.discover_relevant_threads(
        content="Example as an execution layer",
        time_filter="week",
        subreddit_limit=5,
        threads_limit=10,
        job_id="job-123",
    )

    assert rows == []
    assert any(message.startswith("job-123 - step 5/5: Ranking 0 candidate thread(s).") for message in fake_logger.messages)
