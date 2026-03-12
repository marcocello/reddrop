from __future__ import annotations

import asyncio

import pytest

from backend.services import llm_service as llm_service_module
from backend.services.reddit_service import RedditService


class _FakeComments:
    def __init__(self) -> None:
        self.replace_called = False

    def replace_more(self, limit: int = 0) -> None:
        _ = limit
        self.replace_called = True

    def list(self):
        return []


class _FakeSubreddit:
    display_name = "test"


class _FakeSubmission:
    id = "abc123"
    title = "title"
    subreddit = _FakeSubreddit()
    permalink = "/r/test/comments/abc123/title/"
    score = 1
    num_comments = 0
    created_utc = 0.0
    selftext = ""
    author = None
    is_self = True

    def __init__(self) -> None:
        self.comments = _FakeComments()


class _FakeRedditUser:
    @staticmethod
    def me():
        return "me"


class _FakeReddit:
    user = _FakeRedditUser()



def test_create_thread_data_stops_before_comment_fetch_when_stop_requested() -> None:
    service = RedditService.__new__(RedditService)
    service.reddit = _FakeReddit()

    submission = _FakeSubmission()

    with pytest.raises(RuntimeError, match="stopped by user request"):
        service._create_thread_data_from_submission(
            submission,
            search_query="q",
            stop_requested=lambda: True,
        )

    assert submission.comments.replace_called is False


def test_openrouter_client_uses_short_timeout_without_retries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast")
    monkeypatch.setenv("OPENROUTER_TIMEOUT_SECONDS", "7")
    monkeypatch.setattr(llm_service_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    service = llm_service_module.LLMService()

    assert service.client_type == "openrouter"
    assert captured["max_retries"] == 0
    assert captured["timeout"] == 7.0


def test_generate_response_sync_rebinds_client_when_loop_changes(monkeypatch) -> None:
    class _FakeMessage:
        content = "stub response"

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def __init__(self, owner) -> None:
            self.owner = owner

        async def create(self, **kwargs):
            _ = kwargs
            loop_id = id(asyncio.get_running_loop())
            if self.owner.bound_loop_id is None:
                self.owner.bound_loop_id = loop_id
            elif self.owner.bound_loop_id != loop_id:
                raise RuntimeError("Event loop is closed")
            return _FakeResponse()

    class _FakeChat:
        def __init__(self, owner) -> None:
            self.completions = _FakeCompletions(owner)

    class _FakeAsyncOpenAI:
        instances: list["_FakeAsyncOpenAI"] = []

        def __init__(self, **kwargs) -> None:
            _ = kwargs
            self.bound_loop_id = None
            self.chat = _FakeChat(self)
            self.__class__.instances.append(self)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast")
    monkeypatch.setattr(llm_service_module, "AsyncOpenAI", _FakeAsyncOpenAI)

    service = llm_service_module.LLMService()

    first = service.generate_response_sync("sys", "hello")
    second = service.generate_response_sync("sys", "hello again")

    assert first == "stub response"
    assert second == "stub response"
    assert len(_FakeAsyncOpenAI.instances) >= 2
