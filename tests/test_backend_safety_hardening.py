from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.models import AuthProfile, Conversation, PersonaProfile, SearchArtifact
from backend.services.auth_store import AuthStore
from backend.services import linkedin_service as linkedin_service_module
from backend.services.linkedin_service import LinkedinService
from backend.services.persona_store import PersonaStore
from backend.services.search_storage import SearchStorage


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
    return home, runs, jobs


def _create_frozen_search_path(client: TestClient, *, name: str, social: str, topic: str) -> dict:
    generated = client.post(
        "/search-queries/generate",
        json={
            "name": name,
            "social": social,
            "topic": topic,
        },
    )
    assert generated.status_code == 200
    path = generated.json()["search_query"]
    saved = client.post(f"/search-queries/{path['id']}/save")
    assert saved.status_code == 200
    return saved.json()["search_query"]


def test_search_uses_explicit_auth_context_without_mutating_reddit_env(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("REDDIT_USERNAME", "global-user")
    monkeypatch.setenv("REDDIT_PASSWORD", "global-pass")

    class _FakeRedditService:
        captured_auth = None
        seen_env_username = ""

        @staticmethod
        def validate_credentials(**_kwargs) -> bool:
            return True

        def __init__(self, auth=None):
            _FakeRedditService.captured_auth = auth
            _FakeRedditService.seen_env_username = os.getenv("REDDIT_USERNAME", "")

        def discover_relevant_threads(self, **_kwargs):
            return []

    monkeypatch.setattr(api_main, "RedditService", _FakeRedditService)

    with TestClient(api_main.app) as client:
        auths = client.put(
            "/auths",
            json={
                "auths": [
                    {
                        "id": "reddit-auth",
                        "social": "reddit",
                        "reddit_client_id": "id",
                        "reddit_client_secret": "secret",
                        "reddit_username": "alice",
                        "reddit_password": "pw",
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(
            client,
            name="reddit-auth-scope-path",
            social="reddit",
            topic="AI launch strategy",
        )

        created = client.post(
            "/jobs",
            json={
                "name": "reddit-auth-scope",
                "search_query_id": path["id"],
                "auth_id": "reddit-auth",
            },
        )
        assert created.status_code == 200

        run = client.post("/search", json={"name": "reddit-auth-scope"})
        assert run.status_code == 200

    assert _FakeRedditService.captured_auth is not None
    assert _FakeRedditService.captured_auth.id == "reddit-auth"
    assert _FakeRedditService.seen_env_username == "global-user"
    assert os.getenv("REDDIT_USERNAME") == "global-user"
    assert os.getenv("REDDIT_PASSWORD") == "global-pass"


def test_thread_send_uses_explicit_auth_context_without_mutating_linkedin_env(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("LINKEDIN_USER_DATA_DIR", "/tmp/global-linkedin-profile")

    class _FakeLinkedinService:
        captured_auth = None
        seen_user_data_dir = ""

        @staticmethod
        def validate_saved_session(*, user_data_dir: str) -> bool:
            _ = user_data_dir
            return True

        def __init__(self, auth=None):
            _FakeLinkedinService.captured_auth = auth
            _FakeLinkedinService.seen_user_data_dir = os.getenv("LINKEDIN_USER_DATA_DIR", "")

        def post_comment(self, thread_id: str, comment_text: str) -> bool:
            _ = thread_id
            _ = comment_text
            return True

    monkeypatch.setattr(api_main, "LinkedinService", _FakeLinkedinService)

    with TestClient(api_main.app) as client:
        auths = client.put(
            "/auths",
            json={
                "auths": [
                    {
                        "id": "linkedin-auth",
                        "social": "linkedin",
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(
            client,
            name="linkedin-auth-scope-path",
            social="linkedin",
            topic="B2B thought leadership",
        )

        created = client.post(
            "/jobs",
            json={
                "name": "linkedin-auth-scope",
                "search_query_id": path["id"],
                "auth_id": "linkedin-auth",
            },
        )
        assert created.status_code == 200
        job = created.json()

        SearchStorage(api_main.DEFAULT_RUNS_DIR).save_search(
            SearchArtifact(
                id=job["id"],
                name=job["name"],
                topic="B2B thought leadership",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                conversations=[
                    Conversation(
                        thread_id="li-thread-1",
                        title="LinkedIn post",
                        subreddit="linkedin",
                        url="https://www.linkedin.com/feed/update/urn:li:activity:1",
                        score=4,
                        num_comments=2,
                        semantic_similarity=0.82,
                        user_has_commented=False,
                        reply="Looks great",
                    )
                ],
            )
        )

        sent = client.post(f"/threads/{job['id']}/linkedin/li-thread-1/send")
        assert sent.status_code == 200

    assert _FakeLinkedinService.captured_auth is not None
    assert _FakeLinkedinService.captured_auth.id == "linkedin-auth"
    assert _FakeLinkedinService.seen_user_data_dir == "/tmp/global-linkedin-profile"
    assert os.getenv("LINKEDIN_USER_DATA_DIR") == "/tmp/global-linkedin-profile"


def test_auth_store_save_auths_avoids_direct_write_text(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "auth"
    store = AuthStore(base_dir=base_dir, legacy_candidates=[])
    reddit_file = base_dir / "reddit" / "auths.json"
    linkedin_file = base_dir / "linkedin" / "auths.json"

    original_write_text = Path.write_text

    def guarded_write_text(self, *args, **kwargs):
        if self == reddit_file or self == linkedin_file:
            raise AssertionError("direct write_text must not be used for auth persistence")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guarded_write_text)

    store.save_auths(
        [
            AuthProfile(
                id="reddit-main",
                social="reddit",
                reddit_client_id="id",
                reddit_client_secret="secret",
                reddit_username="alice",
                reddit_password="pw",
            ),
            AuthProfile(
                id="linkedin-main",
                social="linkedin",
            ),
        ]
    )

    payload = json.loads(reddit_file.read_text(encoding="utf-8"))
    assert payload["auths"][0]["id"] == "reddit-main"


def test_persona_store_save_personas_avoids_direct_write_text(tmp_path, monkeypatch) -> None:
    personas_file = tmp_path / "personas.json"
    store = PersonaStore([personas_file])

    original_write_text = Path.write_text

    def guarded_write_text(self, *args, **kwargs):
        if self == personas_file:
            raise AssertionError("direct write_text must not be used for persona persistence")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guarded_write_text)

    store.save_personas([PersonaProfile(name="seller", objective="o. direct tone.")])

    payload = json.loads(personas_file.read_text(encoding="utf-8"))
    assert payload["personas"][0]["name"] == "seller"


def test_delete_source_search_job_requires_cascade_when_reply_jobs_are_linked(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    with TestClient(api_main.app) as client:
        auths = client.put(
            "/auths",
            json={
                "auths": [
                    {
                        "id": "reddit-main",
                        "social": "reddit",
                        "reddit_client_id": "id",
                        "reddit_client_secret": "secret",
                        "reddit_username": "alice",
                        "reddit_password": "pw",
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(
            client,
            name="source-search-path",
            social="reddit",
            topic="AI launch strategy",
        )

        search_created = client.post(
            "/jobs",
            json={
                "name": "source-search",
                "job_type": "search",
                "search_query_id": path["id"],
                "auth_id": "reddit-main",
            },
        )
        assert search_created.status_code == 200
        search_job = search_created.json()

        reply_created = client.post(
            "/jobs",
            json={
                "name": "linked-reply",
                "job_type": "reply",
                "source_job_id": search_job["id"],
                "personas": ["seller"],
            },
        )
        assert reply_created.status_code == 200

        blocked = client.delete(f"/jobs/{search_job['id']}")
        assert blocked.status_code == 409
        assert "cascade=true" in str(blocked.json()["detail"]).lower()

        deleted = client.delete(f"/jobs/{search_job['id']}?cascade=true")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert deleted.json()["cascade"] is True

        jobs = client.get("/jobs")
        assert jobs.status_code == 200
        names = {item["name"] for item in jobs.json()["jobs"]}
        assert "source-search" not in names
        assert "linked-reply" not in names


class _FakeSubmitButton:
    def __init__(self):
        self.clicks = 0

    def click(self, timeout: int = 0) -> None:
        _ = timeout
        self.clicks += 1


class _FakeInputBox:
    def __init__(self, text_after_submit: str):
        self._text_after_submit = text_after_submit

    def inner_text(self, timeout: int = 0) -> str:
        _ = timeout
        return self._text_after_submit


class _FakeInputBoxSequence:
    def __init__(self, values: list[str]):
        self._values = list(values)
        self._last = values[-1] if values else ""

    def inner_text(self, timeout: int = 0) -> str:
        _ = timeout
        if self._values:
            self._last = self._values.pop(0)
        return self._last


class _FakeResponse:
    def __init__(self, status: int, *, method: str = "POST", url: str = "https://www.linkedin.com/voyager/api/feedback"):
        self.status = status
        self.url = url

        class _Request:
            def __init__(self, request_method: str):
                self.method = request_method

        self.request = _Request(method)


class _ExpectResponseSuccess:
    def __init__(self, response: _FakeResponse):
        self.value = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _ExpectResponseTimeout:
    def __enter__(self):
        raise RuntimeError("timeout")

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _FakePage:
    def __init__(self, expect_response_ctx):
        self._expect_response_ctx = expect_response_ctx

    def expect_response(self, predicate, timeout: int = 0):
        _ = timeout
        if isinstance(self._expect_response_ctx, _ExpectResponseSuccess):
            assert predicate(self._expect_response_ctx.value)
        return self._expect_response_ctx

    def wait_for_timeout(self, timeout: int) -> None:
        _ = timeout


class _StaticLocatorList:
    def __init__(self, items):
        self._items = list(items)

    def count(self) -> int:
        return len(self._items)

    def nth(self, idx: int):
        return self._items[idx]


class _VisibleButtonCandidate:
    def is_visible(self) -> bool:
        return True

    def get_attribute(self, name: str):
        if name in {"disabled", "aria-disabled"}:
            return None
        if name == "class":
            return ""
        return None


class _SelectorScope:
    def __init__(self, mapping: dict[str, list[object]]):
        self._mapping = mapping

    def locator(self, selector: str):
        return _StaticLocatorList(self._mapping.get(selector, []))

    def wait_for_timeout(self, timeout: int) -> None:
        _ = timeout


class _UrlPage:
    def __init__(self, url: str):
        self.url = url


def test_linkedin_submit_verification_accepts_server_ack() -> None:
    page = _FakePage(
        _ExpectResponseSuccess(
            _FakeResponse(201, url="https://www.linkedin.com/voyager/api/socialActions/urn:li:activity:1/comments"),
        )
    )
    submit_button = _FakeSubmitButton()
    input_box = _FakeInputBox("still there")

    verified = LinkedinService._submit_comment_and_verify(page=page, input_box=input_box, submit_button=submit_button)

    assert verified is True
    assert submit_button.clicks == 1


def test_linkedin_submit_verification_rejects_unverified_submit() -> None:
    page = _FakePage(_ExpectResponseTimeout())
    submit_button = _FakeSubmitButton()
    input_box = _FakeInputBox("not cleared")

    verified = LinkedinService._submit_comment_and_verify(page=page, input_box=input_box, submit_button=submit_button)

    assert verified is False
    assert submit_button.clicks == 2


def test_linkedin_submit_verification_rejects_non_comment_voyager_ack_without_clear_editor() -> None:
    page = _FakePage(
        _ExpectResponseSuccess(
            _FakeResponse(201, url="https://www.linkedin.com/voyager/api/feed"),
        )
    )
    submit_button = _FakeSubmitButton()
    input_box = _FakeInputBox("still there")

    verified = LinkedinService._submit_comment_and_verify(page=page, input_box=input_box, submit_button=submit_button)

    assert verified is False
    assert submit_button.clicks == 2


def test_linkedin_submit_verification_accepts_delayed_editor_clear() -> None:
    page = _FakePage(_ExpectResponseTimeout())
    submit_button = _FakeSubmitButton()
    input_box = _FakeInputBoxSequence(["still there", "still there", ""])

    verified = LinkedinService._submit_comment_and_verify(page=page, input_box=input_box, submit_button=submit_button)

    assert verified is True
    assert submit_button.clicks == 1


def test_linkedin_submit_verification_accepts_visible_posted_comment(monkeypatch) -> None:
    page = _FakePage(_ExpectResponseTimeout())
    submit_button = _FakeSubmitButton()
    input_box = _FakeInputBox("still there")
    monkeypatch.setattr(
        LinkedinService,
        "_page_has_posted_comment_text",
        staticmethod(lambda _page, _text: True),
        raising=False,
    )

    verified = LinkedinService._submit_comment_and_verify(
        page=page,
        input_box=input_box,
        submit_button=submit_button,
        expected_comment_text="Looks great",
    )

    assert verified is True
    assert submit_button.clicks == 1


def test_linkedin_submit_verification_retries_submit_click_before_failing(monkeypatch) -> None:
    page = _FakePage(_ExpectResponseTimeout())
    submit_button = _FakeSubmitButton()
    input_box = _FakeInputBox("still there")
    monkeypatch.setattr(
        LinkedinService,
        "_page_has_posted_comment_text",
        staticmethod(lambda _page, _text: submit_button.clicks >= 2),
    )

    verified = LinkedinService._submit_comment_and_verify(
        page=page,
        input_box=input_box,
        submit_button=submit_button,
        expected_comment_text="Looks great",
    )

    assert verified is True
    assert submit_button.clicks == 2


def test_linkedin_wait_for_submit_button_accepts_generic_submit_selector() -> None:
    button = _VisibleButtonCandidate()
    page = _SelectorScope({"div.comments-comment-box form button[type='submit']": [button]})

    submit_button, disabled_seen = LinkedinService._wait_for_submit_button(page=page, timeout_ms=50)

    assert submit_button is button
    assert disabled_seen is False


def test_linkedin_ensure_authenticated_rejects_challenge_page() -> None:
    page = _UrlPage("https://www.linkedin.com/challenge/123")

    try:
        LinkedinService._ensure_authenticated(page)
    except RuntimeError as exc:
        assert "not authenticated" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for LinkedIn challenge page")


def test_linkedin_context_auth_state_accepts_session_cookie(monkeypatch) -> None:
    monkeypatch.setattr(linkedin_service_module, "_has_linkedin_session_cookie", lambda _context: True)
    monkeypatch.setattr(linkedin_service_module, "_probe_context_authentication", lambda _context: False)

    assert LinkedinService._context_appears_authenticated(object()) is True


def test_linkedin_context_auth_state_accepts_feed_probe_when_cookie_missing(monkeypatch) -> None:
    monkeypatch.setattr(linkedin_service_module, "_has_linkedin_session_cookie", lambda _context: False)
    monkeypatch.setattr(linkedin_service_module, "_probe_context_authentication", lambda _context: True)

    assert LinkedinService._context_appears_authenticated(object()) is True


def test_linkedin_context_auth_state_rejects_when_cookie_and_probe_fail(monkeypatch) -> None:
    monkeypatch.setattr(linkedin_service_module, "_has_linkedin_session_cookie", lambda _context: False)
    monkeypatch.setattr(linkedin_service_module, "_probe_context_authentication", lambda _context: False)

    assert LinkedinService._context_appears_authenticated(object()) is False
