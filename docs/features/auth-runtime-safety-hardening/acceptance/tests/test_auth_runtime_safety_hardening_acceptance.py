from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.models import AuthProfile, Conversation, PersonaProfile, SearchArtifact
from backend.services.auth_store import AuthStore
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


def test_scenario_runtime_services_receive_auth_context_without_env_overwrite(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("LINKEDIN_USER_DATA_DIR", "/tmp/global-linkedin-profile")

    class _FakeLinkedinService:
        captured_auth = None
        seen_env_user_data_dir = ""

        @staticmethod
        def validate_saved_session(*, user_data_dir: str) -> bool:
            _ = user_data_dir
            return True

        def __init__(self, auth=None):
            _FakeLinkedinService.captured_auth = auth
            _FakeLinkedinService.seen_env_user_data_dir = os.getenv("LINKEDIN_USER_DATA_DIR", "")

        def post_comment(self, thread_id: str, comment_text: str) -> bool:
            _ = thread_id
            _ = comment_text
            return True

    monkeypatch.setattr(api_main, "LinkedinService", _FakeLinkedinService)

    with TestClient(api_main.app) as client:
        saved_auths = client.put("/auths", json={"auths": [{"id": "linkedin-auth", "social": "linkedin"}]})
        assert saved_auths.status_code == 200
        path = _create_frozen_search_path(
            client,
            name="linkedin-context-path",
            social="linkedin",
            topic="B2B thought leadership",
        )

        created = client.post(
            "/jobs",
            json={
                "name": "linkedin-context",
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
                        score=1,
                        num_comments=1,
                        semantic_similarity=0.9,
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
    assert _FakeLinkedinService.seen_env_user_data_dir == "/tmp/global-linkedin-profile"


def test_scenario_store_writes_are_atomic_and_non_truncating(tmp_path, monkeypatch) -> None:
    auth_store = AuthStore(base_dir=tmp_path / "auth", legacy_candidates=[])
    persona_path = tmp_path / "personas.json"
    persona_store = PersonaStore([persona_path])

    auth_reddit = tmp_path / "auth" / "reddit" / "auths.json"
    auth_linkedin = tmp_path / "auth" / "linkedin" / "auths.json"

    original_write_text = Path.write_text

    def guarded_write_text(self, *args, **kwargs):
        if self in {auth_reddit, auth_linkedin, persona_path}:
            raise AssertionError("direct write_text must not be used for atomic store persistence")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", guarded_write_text)

    auth_store.save_auths(
        [
            AuthProfile(
                id="reddit-main",
                social="reddit",
                reddit_client_id="id",
                reddit_client_secret="secret",
                reddit_username="alice",
                reddit_password="pw",
            )
        ]
    )
    persona_store.save_personas([PersonaProfile(name="seller", objective="o. direct tone.")])

    assert auth_reddit.exists()
    assert persona_path.exists()


def test_scenario_delete_requires_cascade_and_removes_linked_reply_jobs(tmp_path, monkeypatch) -> None:
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

        created_search = client.post(
            "/jobs",
            json={
                "name": "source-search",
                "job_type": "search",
                "search_query_id": path["id"],
                "auth_id": "reddit-main",
            },
        )
        assert created_search.status_code == 200
        search_job = created_search.json()

        created_reply = client.post(
            "/jobs",
            json={
                "name": "linked-reply",
                "job_type": "reply",
                "source_job_id": search_job["id"],
                "personas": ["seller"],
            },
        )
        assert created_reply.status_code == 200

        blocked = client.delete(f"/jobs/{search_job['id']}")
        assert blocked.status_code == 409
        assert "cascade=true" in str(blocked.json()["detail"]).lower()

        deleted = client.delete(f"/jobs/{search_job['id']}?cascade=true")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert deleted.json()["cascade"] is True


def test_scenario_linkedin_submit_requires_verifiable_signal(monkeypatch) -> None:
    class _SubmitButton:
        def __init__(self):
            self.clicks = 0

        def click(self, timeout: int = 0) -> None:
            _ = timeout
            self.clicks += 1

    class _InputBox:
        def __init__(self, text_after_submit: str):
            self._text = text_after_submit

        def inner_text(self, timeout: int = 0) -> str:
            _ = timeout
            return self._text

    class _Response:
        def __init__(self, status: int):
            self.status = status
            self.url = "https://www.linkedin.com/voyager/api/socialActions/urn:li:activity:1/comments"

            class _Request:
                method = "POST"

            self.request = _Request()

    class _SuccessCtx:
        def __init__(self, response: _Response):
            self.value = response

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type
            _ = exc
            _ = tb
            return False

    class _TimeoutCtx:
        def __enter__(self):
            raise RuntimeError("timeout")

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type
            _ = exc
            _ = tb
            return False

    class _Page:
        def __init__(self, ctx):
            self._ctx = ctx

        def expect_response(self, predicate, timeout: int = 0):
            _ = timeout
            if isinstance(self._ctx, _SuccessCtx):
                assert predicate(self._ctx.value)
            return self._ctx

        def wait_for_timeout(self, timeout: int) -> None:
            _ = timeout

    ok = LinkedinService._submit_comment_and_verify(
        page=_Page(_SuccessCtx(_Response(201))),
        input_box=_InputBox("still text"),
        submit_button=_SubmitButton(),
    )
    failed = LinkedinService._submit_comment_and_verify(
        page=_Page(_TimeoutCtx()),
        input_box=_InputBox("not cleared"),
        submit_button=_SubmitButton(),
    )
    monkeypatch.setattr(
        LinkedinService,
        "_page_has_posted_comment_text",
        staticmethod(lambda _page, _text: True),
    )
    visible = LinkedinService._submit_comment_and_verify(
        page=_Page(_TimeoutCtx()),
        input_box=_InputBox("still text"),
        submit_button=_SubmitButton(),
        expected_comment_text="Looks great",
    )

    assert ok is True
    assert failed is False
    assert visible is True

    class _ChallengePage:
        url = "https://www.linkedin.com/challenge/123"

    try:
        LinkedinService._ensure_authenticated(_ChallengePage())
    except RuntimeError:
        challenge_blocked = True
    else:
        challenge_blocked = False
    assert challenge_blocked is True
