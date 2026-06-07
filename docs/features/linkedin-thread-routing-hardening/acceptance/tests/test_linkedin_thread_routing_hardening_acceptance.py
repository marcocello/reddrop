from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.models import Conversation, SearchArtifact
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


def test_scenario_send_accepts_slash_thread_ids(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    captured: dict[str, str] = {}

    class _FakeLinkedinService:
        def post_comment(self, thread_id: str, comment_text: str) -> bool:
            captured["thread_id"] = thread_id
            captured["comment_text"] = comment_text
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
            name="linkedin-acceptance-send-path",
            social="linkedin",
            topic="B2B thought leadership",
        )

        created = client.post(
            "/jobs",
            json={
                "name": "linkedin-acceptance-send",
                "search_query_id": path["id"],
                "auth_id": "linkedin-auth",
            },
        )
        assert created.status_code == 200
        job = created.json()

        thread_id = "https://www.linkedin.com/posts/john-doe_activity-1234567890-abcD/"
        SearchStorage(api_main.DEFAULT_RUNS_DIR).save_search(
            SearchArtifact(
                id=job["id"],
                name=job["name"],
                topic="B2B thought leadership",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                conversations=[
                    Conversation(
                        thread_id=thread_id,
                        title="LinkedIn post",
                        subreddit="linkedin",
                        url=thread_id,
                        score=4,
                        num_comments=2,
                        semantic_similarity=0.82,
                        user_has_commented=False,
                        reply="Looks great",
                    )
                ],
            )
        )

        encoded_thread_id = quote(thread_id, safe="")
        sent = client.post(f"/threads/{job['id']}/linkedin/{encoded_thread_id}/send")
        assert sent.status_code == 200
        assert sent.json()["user_has_commented"] is True

    assert captured["thread_id"] == thread_id
    assert captured["comment_text"] == "Looks great"


def test_scenario_send_retries_stored_url_when_urn_send_fails(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    attempts: list[str] = []

    class _FakeLinkedinService:
        def post_comment(self, thread_id: str, comment_text: str) -> bool:
            _ = comment_text
            attempts.append(thread_id)
            return thread_id.startswith("https://www.linkedin.com/feed/update/")

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
            name="linkedin-acceptance-urn-fallback-path",
            social="linkedin",
            topic="B2B thought leadership",
        )

        created = client.post(
            "/jobs",
            json={
                "name": "linkedin-acceptance-urn-fallback",
                "search_query_id": path["id"],
                "auth_id": "linkedin-auth",
            },
        )
        assert created.status_code == 200
        job = created.json()

        thread_id = "urn:li:activity:7441444178565312512"
        thread_url = "https://www.linkedin.com/feed/update/urn:li:activity:7441444178565312512/"
        SearchStorage(api_main.DEFAULT_RUNS_DIR).save_search(
            SearchArtifact(
                id=job["id"],
                name=job["name"],
                topic="B2B thought leadership",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                conversations=[
                    Conversation(
                        thread_id=thread_id,
                        title="LinkedIn post",
                        subreddit="linkedin",
                        url=thread_url,
                        score=4,
                        num_comments=2,
                        semantic_similarity=0.82,
                        user_has_commented=False,
                        reply="Looks great",
                    )
                ],
            )
        )

        sent = client.post(f"/threads/{job['id']}/linkedin/{thread_id}/send")
        assert sent.status_code == 200
        assert sent.json()["user_has_commented"] is True

    assert attempts == [thread_id, thread_url]


def test_scenario_case_insensitive_job_name_uniqueness(tmp_path, monkeypatch) -> None:
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
            name="growth-path",
            social="reddit",
            topic="AI launch strategy",
        )

        first = client.post(
            "/jobs",
            json={
                "name": "Growth",
                "search_query_id": path["id"],
                "auth_id": "reddit-main",
            },
        )
        assert first.status_code == 200

        duplicate = client.post(
            "/jobs",
            json={
                "name": "growth",
                "search_query_id": path["id"],
                "auth_id": "reddit-main",
            },
        )
        assert duplicate.status_code == 409
        assert "already exists" in duplicate.json()["detail"].lower()
