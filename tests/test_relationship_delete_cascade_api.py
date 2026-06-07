from __future__ import annotations

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main


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


def _seed_reddit_auth(client: TestClient, auth_id: str = "reddit-main") -> None:
    auths = client.put(
        "/auths",
        json={
            "auths": [
                {
                    "id": auth_id,
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


def _create_frozen_search_path(client: TestClient, *, name: str, social: str, topic: str) -> dict:
    generated = client.post(
        "/search-queries/generate",
        json={"name": name, "social": social, "topic": topic},
    )
    assert generated.status_code == 200
    path = generated.json()["search_query"]
    saved = client.post(f"/search-queries/{path['id']}/save")
    assert saved.status_code == 200
    return saved.json()["search_query"]


def _seed_search_and_reply(client: TestClient, *, auth_id: str = "reddit-main") -> tuple[dict, dict]:
    path = _create_frozen_search_path(client, name="source-search-path", social="reddit", topic="crm sales")
    search = client.post(
        "/jobs",
        json={
            "name": "source-search",
            "job_type": "search",
            "search_query_id": path["id"],
            "auth_id": auth_id,
        },
    )
    assert search.status_code == 200
    search_job = search.json()

    reply = client.post(
        "/jobs",
        json={
            "name": "source-reply",
            "job_type": "reply",
            "source_job_id": search_job["id"],
            "personas": ["seller"],
        },
    )
    assert reply.status_code == 200
    return search_job, reply.json()


def test_delete_search_job_requires_cascade_when_reply_jobs_are_linked(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        _seed_reddit_auth(client)
        search_job, _reply_job = _seed_search_and_reply(client)

        blocked = client.delete(f"/jobs/{search_job['id']}")
        assert blocked.status_code == 409
        assert "cascade=true" in str(blocked.json()["detail"]).lower()

        deleted = client.delete(f"/jobs/{search_job['id']}?cascade=true")
        assert deleted.status_code == 200
        payload = deleted.json()
        assert payload["deleted"] is True
        assert payload["cascade"] is True
        assert len(payload["deleted_jobs"]) == 2

        jobs = client.get("/jobs")
        assert jobs.status_code == 200
        assert jobs.json()["jobs"] == []


def test_delete_persona_requires_cascade_when_used_by_reply_jobs(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        _seed_reddit_auth(client)
        search_job, reply_job = _seed_search_and_reply(client)
        personas = client.put(
            "/personas",
            json={"personas": [{"name": "seller", "objective": "o. direct tone."}]},
        )
        assert personas.status_code == 200

        blocked = client.delete("/personas/seller")
        assert blocked.status_code == 409
        assert "cascade=true" in str(blocked.json()["detail"]).lower()

        deleted = client.delete("/personas/seller?cascade=true")
        assert deleted.status_code == 200
        payload = deleted.json()
        assert payload["deleted"] is True
        assert payload["cascade"] is True
        assert [item["id"] for item in payload["deleted_jobs"]] == [reply_job["id"]]

        jobs = client.get("/jobs")
        assert jobs.status_code == 200
        remaining_names = {item["name"] for item in jobs.json()["jobs"]}
        assert search_job["name"] in remaining_names
        assert reply_job["name"] not in remaining_names

        listed_personas = client.get("/personas")
        assert listed_personas.status_code == 200
        assert listed_personas.json()["personas"] == []


def test_delete_search_path_requires_cascade_when_jobs_use_it(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    class _FakeRedditService:
        @staticmethod
        def validate_credentials(**_kwargs):
            return True

        def __init__(self, auth=None):
            _ = auth

        def generate_search_plan(self, *, topic: str, subreddit_limit: int):
            _ = (topic, subreddit_limit)
            return {"topics": ["crm"], "subreddits": ["sales"], "queries": ["crm sales"]}

    monkeypatch.setattr(api_main, "RedditService", _FakeRedditService)

    with TestClient(api_main.app) as client:
        _seed_reddit_auth(client)
        generated = client.post(
            "/search-queries/generate",
            json={"name": "crm-path", "social": "reddit", "topic": "crm sales"},
        )
        assert generated.status_code == 200
        path_id = generated.json()["search_query"]["id"]
        saved = client.post(f"/search-queries/{path_id}/save")
        assert saved.status_code == 200

        search = client.post(
            "/jobs",
            json={
                "name": "path-search",
                "job_type": "search",
                "social": "reddit",
                "auth_id": "reddit-main",
                "search_query_id": path_id,
            },
        )
        assert search.status_code == 200
        search_job = search.json()

        reply = client.post(
            "/jobs",
            json={
                "name": "path-reply",
                "job_type": "reply",
                "source_job_id": search_job["id"],
                "personas": ["seller"],
            },
        )
        assert reply.status_code == 200

        blocked = client.delete(f"/search-queries/{path_id}")
        assert blocked.status_code == 409
        assert "cascade=true" in str(blocked.json()["detail"]).lower()

        deleted = client.delete(f"/search-queries/{path_id}?cascade=true")
        assert deleted.status_code == 200
        payload = deleted.json()
        assert payload["deleted"] is True
        assert payload["cascade"] is True
        assert len(payload["deleted_jobs"]) == 2

        listed = client.get("/search-queries")
        assert listed.status_code == 200
        assert listed.json()["search_queries"] == []

        jobs = client.get("/jobs")
        assert jobs.status_code == 200
        assert jobs.json()["jobs"] == []


def test_delete_auth_requires_cascade_when_jobs_use_it(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        _seed_reddit_auth(client, auth_id="reddit-auth")
        _seed_search_and_reply(client, auth_id="reddit-auth")

        blocked = client.delete("/auths/reddit-auth")
        assert blocked.status_code == 409
        assert "cascade=true" in str(blocked.json()["detail"]).lower()

        deleted = client.delete("/auths/reddit-auth?cascade=true")
        assert deleted.status_code == 200
        payload = deleted.json()
        assert payload["deleted"] is True
        assert payload["cascade"] is True
        assert len(payload["deleted_jobs"]) == 2

        auths = client.get("/auths")
        assert auths.status_code == 200
        assert auths.json()["auths"] == []
