from __future__ import annotations

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.services import linkedin_service as linkedin_service_module


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


def test_scenario_reddit_and_linkedin_search_plan_generation(monkeypatch, tmp_path) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    class _FakeRedditService:
        @staticmethod
        def validate_credentials(**_kwargs):
            return True

        def __init__(self, auth=None):
            _ = auth

        def generate_search_plan(self, *, topic: str, subreddit_limit: int):
            _ = (topic, subreddit_limit)
            return {
                "topics": ["crm", "sales"],
                "subreddits": ["sales", "startups"],
                "queries": ["crm sales", "sales pipeline"],
            }

    monkeypatch.setattr(api_main, "RedditService", _FakeRedditService)
    monkeypatch.setattr(
        linkedin_service_module.llm_service,
        "generate_response_sync",
        lambda system_message, human_message: '{"query":"pluto AND pizza"}',
    )

    with TestClient(api_main.app) as client:
        reddit_plan = client.post(
            "/search-queries/generate",
            json={"social": "reddit", "topic": "crm sales"},
        )
        assert reddit_plan.status_code == 200
        assert reddit_plan.json()["search_query"]["plan"]["subreddits"] == ["sales", "startups"]

        linkedin_plan = client.post(
            "/search-queries/generate",
            json={"social": "linkedin", "topic": "pluto AND pizza"},
        )
        assert linkedin_plan.status_code == 200
        assert linkedin_plan.json()["search_query"]["plan"]["query"] == "pluto AND pizza"


def test_scenario_create_search_job_stores_locked_plan_and_cannot_be_edited(monkeypatch, tmp_path) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
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

        path_created = client.post(
            "/search-queries",
            json={
                "name": "plan-locked-job-path",
                "social": "reddit",
                "topic": "crm sales",
                "plan": {
                    "topics": ["crm"],
                    "subreddits": ["sales"],
                    "queries": ["crm sales"],
                },
            },
        )
        assert path_created.status_code == 200
        path_id = path_created.json()["search_query"]["id"]
        saved = client.post(f"/search-queries/{path_id}/save")
        assert saved.status_code == 200

        created = client.post(
            "/jobs",
            json={
                "name": "plan-locked-job",
                "social": "reddit",
                "auth_id": "reddit-auth",
                "search_query_id": path_id,
            },
        )
        assert created.status_code == 200
        payload = created.json()
        assert payload["search_query_id"] == path_id
        assert payload["search_query_version"] == 1
        assert "search_plan" not in payload

        update = client.put(
            f"/jobs/{payload['id']}",
            json={
                "name": "plan-locked-job",
                "social": "reddit",
                "auth_id": "reddit-auth",
            },
        )
        assert update.status_code == 405


def test_scenario_runtime_search_uses_locked_plan(monkeypatch, tmp_path) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    class _FakeRedditService:
        @staticmethod
        def validate_credentials(**_kwargs):
            return True

        def __init__(self, auth=None):
            _ = auth

        def discover_relevant_threads(self, **kwargs):
            captured.update(kwargs)
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

        path_created = client.post(
            "/search-queries",
            json={
                "name": "runtime-locked-plan-path",
                "social": "reddit",
                "topic": "crm sales",
                "plan": {
                    "topics": ["locked-topic"],
                    "subreddits": ["sales"],
                    "queries": ["locked query"],
                },
            },
        )
        assert path_created.status_code == 200
        path_id = path_created.json()["search_query"]["id"]
        saved = client.post(f"/search-queries/{path_id}/save")
        assert saved.status_code == 200

        created = client.post(
            "/jobs",
            json={
                "name": "runtime-locked-plan",
                "social": "reddit",
                "auth_id": "reddit-auth",
                "search_query_id": path_id,
            },
        )
        assert created.status_code == 200

        run = client.post("/search", json={"name": "runtime-locked-plan"})
        assert run.status_code == 200

    assert captured["search_plan"] == {
        "topics": ["locked-topic"],
        "subreddits": ["sales"],
        "queries": ["locked query"],
        "query": None,
    }
