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


def test_generate_search_plan_for_reddit_returns_topics_subreddits_queries(
    tmp_path, monkeypatch
) -> None:
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

    with TestClient(api_main.app) as client:
        response = client.post(
            "/search-queries/generate",
            json={
                "social": "reddit",
                "topic": "crm sales",
                "subreddit_limit": 5,
            },
        )
        assert response.status_code == 200
        payload = response.json()["search_query"]
        assert payload["social"] == "reddit"
        assert payload["plan"]["topics"] == ["crm", "sales"]
        assert payload["plan"]["subreddits"] == ["sales", "startups"]
        assert payload["plan"]["queries"] == ["crm sales", "sales pipeline"]


def test_generate_search_plan_for_linkedin_returns_query_list(
    tmp_path, monkeypatch
) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    llm_calls: dict[str, str] = {}

    def _fake_generate_response_sync(system_message: str, human_message: str) -> str:
        llm_calls["system"] = system_message
        llm_calls["human"] = human_message
        return '{"queries":["pluto AND pizza","pluto pizza operators"]}'

    monkeypatch.setattr(
        linkedin_service_module.llm_service,
        "generate_response_sync",
        _fake_generate_response_sync,
    )

    with TestClient(api_main.app) as client:
        response = client.post(
            "/search-queries/generate",
            json={
                "social": "linkedin",
                "topic": "pluto AND pizza",
            },
        )
        assert response.status_code == 200
        payload = response.json()["search_query"]
        assert payload["social"] == "linkedin"
        assert payload["plan"]["query"] == "pluto AND pizza"
        assert payload["plan"]["queries"][:2] == [
            "pluto AND pizza",
            "pluto pizza operators",
        ]
        assert len(payload["plan"]["queries"]) == 5

    assert "pluto AND pizza" in llm_calls.get("human", "")


def test_generate_search_plan_for_linkedin_filters_invalid_queries_and_supplements(
    tmp_path, monkeypatch
) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    def _fake_generate_response_sync(system_message: str, human_message: str) -> str:
        _ = (system_message, human_message)
        return """
        {
          "queries": [
            "https://www.linkedin.com/search/results/content/?keywords=crm",
            "site:linkedin.com crm sales",
            "crm sales"
          ]
        }
        """

    monkeypatch.setattr(
        linkedin_service_module.llm_service,
        "generate_response_sync",
        _fake_generate_response_sync,
    )

    with TestClient(api_main.app) as client:
        response = client.post(
            "/search-queries/generate",
            json={
                "social": "linkedin",
                "topic": "crm sales",
            },
        )
        assert response.status_code == 200
        queries = response.json()["search_query"]["plan"]["queries"]

    assert len(queries) == 5
    assert "crm sales" in queries
    assert all("linkedin.com" not in query.lower() for query in queries)
    assert all("site:" not in query.lower() for query in queries)


def test_linkedin_generate_search_plan_falls_back_to_topic_on_llm_error(
    monkeypatch,
) -> None:
    def _raise_llm_error(system_message: str, human_message: str) -> str:
        _ = (system_message, human_message)
        raise RuntimeError("llm offline")

    monkeypatch.setattr(
        linkedin_service_module.llm_service,
        "generate_response_sync",
        _raise_llm_error,
    )

    payload = linkedin_service_module.LinkedinService.generate_search_plan(
        topic="crm sales"
    )
    assert payload["query"] == "crm sales"
    assert len(payload["queries"]) == 5
    assert payload["queries"][0] == "crm sales"


def test_runtime_search_uses_locked_reddit_plan_values(tmp_path, monkeypatch) -> None:
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

        def generate_search_plan(self, *, topic: str, subreddit_limit: int):
            _ = (topic, subreddit_limit)
            return {
                "topics": ["crm"],
                "subreddits": ["sales"],
                "queries": ["crm sales"],
            }

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
                "name": "locked-reddit-plan-path",
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
                "name": "locked-reddit-plan",
                "social": "reddit",
                "auth_id": "reddit-auth",
                "search_query_id": path_id,
            },
        )
        assert created.status_code == 200
        payload = created.json()
        assert "search_plan" not in payload

        run = client.post("/search", json={"name": "locked-reddit-plan"})
        assert run.status_code == 200

    assert captured.get("search_plan") == {
        "topics": ["locked-topic"],
        "subreddits": ["sales"],
        "queries": ["locked query"],
        "query": None,
    }
