from __future__ import annotations

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main


def _setup_isolated_runtime(tmp_path, monkeypatch):
    home = tmp_path / '.reddrop'
    runs = home / 'runs'
    jobs = home / 'jobs_config.json'
    monkeypatch.setattr(api_main, 'DEFAULT_HOME', home)
    monkeypatch.setattr(api_main, 'DEFAULT_RUNS_DIR', runs)
    monkeypatch.setattr(api_main, 'DEFAULT_JOBS_CONFIG', jobs)
    try:
        api_main.runtime_manager.shutdown()
    except SchedulerNotRunningError:
        pass
    monkeypatch.setattr(api_main, 'runtime_manager', api_main.JobRuntimeManager())
    return home, runs, jobs


def _seed_reddit_auth(client: TestClient) -> None:
    auths = client.put(
        '/auths',
        json={
            'auths': [
                {
                    'id': 'reddit-auth',
                    'social': 'reddit',
                    'reddit_client_id': 'id',
                    'reddit_client_secret': 'secret',
                    'reddit_username': 'alice',
                    'reddit_password': 'pw',
                }
            ]
        },
    )
    assert auths.status_code == 200


def test_scenario_generate_freeze_and_bind_search_path(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    captured: dict[str, object] = {}

    class _FakeRedditService:
        @staticmethod
        def validate_credentials(**_kwargs):
            return True

        def __init__(self, auth=None):
            _ = auth

        def generate_search_plan(self, *, topic: str, subreddit_limit: int):
            _ = (topic, subreddit_limit)
            return {
                'topics': ['crm'],
                'subreddits': ['sales'],
                'queries': ['crm sales'],
            }

        def discover_relevant_threads(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(api_main, 'RedditService', _FakeRedditService)

    with TestClient(api_main.app) as client:
        _seed_reddit_auth(client)

        generated = client.post(
            '/search-queries/generate',
            json={
                'name': 'crm-path',
                'social': 'reddit',
                'topic': 'crm sales',
            },
        )
        assert generated.status_code == 200
        search_query = generated.json()['search_query']
        assert search_query['status'] == 'draft'
        assert 'auth_id' not in search_query

        not_allowed = client.post(
            '/jobs',
            json={
                'name': 'draft-path-job',
                'job_type': 'search',
                'social': 'reddit',
                'auth_id': 'reddit-auth',
                'search_query_id': search_query['id'],
            },
        )
        assert not_allowed.status_code == 400

        saved = client.post(f"/search-queries/{search_query['id']}/save")
        assert saved.status_code == 200
        assert saved.json()['search_query']['status'] == 'saved'
        assert saved.json()['search_query']['version'] == 1

        created = client.post(
            '/jobs',
            json={
                'name': 'saved-path-job',
                'job_type': 'search',
                'social': 'reddit',
                'auth_id': 'reddit-auth',
                'search_query_id': search_query['id'],
            },
        )
        assert created.status_code == 200
        job = created.json()
        assert job['search_query_id'] == search_query['id']
        assert job['search_query_version'] == 1

        run = client.post('/search', json={'name': 'saved-path-job'})
        assert run.status_code == 200

    assert captured['search_plan'] == {
        'topics': ['crm'],
        'subreddits': ['sales'],
        'queries': ['crm sales'],
        'query': None,
    }
