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


def test_search_query_generate_save_and_create_job(tmp_path, monkeypatch) -> None:
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
                'topics': ['crm'],
                'subreddits': ['sales'],
                'queries': ['crm sales'],
            }

    monkeypatch.setattr(api_main, 'RedditService', _FakeRedditService)

    with TestClient(api_main.app) as client:
        _seed_reddit_auth(client)

        generated = client.post(
            '/search-queries/generate',
            json={
                'name': 'crm-path',
                'social': 'reddit',
                'topic': 'crm sales',
                'subreddit_limit': 5,
            },
        )
        assert generated.status_code == 200
        search_query = generated.json()['search_query']
        assert search_query['status'] == 'draft'
        assert search_query['plan']['subreddits'] == ['sales']
        assert 'auth_id' not in search_query

        saved = client.post(f"/search-queries/{search_query['id']}/save")
        assert saved.status_code == 200
        saved_payload = saved.json()['search_query']
        assert saved_payload['status'] == 'saved'
        assert saved_payload['version'] == 1

        created = client.post(
            '/jobs',
            json={
                'name': 'search-by-path',
                'job_type': 'search',
                'search_query_id': search_query['id'],
                'auth_id': 'reddit-auth',
                'social': 'reddit',
            },
        )
        assert created.status_code == 200
        job = created.json()
        assert job['search_query_id'] == search_query['id']
        assert job['search_query_version'] == 1
        assert 'topic' not in job
        assert 'search_plan' not in job


def test_search_job_requires_search_query_id(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    with TestClient(api_main.app) as client:
        _seed_reddit_auth(client)

        created = client.post(
            '/jobs',
            json={
                'name': 'search-missing-path',
                'job_type': 'search',
                'social': 'reddit',
                'auth_id': 'reddit-auth',
            },
        )
        assert created.status_code == 400
        assert 'search_query_id' in created.json()['detail'].lower()


def test_search_job_rejects_non_frozen_search_path(tmp_path, monkeypatch) -> None:
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
                'topics': ['crm'],
                'subreddits': ['sales'],
                'queries': ['crm sales'],
            }

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
        search_query_id = generated.json()['search_query']['id']

        created = client.post(
            '/jobs',
            json={
                'name': 'search-by-draft-path',
                'job_type': 'search',
                'search_query_id': search_query_id,
                'auth_id': 'reddit-auth',
                'social': 'reddit',
            },
        )
        assert created.status_code == 400
        assert 'saved' in created.json()['detail'].lower()


def test_search_path_job_still_requires_auth(tmp_path, monkeypatch) -> None:
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
                'topics': ['crm'],
                'subreddits': ['sales'],
                'queries': ['crm sales'],
            }

    monkeypatch.setattr(api_main, 'RedditService', _FakeRedditService)

    with TestClient(api_main.app) as client:
        generated = client.post(
            '/search-queries/generate',
            json={
                'name': 'crm-path',
                'social': 'reddit',
                'topic': 'crm sales',
            },
        )
        assert generated.status_code == 200
        search_query_id = generated.json()['search_query']['id']
        saved = client.post(f'/search-queries/{search_query_id}/save')
        assert saved.status_code == 200

        created = client.post(
            '/jobs',
            json={
                'name': 'search-by-path-no-auth',
                'job_type': 'search',
                'search_query_id': search_query_id,
                'social': 'reddit',
            },
        )
        assert created.status_code == 400
        assert 'auth_id' in created.json()['detail'].lower()


def test_search_path_can_be_deleted_even_when_frozen(tmp_path, monkeypatch) -> None:
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
                'topics': ['crm'],
                'subreddits': ['sales'],
                'queries': ['crm sales'],
            }

    monkeypatch.setattr(api_main, 'RedditService', _FakeRedditService)

    with TestClient(api_main.app) as client:
        generated = client.post(
            '/search-queries/generate',
            json={
                'name': 'crm-path',
                'social': 'reddit',
                'topic': 'crm sales',
            },
        )
        assert generated.status_code == 200
        search_query_id = generated.json()['search_query']['id']

        saved = client.post(f'/search-queries/{search_query_id}/save')
        assert saved.status_code == 200

        deleted = client.delete(f'/search-queries/{search_query_id}')
        assert deleted.status_code == 200
        assert deleted.json()['deleted'] is True

        listed = client.get('/search-queries')
        assert listed.status_code == 200
        assert listed.json()['search_queries'] == []
