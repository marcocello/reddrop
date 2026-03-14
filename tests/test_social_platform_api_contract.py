from __future__ import annotations

from datetime import datetime, timezone
import re

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.models import Conversation, SearchArtifact
from backend.services.search_storage import SearchStorage


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


def test_jobs_social_defaults_to_reddit_and_accepts_linkedin(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-auth',
                        'name': 'Reddit Auth',
                        'social': 'reddit',
                        'reddit_client_id': 'id',
                        'reddit_client_secret': 'secret',
                        'reddit_user_agent': 'Reddrop:v1.0',
                    },
                    {
                        'id': 'linkedin-auth',
                        'name': 'LinkedIn Auth',
                        'social': 'linkedin',
                        'linkedin_email': 'john@example.com',
                        'linkedin_password': 'pw',
                    },
                ]
            },
        )
        assert auths.status_code == 200

        default_job = client.post(
            '/jobs',
            json={
                'name': 'default-social',
                'topic': 'AI launch strategy',
                'auth_id': 'reddit-auth',
            },
        )
        assert default_job.status_code == 200
        assert default_job.json()['social'] == 'reddit'
        assert default_job.json()['auth_id'] == 'reddit-auth'

        linkedin_job = client.post(
            '/jobs',
            json={
                'name': 'linkedin-social',
                'topic': 'AI launch strategy',
                'social': 'linkedin',
                'auth_id': 'linkedin-auth',
            },
        )
        assert linkedin_job.status_code == 200
        assert linkedin_job.json()['social'] == 'linkedin'
        assert linkedin_job.json()['auth_id'] == 'linkedin-auth'


def test_search_jobs_require_auth_and_social_must_match_auth(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        missing_auth = client.post(
            '/jobs',
            json={
                'name': 'no-auth',
                'topic': 'AI launch strategy',
                'social': 'linkedin',
            },
        )
        assert missing_auth.status_code == 400
        assert 'auth_id' in missing_auth.json()['detail']

        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-auth',
                        'name': 'Reddit Auth',
                        'social': 'reddit',
                        'reddit_client_id': 'id',
                        'reddit_client_secret': 'secret',
                    }
                ]
            },
        )
        assert auths.status_code == 200

        mismatch = client.post(
            '/jobs',
            json={
                'name': 'bad-social-auth',
                'topic': 'AI launch strategy',
                'social': 'linkedin',
                'auth_id': 'reddit-auth',
            },
        )
        assert mismatch.status_code == 400
        assert 'social' in mismatch.json()['detail'].lower()


def test_search_endpoint_routes_linkedin_jobs_to_linkedin_service(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    class _FailIfRedditUsed:
        def discover_relevant_threads(self, **_kwargs):
            raise AssertionError('Reddit service should not be used for linkedin jobs')

    class _FakeLinkedinService:
        calls = 0

        def discover_relevant_threads(self, **_kwargs):
            _FakeLinkedinService.calls += 1
            return [
                {
                    'id': 'li-1',
                    'title': 'LinkedIn post',
                    'subreddit': 'linkedin',
                    'url': 'https://www.linkedin.com/feed/update/urn:li:activity:1',
                    'score': 7,
                    'num_comments': 3,
                    'created_utc': 0,
                    'selftext': 'body',
                    'semantic_similarity': 0.9,
                    'user_has_commented': False,
                }
            ]

    monkeypatch.setattr(api_main, 'RedditService', _FailIfRedditUsed)
    monkeypatch.setattr(api_main, 'LinkedinService', _FakeLinkedinService)

    with TestClient(api_main.app) as client:
        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'linkedin-auth',
                        'name': 'LinkedIn Auth',
                        'social': 'linkedin',
                        'linkedin_email': 'john@example.com',
                        'linkedin_password': 'pw',
                    }
                ]
            },
        )
        assert auths.status_code == 200

        created = client.post(
            '/jobs',
            json={
                'name': 'linkedin-search',
                'topic': 'B2B thought leadership',
                'social': 'linkedin',
                'auth_id': 'linkedin-auth',
            },
        )
        assert created.status_code == 200

        run = client.post('/search', json={'name': 'linkedin-search'})
        assert run.status_code == 200
        runs = run.json()['runs']
        assert len(runs) == 1
        assert runs[0]['conversations'][0]['thread_id'] == 'li-1'
        assert _FakeLinkedinService.calls == 1


def test_thread_send_routes_linkedin_jobs_to_linkedin_service(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    class _FailIfRedditUsed:
        def post_comment(self, **_kwargs):
            raise AssertionError('Reddit service should not be used for linkedin jobs')

    class _FakeLinkedinService:
        calls = 0

        def post_comment(self, thread_id: str, comment_text: str) -> bool:
            assert thread_id == 'li-thread-1'
            assert comment_text == 'Looks great'
            _FakeLinkedinService.calls += 1
            return True

    monkeypatch.setattr(api_main, 'RedditService', _FailIfRedditUsed)
    monkeypatch.setattr(api_main, 'LinkedinService', _FakeLinkedinService)

    with TestClient(api_main.app) as client:
        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'linkedin-auth',
                        'name': 'LinkedIn Auth',
                        'social': 'linkedin',
                        'linkedin_email': 'john@example.com',
                        'linkedin_password': 'pw',
                    }
                ]
            },
        )
        assert auths.status_code == 200

        created = client.post(
            '/jobs',
            json={
                'name': 'linkedin-send',
                'topic': 'B2B thought leadership',
                'social': 'linkedin',
                'auth_id': 'linkedin-auth',
            },
        )
        assert created.status_code == 200
        job = created.json()

        SearchStorage(api_main.DEFAULT_RUNS_DIR).save_search(
            SearchArtifact(
                id=job['id'],
                name=job['name'],
                topic=job['topic'],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                conversations=[
                    Conversation(
                        thread_id='li-thread-1',
                        title='LinkedIn post',
                        subreddit='linkedin',
                        url='https://www.linkedin.com/feed/update/urn:li:activity:1',
                        score=4,
                        num_comments=2,
                        semantic_similarity=0.82,
                        user_has_commented=False,
                        reply='Looks great',
                    )
                ],
            )
        )

        sent = client.post(f"/threads/{job['id']}/linkedin/li-thread-1/send")
        assert sent.status_code == 200
        assert sent.json()['user_has_commented'] is True
        assert _FakeLinkedinService.calls == 1


def test_auth_id_is_generated_as_uuid_and_linkedin_dir_is_derived(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        saved = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': '',
                        'name': 'LinkedIn Prospecting North America',
                        'social': 'linkedin',
                    }
                ]
            },
        )
        assert saved.status_code == 200
        payload = saved.json()
        assert len(payload['auths']) == 1
        item = payload['auths'][0]
        assert re.fullmatch(r'[0-9a-fA-F-]{36}', item['id'])
        assert item['linkedin_user_data_dir'].startswith('~/.reddrop/linkedin/')
        assert item['linkedin_user_data_dir'].endswith(item['id'][:8])
