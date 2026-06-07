from __future__ import annotations

from datetime import datetime, timezone
import re
from urllib.parse import quote

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.models import AuthProfile, Conversation, SearchArtifact
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


def _create_frozen_search_path(
    client: TestClient,
    *,
    name: str,
    social: str,
    topic: str,
) -> dict:
    generated = client.post(
        '/search-queries/generate',
        json={
            'name': name,
            'social': social,
            'topic': topic,
        },
    )
    assert generated.status_code == 200
    path = generated.json()['search_query']
    saved = client.post(f"/search-queries/{path['id']}/save")
    assert saved.status_code == 200
    return saved.json()['search_query']


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
                        'reddit_username': 'alice',
                        'reddit_password': 'pw',
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
        reddit_path = _create_frozen_search_path(
            client,
            name='default-social-path',
            social='reddit',
            topic='AI launch strategy',
        )
        linkedin_path = _create_frozen_search_path(
            client,
            name='linkedin-social-path',
            social='linkedin',
            topic='AI launch strategy',
        )

        default_job = client.post(
            '/jobs',
            json={
                'name': 'default-social',
                'search_query_id': reddit_path['id'],
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
                'search_query_id': linkedin_path['id'],
                'auth_id': 'linkedin-auth',
            },
        )
        assert linkedin_job.status_code == 200
        assert linkedin_job.json()['social'] == 'linkedin'
        assert linkedin_job.json()['auth_id'] == 'linkedin-auth'


def test_search_jobs_require_auth_and_social_must_match_auth(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        linkedin_path = _create_frozen_search_path(
            client,
            name='no-auth-path',
            social='linkedin',
            topic='AI launch strategy',
        )
        missing_auth = client.post(
            '/jobs',
            json={
                'name': 'no-auth',
                'search_query_id': linkedin_path['id'],
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
                        'reddit_username': 'alice',
                        'reddit_password': 'pw',
                    }
                ]
            },
        )
        assert auths.status_code == 200
        mismatch_path = _create_frozen_search_path(
            client,
            name='bad-social-auth-path',
            social='linkedin',
            topic='AI launch strategy',
        )

        mismatch = client.post(
            '/jobs',
            json={
                'name': 'bad-social-auth',
                'search_query_id': mismatch_path['id'],
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
        path = _create_frozen_search_path(
            client,
            name='linkedin-search-path',
            social='linkedin',
            topic='B2B thought leadership',
        )

        created = client.post(
            '/jobs',
            json={
                'name': 'linkedin-search',
                'search_query_id': path['id'],
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
        path = _create_frozen_search_path(
            client,
            name='linkedin-send-path',
            social='linkedin',
            topic='B2B thought leadership',
        )

        created = client.post(
            '/jobs',
            json={
                'name': 'linkedin-send',
                'search_query_id': path['id'],
                'auth_id': 'linkedin-auth',
            },
        )
        assert created.status_code == 200
        job = created.json()

        SearchStorage(api_main.DEFAULT_RUNS_DIR).save_search(
            SearchArtifact(
                id=job['id'],
                name=job['name'],
                topic='B2B thought leadership',
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


def test_linkedin_thread_send_route_accepts_slash_thread_ids(tmp_path, monkeypatch) -> None:
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
            name='linkedin-send-url-path',
            social='linkedin',
            topic='B2B thought leadership',
        )

        created = client.post(
            "/jobs",
            json={
                "name": "linkedin-send-url-thread-id",
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


def test_linkedin_thread_send_retries_with_url_when_urn_post_fails(tmp_path, monkeypatch) -> None:
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
            name="linkedin-send-urn-fallback-path",
            social="linkedin",
            topic="B2B thought leadership",
        )

        created = client.post(
            "/jobs",
            json={
                "name": "linkedin-send-urn-fallback",
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


def test_linkedin_thread_url_normalizes_urn_and_host_prefixed_ids() -> None:
    assert (
        api_main.LinkedinService._thread_url("urn:li:activity:123")
        == "https://www.linkedin.com/feed/update/urn:li:activity:123/"
    )
    assert (
        api_main.LinkedinService._thread_url("www.linkedin.com/posts/jane_doe-activity-123/")
        == "https://www.linkedin.com/posts/jane_doe-activity-123/"
    )


def test_linkedin_search_job_passes_scroll_rounds_to_discovery(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    class _FakeLinkedinService:
        last_kwargs = None

        def discover_relevant_threads(self, **kwargs):
            _FakeLinkedinService.last_kwargs = kwargs
            return []

    monkeypatch.setattr(api_main, 'LinkedinService', _FakeLinkedinService)

    with TestClient(api_main.app) as client:
        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'linkedin-auth',
                        'social': 'linkedin',
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(
            client,
            name='linkedin-scroll-path',
            social='linkedin',
            topic='CRO forecast accuracy pipeline',
        )

        created = client.post(
            '/jobs',
            json={
                'name': 'linkedin-scroll',
                'search_query_id': path['id'],
                'auth_id': 'linkedin-auth',
                'threads_limit': 20,
                'scroll_rounds': 15,
                'time_filter': 'day',
                'sort_by': 'recency',
                'selector_wait_timeout_seconds': 24,
                'detail_wait_timeout_seconds': 7,
                'max_comment_expand_clicks': 35,
            },
        )
        assert created.status_code == 200
        assert created.json()['scroll_rounds'] == 15
        assert created.json()['time_filter'] == 'day'
        assert created.json()['sort_by'] == 'recency'
        assert created.json()['selector_wait_timeout_seconds'] == 24
        assert created.json()['detail_wait_timeout_seconds'] == 7
        assert created.json()['max_comment_expand_clicks'] == 35

        run = client.post('/search', json={'name': 'linkedin-scroll'})
        assert run.status_code == 200

    assert _FakeLinkedinService.last_kwargs is not None
    assert _FakeLinkedinService.last_kwargs['scroll_rounds'] == 15
    assert _FakeLinkedinService.last_kwargs['time_filter'] == 'day'
    assert _FakeLinkedinService.last_kwargs['sort_by'] == 'recency'
    assert _FakeLinkedinService.last_kwargs['selector_wait_timeout_seconds'] == 24
    assert _FakeLinkedinService.last_kwargs['detail_wait_timeout_seconds'] == 7
    assert _FakeLinkedinService.last_kwargs['max_comment_expand_clicks'] == 35


def test_threads_endpoint_exposes_linkedin_published_at(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    with TestClient(api_main.app) as client:
        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'linkedin-auth',
                        'name': 'LinkedIn Auth',
                        'social': 'linkedin',
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(
            client,
            name='linkedin-published-at-path',
            social='linkedin',
            topic='B2B thought leadership',
        )

        created = client.post(
            '/jobs',
            json={
                'name': 'linkedin-published-at',
                'search_query_id': path['id'],
                'auth_id': 'linkedin-auth',
            },
        )
        assert created.status_code == 200
        job = created.json()

        SearchStorage(api_main.DEFAULT_RUNS_DIR).save_search(
            SearchArtifact(
                id=job['id'],
                name=job['name'],
                topic='B2B thought leadership',
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
                        created_utc=1735732800,
                        published_at='2025-01-01T12:00:00Z',
                        semantic_similarity=0.82,
                        user_has_commented=False,
                        reply='',
                    )
                ],
            )
        )

        threads = client.get('/threads')
        assert threads.status_code == 200
        items = threads.json()['threads']
        assert len(items) == 1
        assert items[0]['published_at'] == '2025-01-01T12:00:00Z'


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
        assert item['linkedin_user_data_dir'].startswith('.reddrop/auth/linkedin/')
        assert not item['linkedin_user_data_dir'].startswith('~/.reddrop/')
        assert item['linkedin_user_data_dir'].endswith(item['id'][:8])


def test_auths_are_persisted_under_split_reddit_and_linkedin_paths(tmp_path, monkeypatch) -> None:
    home, _runs, _jobs = _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        saved = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-main',
                        'name': 'Reddit Main',
                        'social': 'reddit',
                        'reddit_client_id': 'id',
                        'reddit_client_secret': 'secret',
                        'reddit_username': 'alice',
                        'reddit_password': 'pw',
                    },
                    {
                        'id': 'linkedin-main',
                        'name': 'LinkedIn Main',
                        'social': 'linkedin',
                    },
                ]
            },
        )
        assert saved.status_code == 200

    reddit_dir = home / 'auth' / 'reddit'
    linkedin_dir = home / 'auth' / 'linkedin'
    assert reddit_dir.exists()
    assert linkedin_dir.exists()
    assert (reddit_dir / 'auths.json').exists()
    assert (linkedin_dir / 'auths.json').exists()


def test_auth_upsert_rejects_incomplete_reddit_credentials(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)

    with TestClient(api_main.app) as client:
        missing_client = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-missing-client',
                        'social': 'reddit',
                        'reddit_username': 'alice',
                        'reddit_password': 'pw',
                    }
                ]
            },
        )
        assert missing_client.status_code == 400
        assert "client_id" in missing_client.json()['detail']

        missing_user = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-missing-user',
                        'social': 'reddit',
                        'reddit_client_id': 'cid',
                        'reddit_client_secret': 'csecret',
                        'reddit_password': 'pw',
                    }
                ]
            },
        )
        assert missing_user.status_code == 400
        assert "username" in missing_user.json()['detail']

def test_auth_catalog_reports_health_status(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    calls: list[tuple[str, object]] = []

    def fake_validate_reddit_credentials(*, client_id: str, client_secret: str, user_agent: str, username: str, password: str) -> bool:
        calls.append(
            (
                "reddit",
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "user_agent": user_agent,
                    "username": username,
                    "password": password,
                },
            )
        )
        return username == "alice"

    def fake_validate_linkedin_saved_session(*, user_data_dir: str) -> bool:
        calls.append(("linkedin", user_data_dir))
        return "linkedin" in user_data_dir

    monkeypatch.setenv("REDDIT_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "env-agent")
    monkeypatch.setattr(api_main.RedditService, "validate_credentials", staticmethod(fake_validate_reddit_credentials))
    monkeypatch.setattr(api_main.LinkedinService, "validate_saved_session", staticmethod(fake_validate_linkedin_saved_session))

    with TestClient(api_main.app) as client:
        saved = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-ok',
                        'social': 'reddit',
                        'reddit_username': 'alice',
                        'reddit_password': 'pw',
                    },
                    {
                        'id': 'reddit-invalid',
                        'social': 'reddit',
                        'reddit_username': 'bob',
                        'reddit_password': 'pw',
                    },
                    {
                        'id': 'linkedin-ok',
                        'social': 'linkedin',
                    },
                ]
            },
        )
        assert saved.status_code == 200
        payload = saved.json()
        by_id = {item['id']: item for item in payload['auths']}
        assert by_id['reddit-ok']['status'] == 'ok'
        assert by_id['reddit-invalid']['status'] == 'invalid'
        assert by_id['linkedin-ok']['status'] == 'ok'

    assert any(item[0] == "reddit" for item in calls)
    assert any(item[0] == "linkedin" for item in calls)


def test_auth_validation_endpoints_check_reddit_and_linkedin(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    calls: list[tuple[str, object]] = []

    def fake_validate_reddit_credentials(*, client_id: str, client_secret: str, user_agent: str, username: str, password: str) -> bool:
        calls.append(
            (
                "reddit",
                {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "user_agent": user_agent,
                    "username": username,
                    "password": password,
                },
            )
        )
        return username == "alice"

    def fake_validate_linkedin_saved_session(*, user_data_dir: str) -> bool:
        calls.append(("linkedin", user_data_dir))
        return "linkedin" in user_data_dir

    monkeypatch.setenv("REDDIT_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "env-agent")
    monkeypatch.setattr(api_main.RedditService, "validate_credentials", staticmethod(fake_validate_reddit_credentials))
    monkeypatch.setattr(api_main.LinkedinService, "validate_saved_session", staticmethod(fake_validate_linkedin_saved_session))

    with TestClient(api_main.app) as client:
        saved = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-ok',
                        'social': 'reddit',
                        'reddit_username': 'alice',
                        'reddit_password': 'pw',
                    },
                    {
                        'id': 'linkedin-ok',
                        'social': 'linkedin',
                    },
                ]
            },
        )
        assert saved.status_code == 200

        single = client.post('/auths/reddit-ok/validate')
        assert single.status_code == 200
        assert single.json() == {
            'auth_id': 'reddit-ok',
            'social': 'reddit',
            'status': 'ok',
        }

        all_auths = client.post('/auths/validate')
        assert all_auths.status_code == 200
        payload = all_auths.json()
        by_id = {item['auth_id']: item for item in payload['results']}
        assert by_id['reddit-ok']['status'] == 'ok'
        assert by_id['linkedin-ok']['status'] == 'ok'

    assert any(item[0] == "reddit" for item in calls)
    assert any(item[0] == "linkedin" for item in calls)


def test_auth_upsert_revalidates_existing_auths_on_every_save(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    reddit_calls: list[str] = []
    linkedin_calls: list[str] = []

    def fake_validate_reddit_credentials(*, client_id: str, client_secret: str, user_agent: str, username: str, password: str) -> bool:
        _ = client_id
        _ = client_secret
        _ = user_agent
        _ = password
        reddit_calls.append(username)
        return True

    def fake_validate_linkedin_saved_session(*, user_data_dir: str) -> bool:
        linkedin_calls.append(user_data_dir)
        return True

    monkeypatch.setenv("REDDIT_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "env-agent")
    monkeypatch.setattr(api_main.RedditService, "validate_credentials", staticmethod(fake_validate_reddit_credentials))
    monkeypatch.setattr(api_main.LinkedinService, "validate_saved_session", staticmethod(fake_validate_linkedin_saved_session))

    payload = {
        'auths': [
            {
                'id': 'reddit-main',
                'social': 'reddit',
                'reddit_username': 'alice',
                'reddit_password': 'pw',
            },
            {
                'id': 'linkedin-main',
                'social': 'linkedin',
            },
        ]
    }

    with TestClient(api_main.app) as client:
        first = client.put('/auths', json=payload)
        assert first.status_code == 200
        second = client.put('/auths', json=payload)
        assert second.status_code == 200

    assert len(reddit_calls) >= 2
    assert reddit_calls.count("alice") >= 2
    assert len(linkedin_calls) >= 2


def test_auth_catalog_reads_linkedin_username_from_user_data_dir(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    username_calls: list[str] = []

    def fake_validate_linkedin_saved_session(*, user_data_dir: str) -> bool:
        return True

    def fake_resolve_linkedin_username(*, user_data_dir: str) -> str:
        username_calls.append(user_data_dir)
        return "example-user"

    monkeypatch.setattr(api_main.LinkedinService, "validate_saved_session", staticmethod(fake_validate_linkedin_saved_session))
    monkeypatch.setattr(
        api_main.LinkedinService,
        "resolve_saved_session_username",
        staticmethod(fake_resolve_linkedin_username),
        raising=False,
    )

    with TestClient(api_main.app) as client:
        saved = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'linkedin-main',
                        'social': 'linkedin',
                    },
                ]
            },
        )
        assert saved.status_code == 200
        payload = saved.json()
        assert payload['auths'][0]['linkedin_username'] == 'example-user'

    assert len(username_calls) >= 1


def test_auth_catalog_does_not_validate_or_resolve_linkedin_username(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    resolver_calls = 0

    def fail_reddit_validation(*, client_id: str, client_secret: str, user_agent: str, username: str, password: str) -> bool:
        _ = client_id
        _ = client_secret
        _ = user_agent
        _ = username
        _ = password
        raise AssertionError("GET /auths must not call Reddit credential validation")

    def fail_linkedin_validation(*, user_data_dir: str) -> bool:
        _ = user_data_dir
        raise AssertionError("GET /auths must not call LinkedIn session validation")

    def fake_resolve_linkedin_username(*, user_data_dir: str) -> str:
        nonlocal resolver_calls
        _ = user_data_dir
        resolver_calls += 1
        return "resolved-via-callback"

    monkeypatch.setattr(api_main.RedditService, "validate_credentials", staticmethod(fail_reddit_validation))
    monkeypatch.setattr(api_main.LinkedinService, "validate_saved_session", staticmethod(fail_linkedin_validation))
    monkeypatch.setattr(
        api_main.LinkedinService,
        "resolve_saved_session_username",
        staticmethod(fake_resolve_linkedin_username),
        raising=False,
    )

    store = api_main._auth_store()
    store.save_auths(
        [
            AuthProfile(
                id="reddit-main",
                social="reddit",
                status="ok",
                reddit_client_id="cid",
                reddit_client_secret="csecret",
                reddit_user_agent="agent",
                reddit_username="alice",
                reddit_password="pw",
            ),
            AuthProfile(
                id="linkedin-main",
                social="linkedin",
                status="invalid",
                linkedin_username="stored-linkedin-user",
                linkedin_user_data_dir=".reddrop/auth/linkedin/linkedin-main-linkedin",
            ),
        ]
    )

    with TestClient(api_main.app) as client:
        response = client.get('/auths')
        assert response.status_code == 200
        payload = response.json()
        by_id = {item['id']: item for item in payload['auths']}
        assert by_id['reddit-main']['status'] == 'ok'
        assert by_id['linkedin-main']['status'] == 'invalid'
        assert by_id['linkedin-main']['linkedin_username'] == 'stored-linkedin-user'
    assert resolver_calls == 0


def test_auth_validation_does_not_log_start_or_result_markers(tmp_path, monkeypatch, caplog) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    def fake_validate_linkedin_saved_session(*, user_data_dir: str) -> bool:
        _ = user_data_dir
        return True

    monkeypatch.setattr(api_main.LinkedinService, "validate_saved_session", staticmethod(fake_validate_linkedin_saved_session))

    payload = {
        'auths': [
            {
                'id': 'linkedin-main',
                'social': 'linkedin',
            },
        ]
    }

    with TestClient(api_main.app) as client:
        with caplog.at_level("INFO"):
            first = client.put('/auths', json=payload)
            assert first.status_code == 200
            second = client.put('/auths', json=payload)
            assert second.status_code == 200

    messages = [record.getMessage() for record in caplog.records]
    assert not any("Auth validation start:" in message for message in messages)
    assert not any("Auth validation result:" in message for message in messages)
