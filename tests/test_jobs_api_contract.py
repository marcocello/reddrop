from __future__ import annotations

from datetime import datetime, timezone
import json

from fastapi.testclient import TestClient
from apscheduler.schedulers import SchedulerNotRunningError

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


def _create_frozen_search_path(
    client: TestClient,
    *,
    name: str,
    social: str,
    topic: str,
    plan: dict | None = None,
) -> dict:
    if plan is None:
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
    else:
        created = client.post(
            '/search-queries',
            json={
                'name': name,
                'social': social,
                'topic': topic,
                'plan': plan,
            },
        )
        assert created.status_code == 200
        path = created.json()['search_query']

    saved = client.post(f"/search-queries/{path['id']}/save")
    assert saved.status_code == 200
    return saved.json()['search_query']


def test_jobs_are_immutable_after_creation(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        auths = client.put(
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
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(client, name='growth-path', social='reddit', topic='AI launch strategy')
        created = client.post(
            '/jobs',
            json={
                'name': 'growth',
                'search_query_id': path['id'],
                'auth_id': 'reddit-main',
            },
        )
        assert created.status_code == 200
        job_id = created.json()['id']

        update = client.put(
            f'/jobs/{job_id}',
            json={
                'name': 'growth',
                'topic': 'changed',
                'active': True,
                'time_filter': 'week',
                'subreddit_limit': 5,
                'threads_limit': 10,
                'replies_per_iteration': 3,
                'max_runtime_minutes': 60,
                'personas': [],
            },
        )

        assert update.status_code == 405


def test_create_job_rejects_case_insensitive_duplicate_names(tmp_path, monkeypatch) -> None:
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
        path = _create_frozen_search_path(client, name='growth-path', social='reddit', topic='AI launch strategy')

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


def test_jobs_list_includes_summary_counters(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        auths = client.put(
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
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(client, name='growth-path', social='reddit', topic='AI launch strategy')
        created = client.post('/jobs', json={'name': 'growth', 'search_query_id': path['id'], 'auth_id': 'reddit-main'})
        assert created.status_code == 200
        job = created.json()

        SearchStorage(api_main.DEFAULT_RUNS_DIR).save_search(
            SearchArtifact(
                id=job['id'],
                name=job['name'],
                topic='AI launch strategy',
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                conversations=[
                    Conversation(
                        thread_id='t1',
                        title='Thread 1',
                        subreddit='sales',
                        url='https://reddit.com/r/sales/comments/t1',
                        score=10,
                        num_comments=3,
                        semantic_similarity=0.7,
                        user_has_commented=True,
                        reply='sent reply',
                    ),
                    Conversation(
                        thread_id='t2',
                        title='Thread 2',
                        subreddit='sales',
                        url='https://reddit.com/r/sales/comments/t2',
                        score=5,
                        num_comments=1,
                        semantic_similarity=0.6,
                        user_has_commented=False,
                        reply='draft reply',
                    ),
                ],
            )
        )

        payload = client.get('/jobs')
        assert payload.status_code == 200
        jobs = payload.json()['jobs']
        assert len(jobs) == 1
        item = jobs[0]

        assert 'last_run_status' in item
        assert 'successful_runs' in item
        assert item['threads_found_total'] == 2
        assert item['threads_replied_total'] == 1


def test_replying_jobs_require_source_search_job_and_persona(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        auths = client.put(
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
                    }
                ]
            },
        )
        assert auths.status_code == 200

        missing_source = client.post(
            '/jobs',
            json={
                'name': 'reply-only',
                'job_type': 'reply',
                'personas': ['seller'],
            },
        )
        assert missing_source.status_code == 400
        assert 'source_job_id' in missing_source.json()['detail']

        missing_persona = client.post(
            '/jobs',
            json={
                'name': 'reply-only-2',
                'job_type': 'reply',
                'source_job_id': 'does-not-exist',
                'personas': [],
            },
        )
        assert missing_persona.status_code == 400
        assert 'persona' in missing_persona.json()['detail'].lower()
        path = _create_frozen_search_path(client, name='search-only-path', social='reddit', topic='AI launch strategy')

        search_job = client.post(
            '/jobs',
            json={
                'name': 'search-only',
                'job_type': 'search',
                'search_query_id': path['id'],
                'auth_id': 'reddit-main',
            },
        )
        assert search_job.status_code == 200
        search_job_id = search_job.json()['id']

        replying_job = client.post(
            '/jobs',
            json={
                'name': 'reply-worker',
                'job_type': 'reply',
                'source_job_id': search_job_id,
                'personas': ['seller'],
            },
        )
        assert replying_job.status_code == 200
        payload = replying_job.json()
        assert payload['job_type'] == 'reply'
        assert payload['source_job_id'] == search_job_id
        assert payload['auth_id'] == 'reddit-main'
        assert 'topic' not in payload
        assert 'time_filter' not in payload
        assert 'subreddit_limit' not in payload
        assert 'threads_limit' not in payload

        custom_threshold = client.post(
            '/jobs',
            json={
                'name': 'reply-worker-custom-threshold',
                'job_type': 'reply',
                'source_job_id': search_job_id,
                'personas': ['seller'],
                'min_similarity_score': 0.72,
            },
        )
        assert custom_threshold.status_code == 200
        custom_payload = custom_threshold.json()
        assert custom_payload['min_similarity_score'] == 0.72


def test_job_payloads_only_include_fields_for_their_type(tmp_path, monkeypatch) -> None:
    _home, _runs, jobs_config = _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        auths = client.put(
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
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(client, name='search-only-split-path', social='reddit', topic='AI launch strategy')

        search_created = client.post(
            '/jobs',
            json={
                'name': 'search-only-split',
                'job_type': 'search',
                'search_query_id': path['id'],
                'auth_id': 'reddit-main',
            },
        )
        assert search_created.status_code == 200
        search_payload = search_created.json()
        assert 'topic' not in search_payload
        assert 'time_filter' in search_payload
        assert 'subreddit_limit' in search_payload
        assert 'threads_limit' in search_payload
        assert 'min_similarity_score' not in search_payload
        assert 'replies_per_iteration' not in search_payload
        assert 'personas' not in search_payload

        reply_created = client.post(
            '/jobs',
            json={
                'name': 'reply-only-split',
                'job_type': 'reply',
                'source_job_id': search_payload['id'],
                'personas': ['seller'],
                'min_similarity_score': 0.55,
                'replies_per_iteration': 4,
            },
        )
        assert reply_created.status_code == 200
        reply_payload = reply_created.json()
        assert 'min_similarity_score' in reply_payload
        assert 'replies_per_iteration' in reply_payload
        assert 'personas' in reply_payload
        assert 'topic' not in reply_payload
        assert 'time_filter' not in reply_payload
        assert 'subreddit_limit' not in reply_payload
        assert 'threads_limit' not in reply_payload

    persisted = json.loads(jobs_config.read_text(encoding='utf-8'))
    assert isinstance(persisted, dict)
    jobs = persisted.get('jobs')
    assert isinstance(jobs, list)
    by_name = {str(item.get('name')): item for item in jobs if isinstance(item, dict)}

    search_record = by_name['search-only-split']
    assert 'topic' not in search_record
    assert 'time_filter' in search_record
    assert 'subreddit_limit' in search_record
    assert 'threads_limit' in search_record
    assert 'source_job_id' not in search_record
    assert 'min_similarity_score' not in search_record
    assert 'replies_per_iteration' not in search_record
    assert 'personas' not in search_record

    reply_record = by_name['reply-only-split']
    assert 'source_job_id' in reply_record
    assert 'min_similarity_score' in reply_record
    assert 'replies_per_iteration' in reply_record
    assert 'personas' in reply_record
    assert 'topic' not in reply_record
    assert 'time_filter' not in reply_record
    assert 'subreddit_limit' not in reply_record
    assert 'threads_limit' not in reply_record


def test_linkedin_search_job_payload_includes_time_filter_and_sort_by(tmp_path, monkeypatch) -> None:
    _home, _runs, jobs_config = _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'linkedin-main',
                        'social': 'linkedin',
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(client, name='linkedin-contract-path', social='linkedin', topic='crm sales')

        created = client.post(
            '/jobs',
            json={
                'name': 'linkedin-contract',
                'job_type': 'search',
                'auth_id': 'linkedin-main',
                'search_query_id': path['id'],
                'threads_limit': 20,
                'scroll_rounds': 12,
                'time_filter': 'day',
                'sort_by': 'engagement',
                'selector_wait_timeout_seconds': 25,
                'detail_wait_timeout_seconds': 9,
                'max_comment_expand_clicks': 40,
            },
        )
        assert created.status_code == 200
        payload = created.json()
        assert payload['social'] == 'linkedin'
        assert payload['time_filter'] == 'day'
        assert payload['sort_by'] == 'engagement'
        assert payload['scroll_rounds'] == 12
        assert payload['threads_limit'] == 20
        assert payload['selector_wait_timeout_seconds'] == 25
        assert payload['detail_wait_timeout_seconds'] == 9
        assert payload['max_comment_expand_clicks'] == 40
        assert 'subreddit_limit' not in payload

    persisted = json.loads(jobs_config.read_text(encoding='utf-8'))
    jobs = persisted.get('jobs', [])
    record = next(item for item in jobs if item.get('name') == 'linkedin-contract')
    assert record['social'] == 'linkedin'
    assert record['time_filter'] == 'day'
    assert record['sort_by'] == 'engagement'
    assert record['scroll_rounds'] == 12
    assert record['selector_wait_timeout_seconds'] == 25
    assert record['detail_wait_timeout_seconds'] == 9
    assert record['max_comment_expand_clicks'] == 40
    assert 'subreddit_limit' not in record
