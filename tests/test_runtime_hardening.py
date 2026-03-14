from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.models import Conversation, PersonaProfile, SearchArtifact
from backend.services import llm_service as llm_service_module
from backend.services.job_config_store import JobConfigStore
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


def test_job_config_store_persist_avoids_direct_write_text(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / 'jobs_config.json'
    config_path.write_text('{"jobs": []}', encoding='utf-8')
    store = JobConfigStore(config_path=config_path)

    original_write_text = Path.write_text

    def guarded_write_text(self, *args, **kwargs):
        if self == config_path:
            raise AssertionError('direct write_text must not be used for job config persistence')
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', guarded_write_text)

    store.upsert_job(
        job_id='job-1',
        name='search-one',
        job_type='search',
        source_job_id=None,
        topic='topic',
        min_similarity_score=0.35,
        active=False,
        time_filter='week',
        subreddit_limit=5,
        threads_limit=10,
        replies_per_iteration=3,
        max_runtime_minutes=60,
        personas=[],
    )

    payload = json.loads(config_path.read_text(encoding='utf-8'))
    assert payload['jobs'][0]['name'] == 'search-one'


def test_threads_endpoint_reuses_cached_rows_when_artifacts_unchanged(tmp_path, monkeypatch) -> None:
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
                    }
                ]
            },
        )
        assert auths.status_code == 200
        created = client.post('/jobs', json={'name': 'search-only', 'topic': 'AI launch strategy', 'auth_id': 'reddit-main'})
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
                        thread_id='t1',
                        title='Thread 1',
                        subreddit='sales',
                        url='https://reddit.com/r/sales/comments/t1',
                        score=10,
                        num_comments=3,
                        semantic_similarity=0.7,
                        user_has_commented=False,
                        reply='',
                    )
                ],
            )
        )

        read_calls = {'count': 0}
        original_read_search_artifact = api_main._read_search_artifact

        def counting_read(path: Path):
            read_calls['count'] += 1
            return original_read_search_artifact(path)

        monkeypatch.setattr(api_main, '_read_search_artifact', counting_read)

        first = client.get('/threads?limit=50&offset=0')
        assert first.status_code == 200
        assert first.json()['total'] == 1
        first_calls = read_calls['count']
        assert first_calls >= 1

        second = client.get('/threads?limit=50&offset=0')
        assert second.status_code == 200
        assert second.json()['total'] == 1
        assert read_calls['count'] == first_calls


def test_threads_expose_reply_linkage_and_reply_endpoint_accepts_reply_job_ref(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    class _FakePersonaStore:
        def get_persona(self, name: str):
            normalized = name.strip()
            if not normalized:
                return None
            return PersonaProfile(name=normalized, description='d', objective='o')

    monkeypatch.setattr(api_main, 'PersonaStore', _FakePersonaStore)

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
                    }
                ]
            },
        )
        assert auths.status_code == 200
        created_search = client.post('/jobs', json={'name': 'search-only', 'topic': 'AI launch strategy', 'auth_id': 'reddit-main'})
        assert created_search.status_code == 200
        search_job = created_search.json()

        created_reply = client.post(
            '/jobs',
            json={
                'name': 'reply-worker',
                'job_type': 'reply',
                'source_job_id': search_job['id'],
                'personas': ['seller'],
            },
        )
        assert created_reply.status_code == 200
        reply_job = created_reply.json()

        SearchStorage(api_main.DEFAULT_RUNS_DIR).save_search(
            SearchArtifact(
                id=search_job['id'],
                name=search_job['name'],
                topic=search_job['topic'],
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
                        user_has_commented=False,
                        reply='',
                    )
                ],
            )
        )

        rows_payload = client.get('/threads?limit=50&offset=0')
        assert rows_payload.status_code == 200
        rows = rows_payload.json()['threads']
        assert len(rows) == 1
        row = rows[0]
        assert row['reply_job_id'] == reply_job['id']
        assert row['reply_job_name'] == reply_job['name']
        assert row['reply_personas'] == ['seller']

        reply_response = client.post(
            f"/threads/{reply_job['id']}/sales/t1/reply",
            json={'persona': 'seller', 'reply': 'custom reply'},
        )
        assert reply_response.status_code == 200
        reply_payload = reply_response.json()
        assert reply_payload['job_id'] == search_job['id']
        assert reply_payload['reply_job_id'] == reply_job['id']
        assert reply_payload['reply'] == 'custom reply'


def test_llm_service_raises_when_openrouter_is_unavailable(monkeypatch) -> None:
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setattr(llm_service_module.SettingsStore, 'load', lambda self: {})

    service = llm_service_module.LLMService()

    with pytest.raises(RuntimeError):
        service.generate_response_sync('system', 'hello')
