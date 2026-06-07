from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.cli.commands import search as search_command
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


def _create_frozen_search_path(client: TestClient, *, name: str, social: str, topic: str) -> dict:
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


def test_scenario_conversations_have_uuid_identity_and_dedupe_by_url(tmp_path) -> None:
    storage = SearchStorage(tmp_path / 'runs')
    now = datetime.now(timezone.utc)

    first = SearchArtifact(
        id='job-1',
        name='job-1',
        topic='crm sales',
        created_at=now,
        updated_at=now,
        conversations=[
            Conversation(
                thread_id='t1',
                title='First',
                subreddit='sales',
                url='https://example.com/post/1',
                score=1,
                num_comments=2,
                semantic_similarity=0.5,
                user_has_commented=True,
                reply='existing reply',
            )
        ],
    )
    storage.save_search(first)

    second = SearchArtifact(
        id='job-1',
        name='job-1',
        topic='crm sales',
        created_at=now,
        updated_at=datetime.now(timezone.utc),
        conversations=[
            Conversation(
                thread_id='t2',
                title='Second',
                subreddit='marketing',
                url='https://example.com/post/1',
                score=5,
                num_comments=8,
                semantic_similarity=0.9,
                user_has_commented=False,
                reply='',
            )
        ],
    )
    storage.save_search(second)

    persisted = storage.get_search('job-1')
    assert persisted is not None
    assert len(persisted.conversations) == 1
    item = persisted.conversations[0]
    assert item.reply == 'existing reply'
    assert item.user_has_commented is True
    assert isinstance(item.id, UUID)
    assert item.id == uuid5(NAMESPACE_URL, 'https://example.com/post/1')


def test_scenario_threads_endpoint_exposes_conversation_uuid(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    with TestClient(api_main.app) as client:
        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-main',
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
        path = _create_frozen_search_path(client, name='uuid-conversations-acc-path', social='reddit', topic='AI')

        job = client.post(
            '/jobs',
            json={'name': 'uuid-conversations-acc', 'search_query_id': path['id'], 'auth_id': 'reddit-main'},
        )
        assert job.status_code == 200
        payload = job.json()

        SearchStorage(api_main.DEFAULT_RUNS_DIR).save_search(
            SearchArtifact(
                id=payload['id'],
                name=payload['name'],
                topic='AI',
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                conversations=[
                    Conversation(
                        thread_id='t1',
                        title='Thread',
                        subreddit='sales',
                        url='https://example.com/post/1',
                        score=10,
                        num_comments=3,
                        semantic_similarity=0.7,
                        user_has_commented=False,
                        reply='',
                    )
                ],
            )
        )

        rows = client.get('/threads?limit=20&offset=0')
        assert rows.status_code == 200
        thread_rows = rows.json()['threads']
        assert len(thread_rows) == 1
        UUID(thread_rows[0]['conversation_id'])
        assert thread_rows[0]['conversation_id'] == str(uuid5(NAMESPACE_URL, 'https://example.com/post/1'))


def test_scenario_rerun_initialization_does_not_wipe_existing_conversations(tmp_path) -> None:
    class _FakeLogger:
        def info(self, message: str, *args) -> None:
            _ = (message, args)

    class _FailingRedditService:
        def discover_relevant_threads(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError('discovery failed')

    storage = SearchStorage(tmp_path / 'runs')
    now = datetime.now(timezone.utc)
    storage.save_search(
        SearchArtifact(
            id='job-1',
            name='job-1',
            topic='crm sales',
            created_at=now,
            updated_at=now,
            conversations=[
                Conversation(
                    thread_id='t1',
                    title='Existing thread',
                    subreddit='sales',
                    url='https://example.com/post/1',
                    score=1,
                    num_comments=2,
                    semantic_similarity=0.4,
                    user_has_commented=False,
                    reply='existing reply',
                )
            ],
        )
    )

    deps = search_command.SearchDependencies(
        logger=_FakeLogger(),
        search_storage_factory=lambda: storage,
        reddit_service_cls=_FailingRedditService,
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=lambda: datetime.now(timezone.utc),
    )
    job = SimpleNamespace(
        id='job-1',
        name='job-1',
        topic='crm sales',
        time_filter='week',
        subreddit_limit=5,
        threads_limit=10,
        search_plan=SimpleNamespace(
            model_dump=lambda mode='python': {
                'topics': ['crm sales'],
                'subreddits': ['sales'],
                'queries': ['crm sales'],
            }
        ),
    )

    with pytest.raises(RuntimeError, match='discovery failed'):
        search_command._search_job(job, storage, deps)

    persisted = storage.get_search('job-1')
    assert persisted is not None
    assert len(persisted.conversations) == 1
    assert persisted.conversations[0].thread_id == 't1'
    assert persisted.conversations[0].reply == 'existing reply'


def test_scenario_threads_listing_dedupes_same_url_across_jobs(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    with TestClient(api_main.app) as client:
        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-main',
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
        path = _create_frozen_search_path(client, name='acc-cross-job-dedupe-path', social='reddit', topic='AI')

        first_job = client.post(
            '/jobs',
            json={'name': 'acc-cross-job-dedupe-1', 'search_query_id': path['id'], 'auth_id': 'reddit-main'},
        )
        assert first_job.status_code == 200
        first_payload = first_job.json()

        second_job = client.post(
            '/jobs',
            json={'name': 'acc-cross-job-dedupe-2', 'search_query_id': path['id'], 'auth_id': 'reddit-main'},
        )
        assert second_job.status_code == 200
        second_payload = second_job.json()

        storage = SearchStorage(api_main.DEFAULT_RUNS_DIR)
        shared_url = 'https://example.com/post/1'
        now = datetime.now(timezone.utc)
        storage.save_search(
            SearchArtifact(
                id=first_payload['id'],
                name=first_payload['name'],
                topic='AI',
                created_at=now,
                updated_at=now,
                conversations=[
                    Conversation(
                        thread_id='t1',
                        title='Thread 1',
                        subreddit='sales',
                        url=shared_url,
                        score=10,
                        num_comments=3,
                        semantic_similarity=0.8,
                        user_has_commented=False,
                        reply='',
                    )
                ],
            )
        )
        storage.save_search(
            SearchArtifact(
                id=second_payload['id'],
                name=second_payload['name'],
                topic='AI',
                created_at=now,
                updated_at=now,
                conversations=[
                    Conversation(
                        thread_id='t2',
                        title='Thread 2',
                        subreddit='sales',
                        url=shared_url,
                        score=5,
                        num_comments=2,
                        semantic_similarity=0.3,
                        user_has_commented=False,
                        reply='from second job',
                    )
                ],
            )
        )

        rows = client.get('/threads?limit=20&offset=0')
        assert rows.status_code == 200
        payload = rows.json()
        assert payload['total'] == 1
        assert len(payload['threads']) == 1
        assert payload['threads'][0]['url'] == shared_url
        assert payload['threads'][0]['reply'] == 'from second job'
        assert payload['threads'][0]['has_reply'] is True


def test_scenario_linkedin_activity_and_ugcpost_variants_dedupe_as_one_post(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    with TestClient(api_main.app) as client:
        auths = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'reddit-main',
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
        path = _create_frozen_search_path(client, name='acc-linkedin-alias-dedupe-path', social='reddit', topic='AI')

        first_job = client.post(
            '/jobs',
            json={'name': 'acc-linkedin-alias-dedupe-1', 'search_query_id': path['id'], 'auth_id': 'reddit-main'},
        )
        assert first_job.status_code == 200
        first_payload = first_job.json()

        second_job = client.post(
            '/jobs',
            json={'name': 'acc-linkedin-alias-dedupe-2', 'search_query_id': path['id'], 'auth_id': 'reddit-main'},
        )
        assert second_job.status_code == 200
        second_payload = second_job.json()

        storage = SearchStorage(api_main.DEFAULT_RUNS_DIR)
        shared_text = (
            "Accurate forecasting depends on trusted CRM data. "
            "Teams with clean and consistent updates plan better and execute faster."
        )
        now = datetime.now(timezone.utc)
        storage.save_search(
            SearchArtifact(
                id=first_payload['id'],
                name=first_payload['name'],
                topic='AI',
                created_at=now,
                updated_at=now,
                conversations=[
                    Conversation(
                        thread_id='urn:li:activity:7439249142608199680',
                        title='Accurate forecasting depends on trusted CRM data',
                        subreddit='linkedin',
                        url='https://www.linkedin.com/feed/update/urn:li:activity:7439249142608199680/',
                        score=0,
                        num_comments=0,
                        created_utc=1773667721.424225,
                        semantic_similarity=0.44,
                        user_has_commented=False,
                        reply='',
                        selftext=shared_text,
                    )
                ],
            )
        )
        storage.save_search(
            SearchArtifact(
                id=second_payload['id'],
                name=second_payload['name'],
                topic='AI',
                created_at=now,
                updated_at=now,
                conversations=[
                    Conversation(
                        thread_id='urn:li:ugcPost:7432062048492081152',
                        title='Accurate forecasting depends on trusted CRM data',
                        subreddit='linkedin',
                        url='https://www.linkedin.com/feed/update/urn:li:ugcPost:7432062048492081152/',
                        score=0,
                        num_comments=0,
                        created_utc=1773667721.424225,
                        semantic_similarity=0.44,
                        user_has_commented=False,
                        reply='draft',
                        selftext=shared_text,
                    )
                ],
            )
        )

        rows = client.get('/threads?limit=20&offset=0')
        assert rows.status_code == 200
        payload = rows.json()
        assert payload['total'] == 1
        assert len(payload['threads']) == 1
        assert payload['threads'][0]['subreddit'] == 'linkedin'
        assert payload['threads'][0]['reply'] == 'draft'
        assert payload['threads'][0]['has_reply'] is True


def test_scenario_search_persistence_maintains_single_conversation_file(tmp_path) -> None:
    runs_dir = tmp_path / 'runs'
    storage = SearchStorage(runs_dir)
    now = datetime.now(timezone.utc)
    storage.save_search(
        SearchArtifact(
            id='job-1',
            name='job-1',
            topic='crm sales',
            created_at=now,
            updated_at=now,
            conversations=[
                Conversation(
                    thread_id='t1',
                    title='Thread 1',
                    subreddit='sales',
                    url='https://example.com/post/1',
                    score=1,
                    num_comments=1,
                    semantic_similarity=0.4,
                    user_has_commented=False,
                    reply='',
                )
            ],
        )
    )
    storage.save_search(
        SearchArtifact(
            id='job-2',
            name='job-2',
            topic='crm ops',
            created_at=now,
            updated_at=now,
            conversations=[
                Conversation(
                    thread_id='t2',
                    title='Thread 2',
                    subreddit='sales',
                    url='https://example.com/post/2',
                    score=2,
                    num_comments=2,
                    semantic_similarity=0.5,
                    user_has_commented=False,
                    reply='draft',
                )
            ],
        )
    )

    payload = json.loads((runs_dir / 'conversation.json').read_text(encoding='utf-8'))
    assert len(payload['artifacts']) == 2
    assert len(payload['conversations']) == 2
    assert all('topic' not in item for item in payload['artifacts'])
    assert all('topic' not in item for item in payload['conversations'])


def test_scenario_search_persistence_uses_single_shared_lock_file(tmp_path) -> None:
    runs_dir = tmp_path / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    legacy_lock = runs_dir / 'search_deadbeef.json.lock'
    legacy_lock.write_text('', encoding='utf-8')

    storage = SearchStorage(runs_dir)
    assert not legacy_lock.exists()

    now = datetime.now(timezone.utc)
    storage.save_search(
        SearchArtifact(
            id='job-1',
            name='job-1',
            topic='crm sales',
            created_at=now,
            updated_at=now,
            conversations=[],
        )
    )

    assert (runs_dir / '.search_storage.lock').exists()


def test_scenario_search_artifact_payload_omits_topic(tmp_path) -> None:
    runs_dir = tmp_path / 'runs'
    storage = SearchStorage(runs_dir)
    now = datetime.now(timezone.utc)
    path = storage.save_search(
        SearchArtifact(
            id='job-1',
            name='job-1',
            topic='crm sales',
            created_at=now,
            updated_at=now,
            conversations=[],
        )
    )

    payload = json.loads(path.read_text(encoding='utf-8'))
    assert 'topic' not in payload
