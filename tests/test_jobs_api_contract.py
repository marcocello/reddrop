from __future__ import annotations

from datetime import datetime, timezone

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


def test_jobs_are_immutable_after_creation(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        created = client.post(
            '/jobs',
            json={
                'name': 'growth',
                'topic': 'AI launch strategy',
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


def test_jobs_list_includes_summary_counters(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        created = client.post('/jobs', json={'name': 'growth', 'topic': 'AI launch strategy'})
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
