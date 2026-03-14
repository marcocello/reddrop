from __future__ import annotations

import time

from apscheduler.schedulers import SchedulerNotRunningError
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.api import job_runtime as job_runtime_module


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


def test_search_handler_logs_are_captured_in_job_panel(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    def fake_search_handle(args, store, emit, _error, deps):
        deps.logger.info('custom-search-log for %s', args.name)
        job = store.get_job_by_name(args.name)
        assert job is not None
        emit({'runs': [{'id': job.id}]})
        return 0

    monkeypatch.setattr(api_main.search_command, 'handle', fake_search_handle)

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
        created = client.post('/jobs', json={'name': 'log-bridge', 'topic': 'AI launch strategy', 'auth_id': 'reddit-main'})
        assert created.status_code == 200
        job = created.json()

        started = client.post(f"/jobs/{job['id']}/start")
        assert started.status_code == 200

        for _ in range(40):
            status = client.get(f"/jobs/{job['id']}/status")
            assert status.status_code == 200
            payload = status.json()
            if payload.get('last_run_status') in {'success', 'failed', 'stopped'}:
                break
            time.sleep(0.05)

        status = client.get(f"/jobs/{job['id']}/status")
        assert status.status_code == 200
        log_messages = [entry['message'] for entry in status.json().get('logs', [])]
        assert any('custom-search-log for log-bridge' in message for message in log_messages)


def test_runtime_manager_log_entries_are_emitted_to_terminal(monkeypatch) -> None:
    emitted: list[str] = []

    class _FakeLogger:
        def info(self, message: str, *args) -> None:
            rendered = message % args if args else message
            emitted.append(rendered)

    monkeypatch.setattr(job_runtime_module, 'logger', _FakeLogger())

    manager = job_runtime_module.JobRuntimeManager()
    try:
        manager.append_log('job-one', 'log message to both sinks')
    finally:
        manager.shutdown()

    assert any('log message to both sinks' in message for message in emitted)
