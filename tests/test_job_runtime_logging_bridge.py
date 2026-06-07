from __future__ import annotations

import time
from threading import Event

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
                        'reddit_username': 'alice',
                        'reddit_password': 'pw',
                    }
                ]
            },
        )
        assert auths.status_code == 200
        path = _create_frozen_search_path(client, name='log-bridge-path', social='reddit', topic='AI launch strategy')
        created = client.post('/jobs', json={'name': 'log-bridge', 'search_query_id': path['id'], 'auth_id': 'reddit-main'})
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


def test_runtime_manager_status_contract_keeps_rich_phase_states() -> None:
    started = Event()
    release = Event()

    def _runner(set_phase):
        set_phase('searching')
        started.set()
        assert release.wait(timeout=3)
        return 'ok'

    manager = job_runtime_module.JobRuntimeManager()
    try:
        manager.register('job-status', interval_minutes=60, runner=_runner)
        initial = manager.status('job-status')
        assert initial['status'] == 'inactive'

        manager.start('job-status')
        assert started.wait(timeout=1.5)

        running_payload = manager.status('job-status')
        for _ in range(20):
            if running_payload.get('status') == 'searching':
                break
            time.sleep(0.05)
            running_payload = manager.status('job-status')
        assert running_payload['status'] == 'searching'

        stopping_payload = manager.stop('job-status')
        assert stopping_payload['status'] == 'stopping'

        release.set()
        finished = manager.status('job-status')
        for _ in range(40):
            if finished.get('last_run_status') in {'success', 'failed', 'stopped'}:
                break
            time.sleep(0.05)
            finished = manager.status('job-status')
        assert finished['status'] in {'idle', 'inactive', 'failed'}
    finally:
        manager.shutdown()


def test_runtime_manager_keeps_logs_from_previous_runs() -> None:
    run_count = {'value': 0}

    def _runner(set_phase):
        run_count['value'] += 1
        set_phase('searching')
        return f'run={run_count["value"]}'

    manager = job_runtime_module.JobRuntimeManager()
    try:
        manager.register('job-history', interval_minutes=60, runner=_runner)

        for _ in range(2):
            manager.start('job-history')
            payload = manager.status('job-history')
            for _ in range(40):
                if payload.get('last_run_status') in {'success', 'failed', 'stopped'}:
                    break
                time.sleep(0.05)
                payload = manager.status('job-history')
            assert payload.get('last_run_status') == 'success'

        payload = manager.status('job-history')
        messages = [entry.get('message', '') for entry in payload.get('logs', [])]
        assert any('Run output: run=1' in message for message in messages)
        assert any('Run output: run=2' in message for message in messages)
    finally:
        manager.shutdown()


def test_runtime_manager_restores_job_logs_from_disk_state(tmp_path) -> None:
    state_file = tmp_path / 'runtime-state.json'

    manager = job_runtime_module.JobRuntimeManager(state_path=state_file)
    try:
        manager.register('job-persisted', interval_minutes=60, runner=lambda _set_phase: 'persisted-output')
        manager.start('job-persisted')
        payload = manager.status('job-persisted')
        for _ in range(40):
            if payload.get('last_run_status') in {'success', 'failed', 'stopped'}:
                break
            time.sleep(0.05)
            payload = manager.status('job-persisted')
        assert payload.get('last_run_status') == 'success'
        assert any('Run output: persisted-output' in entry.get('message', '') for entry in payload.get('logs', []))
    finally:
        manager.shutdown()

    restored = job_runtime_module.JobRuntimeManager(state_path=state_file)
    try:
        payload = restored.status('job-persisted')
        messages = [entry.get('message', '') for entry in payload.get('logs', [])]
        assert any('Run output: persisted-output' in message for message in messages)
    finally:
        restored.shutdown()
