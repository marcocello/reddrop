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
    return home


def test_personas_can_be_saved_and_loaded_from_project_local_store(tmp_path, monkeypatch) -> None:
    home = _setup_isolated_runtime(tmp_path, monkeypatch)
    with TestClient(api_main.app) as client:
        saved = client.put(
            '/personas',
            json={
                'personas': [
                    {
                        'name': 'seller',
                        'objective': 'Book intro calls. Be direct and practical.',
                    }
                ]
            },
        )
        assert saved.status_code == 200
        payload = saved.json()
        assert len(payload['personas']) == 1
        assert payload['personas'][0]['name'] == 'seller'
        assert payload['personas'][0]['objective'] == 'Book intro calls. Be direct and practical.'
        assert payload['source'] == str(home / 'personas.json')

        listed = client.get('/personas')
        assert listed.status_code == 200
        listed_payload = listed.json()
        assert len(listed_payload['personas']) == 1
        assert listed_payload['personas'][0]['name'] == 'seller'
        assert listed_payload['personas'][0]['objective'] == 'Book intro calls. Be direct and practical.'
