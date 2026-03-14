from __future__ import annotations

from pathlib import Path

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


def test_settings_include_linkedin_login_runtime_fields(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    with TestClient(api_main.app) as client:
        save = client.put(
            '/settings',
            json={
                'openrouter': {
                    'api_key': 'ok',
                    'base_url': 'https://openrouter.ai/api/v1',
                    'model': 'x-ai/grok-4.1-fast',
                    'http_referer': 'http://localhost',
                    'x_title': 'reddrop',
                    'timeout_seconds': '20',
                },
            },
        )
        assert save.status_code == 200
        payload = save.json()
        assert set(payload.keys()) == {'openrouter'}
        assert payload['openrouter']['model'] == 'x-ai/grok-4.1-fast'


def test_linkedin_login_endpoint_runs_scraper_command(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    captured: dict[str, object] = {}

    class _Result:
        returncode = 0
        stdout = 'ok'
        stderr = ''

    def fake_run(cmd, capture_output, text, timeout, cwd):
        captured['cmd'] = cmd
        captured['capture_output'] = capture_output
        captured['text'] = text
        captured['timeout'] = timeout
        captured['cwd'] = cwd
        return _Result()

    monkeypatch.setattr(api_main.subprocess, 'run', fake_run)

    with TestClient(api_main.app) as client:
        save = client.put(
            '/auths',
            json={
                'auths': [
                    {
                        'id': 'linkedin-main',
                        'name': 'LinkedIn main',
                        'social': 'linkedin',
                        'linkedin_email': 'john@example.com',
                        'linkedin_password': 'pw',
                        'linkedin_headless': 'true',
                        'linkedin_user_data_dir': '/tmp/linkedin-profile',
                        'linkedin_wait_seconds': '180',
                    }
                ],
            },
        )
        assert save.status_code == 200

        login = client.post('/auths/linkedin-main/linkedin/login')
        assert login.status_code == 200
        payload = login.json()
        assert payload['status'] == 'ok'

    command = captured['cmd']
    assert isinstance(command, list)
    assert '-m' in command
    assert 'app.linkedin_scraper' in command
    assert 'login' in command
    assert '--user-data-dir' in command
    assert any('.reddrop/auth/linkedin/linkedin-main-linkedin' in str(part) for part in command)
    assert not any(str(part).startswith(str(Path.home() / '.reddrop')) for part in command)
    assert '--wait-seconds' in command
    assert '120' in command
    assert captured['cwd'] == str(api_main._REPO_ROOT)


def test_linkedin_scraper_verifies_login_completion_and_closes_browser() -> None:
    source = Path('app/linkedin_scraper.py').read_text(encoding='utf-8')
    assert 'def _url_requires_login' in source
    assert 'def _url_is_authenticated' in source
    assert 'def _url_points_to_feed' in source
    assert 'linkedin.com/feed' in source
    assert 'def _has_linkedin_session_cookie' in source
    assert 'cookie.get("name") == "li_at"' in source
    assert 'context.request.get(' in source
    assert 'https://www.linkedin.com/feed/' in source
    assert 'LinkedIn login not completed' in source
    assert 'page.is_closed()' in source
    assert 'context.pages' in source
    assert 'time.sleep(wait_seconds)' not in source
    assert 'finally:' in source
    assert 'context.close()' in source


def test_linkedin_defaults_use_project_local_reddrop_paths() -> None:
    scraper_source = Path('app/linkedin_scraper.py').read_text(encoding='utf-8')
    models_source = Path('backend/models.py').read_text(encoding='utf-8')
    service_source = Path('backend/services/linkedin_service.py').read_text(encoding='utf-8')
    api_source = Path('backend/api/main.py').read_text(encoding='utf-8')
    cli_source = Path('backend/cli/reddrop.py').read_text(encoding='utf-8')

    assert '.reddrop/auth/linkedin' in scraper_source
    assert '~/.linkedin-playwright' not in scraper_source
    assert '~/.reddrop' not in scraper_source

    assert 'linkedin_user_data_dir: str = ".reddrop/auth/linkedin"' in models_source
    assert '~/.linkedin-playwright' not in models_source

    assert 'LINKEDIN_USER_DATA_DIR") or ".reddrop/auth/linkedin"' in service_source
    assert '~/.linkedin-playwright' not in service_source

    assert 'DEFAULT_HOME = _REPO_ROOT / ".reddrop"' in api_source
    assert '_resolve_linkedin_user_data_dir' in api_source
    assert 'DEFAULT_HOME = _REPO_ROOT / ".reddrop"' in cli_source
    assert '_resolve_linkedin_user_data_dir' in cli_source
