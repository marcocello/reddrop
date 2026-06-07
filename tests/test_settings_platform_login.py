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


def test_linkedin_login_endpoint_uses_service_login_flow(tmp_path, monkeypatch) -> None:
    _setup_isolated_runtime(tmp_path, monkeypatch)

    captured: dict[str, object] = {}
    resolver_state = {"value": "example-user"}

    class _FakeLinkedinService:
        @staticmethod
        def login_and_save_session(*, user_data_dir: str, wait_seconds: int, login_url: str = "https://www.linkedin.com/login"):
            captured["user_data_dir"] = user_data_dir
            captured["wait_seconds"] = wait_seconds
            captured["login_url"] = login_url
            return {
                "status": "session_saved",
                "user_data_dir": user_data_dir,
                "login_url": login_url,
            }

        @staticmethod
        def resolve_saved_session_username(*, user_data_dir: str) -> str:
            captured["resolve_user_data_dir"] = user_data_dir
            return str(resolver_state["value"])

        @staticmethod
        def validate_saved_session(*, user_data_dir: str) -> bool:
            captured["validate_user_data_dir"] = user_data_dir
            return True

    monkeypatch.setattr(api_main, "LinkedinService", _FakeLinkedinService)

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
        assert payload['status'] == 'session_saved'
        assert 'auth_id' in payload
        assert payload['auth_id'] == 'linkedin-main'
        resolver_state["value"] = "changed-after-login"
        auths = client.get('/auths')
        assert auths.status_code == 200
        by_id = {item['id']: item for item in auths.json()['auths']}
        assert by_id['linkedin-main']['linkedin_username'] == 'example-user'
        assert by_id['linkedin-main']['status'] == 'ok'

    assert '.reddrop/auth/linkedin/linkedin-main-linkedin' in str(captured['user_data_dir'])
    assert not str(captured['user_data_dir']).startswith(str(Path.home() / '.reddrop'))
    assert captured['wait_seconds'] == 120
    assert captured['login_url'] == "https://www.linkedin.com/login"


def test_linkedin_service_verifies_login_completion_and_closes_browser() -> None:
    source = Path('backend/services/linkedin_service.py').read_text(encoding='utf-8')
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


def test_linkedin_username_resolution_uses_voyager_api_fallback() -> None:
    source = Path('backend/services/linkedin_service.py').read_text(encoding='utf-8')
    assert 'https://www.linkedin.com/voyager/api/me' in source
    assert 'csrf-token' in source
    assert 'JSESSIONID' in source
    assert 'publicIdentifier' in source


def test_linkedin_login_persists_session_in_profile_dir() -> None:
    source = Path('backend/services/linkedin_service.py').read_text(encoding='utf-8')
    assert 'launch_persistent_context(' in source
    assert '--incognito' not in source


def test_linkedin_defaults_use_project_local_reddrop_paths() -> None:
    models_source = Path('backend/models.py').read_text(encoding='utf-8')
    service_source = Path('backend/services/linkedin_service.py').read_text(encoding='utf-8')
    api_source = Path('backend/api/main.py').read_text(encoding='utf-8')
    cli_source = Path('backend/cli/reddrop.py').read_text(encoding='utf-8')

    assert 'linkedin_user_data_dir: str = ".reddrop/auth/linkedin"' in models_source
    assert '~/.reddrop' not in models_source

    assert 'LINKEDIN_USER_DATA_DIR") or ".reddrop/auth/linkedin"' in service_source
    assert 'def login_and_save_session' in service_source
    assert '~/.reddrop' not in service_source

    assert 'DEFAULT_HOME = _BACKEND_ROOT / ".reddrop"' in api_source
    assert '_resolve_linkedin_user_data_dir' in api_source
    assert 'DEFAULT_HOME = _BACKEND_ROOT / ".reddrop"' in cli_source
    assert '_resolve_linkedin_user_data_dir' in cli_source


def test_backend_source_relative_defaults_are_used_for_runtime_files() -> None:
    api_source = Path('backend/api/main.py').read_text(encoding='utf-8')
    cli_source = Path('backend/cli/reddrop.py').read_text(encoding='utf-8')
    linkedin_service_source = Path('backend/services/linkedin_service.py').read_text(encoding='utf-8')
    persona_store_source = Path('backend/services/persona_store.py').read_text(encoding='utf-8')
    settings_store_source = Path('backend/services/settings_store.py').read_text(encoding='utf-8')
    job_store_source = Path('backend/services/job_config_store.py').read_text(encoding='utf-8')
    auth_store_source = Path('backend/services/auth_store.py').read_text(encoding='utf-8')

    assert '_BACKEND_ROOT = Path(__file__).resolve().parents[1]' in api_source
    assert '_BACKEND_ROOT = Path(__file__).resolve().parents[1]' in cli_source
    assert '_BACKEND_ROOT = Path(__file__).resolve().parents[1]' in linkedin_service_source
    assert '_BACKEND_ROOT / ".reddrop/personas.json"' in persona_store_source
    assert '_BACKEND_ROOT / ".reddrop/settings.yaml"' in settings_store_source
    assert '_BACKEND_ROOT / ".reddrop/jobs_config.json"' in job_store_source
    assert '_BACKEND_ROOT / ".reddrop/auth"' in auth_store_source
