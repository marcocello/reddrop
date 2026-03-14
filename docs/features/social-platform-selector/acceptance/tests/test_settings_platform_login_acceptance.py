from pathlib import Path


def test_auths_support_linkedin_login_trigger_and_runtime_fields() -> None:
    api_source = Path('backend/api/main.py').read_text(encoding='utf-8')
    auths_source = Path('backend/models.py').read_text(encoding='utf-8')
    ui_source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    store_source = Path('backend/services/auth_store.py').read_text(encoding='utf-8')
    linkedin_service_source = Path('backend/services/linkedin_service.py').read_text(encoding='utf-8')

    assert "/auths/{auth_ref}/linkedin/login" in api_source
    assert 'linkedin_user_data_dir' in auths_source
    assert 'linkedin_wait_seconds' in auths_source
    assert '.reddrop/auth/linkedin/' in api_source
    assert '~/.reddrop/linkedin/' not in api_source
    assert 'Run LinkedIn login' in ui_source
    assert "if (selectedAuth === null && normalized.social === 'linkedin')" in ui_source
    assert '.reddrop/auth/linkedin/' in ui_source
    assert '~/.reddrop/linkedin/' not in ui_source
    assert 'LinkedIn login failed. Auth was not created.' in ui_source
    assert 'def login_and_save_session' in linkedin_service_source
    assert 'LinkedIn login not completed. Auth object was not created.' in linkedin_service_source
    assert '"reddit": self._base_dir / "reddit" / "auths.json"' in store_source
    assert '"linkedin": self._base_dir / "linkedin" / "auths.json"' in store_source
