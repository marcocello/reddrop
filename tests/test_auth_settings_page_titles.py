from pathlib import Path


def test_auth_page_has_auth_heading() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert ">Auth</h2>" in source


def test_settings_page_has_settings_heading() -> None:
    source = Path('frontend/src/features/settings/index.tsx').read_text(encoding='utf-8')
    assert ">Settings</h2>" in source
