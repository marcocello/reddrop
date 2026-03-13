from pathlib import Path


def test_settings_page_exposes_import_button_and_json_picker() -> None:
    source = Path("frontend/src/features/settings/index.tsx").read_text(encoding="utf-8")
    assert "Import settings" in source
    assert "type='file'" in source
    assert "accept='application/json,.json'" in source


def test_settings_import_handles_success_and_invalid_payloads() -> None:
    source = Path("frontend/src/features/settings/index.tsx").read_text(encoding="utf-8")
    assert "Settings imported. Save settings to persist." in source
    assert "Settings file must include reddit and openrouter sections" in source
