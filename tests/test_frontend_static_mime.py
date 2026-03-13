from pathlib import Path


def test_nginx_does_not_fallback_html_for_module_assets() -> None:
    source = Path("deployment/frontend.nginx.conf").read_text(encoding="utf-8")
    assert "location ~* \\.(" in source
    assert "js|mjs" in source
    assert "try_files $uri =404;" in source


def test_runtime_env_module_exists_for_static_serving() -> None:
    source = Path("frontend/public/env.mjs").read_text(encoding="utf-8")
    assert "export const env" in source
    assert "export default env" in source
