from pathlib import Path


def test_repo_layout_for_backend_and_frontend() -> None:
    assert Path("backend").is_dir()
    assert Path("frontend").is_dir()
    assert Path("backend/Dockerfile").is_file()
    assert Path("frontend/Dockerfile").is_file()
    assert Path("deployment/docker-compose.yml").is_file()
    assert Path("deployment/frontend.nginx.conf").is_file()
    assert Path("backend/api/main.py").is_file()
    assert Path("backend/cli/reddrop.py").is_file()
    assert Path("backend/services/reddit_service.py").is_file()
    assert Path("frontend/src/main.tsx").is_file()
    assert Path("backend/reddrop").is_file()
    assert Path("backend/requirements.txt").is_file()
    assert not Path("requirements.txt").exists()
    assert not Path(".gitignore copy").exists()

    compose = Path("deployment/docker-compose.yml").read_text(encoding="utf-8")
    assert "dockerfile: backend/Dockerfile" in compose
    assert "dockerfile: frontend/Dockerfile" in compose
    assert "\n  frontend:" in compose
    assert "5173:80" in compose

    nginx = Path("deployment/frontend.nginx.conf").read_text(encoding="utf-8")
    assert "location /api/" in nginx
    assert "proxy_pass http://backend:8000/" in nginx

    frontend_dockerfile = Path("frontend/Dockerfile").read_text(encoding="utf-8")
    assert "if [ -f package-lock.json ]" in frontend_dockerfile

    settings_ui = Path("frontend/src/features/settings/index.tsx").read_text(encoding="utf-8")
    assert "Import settings" in settings_ui
    assert "type='file'" in settings_ui


def test_gitignore_is_merged() -> None:
    source = Path(".gitignore").read_text(encoding="utf-8")
    assert "!**/reddrop" in source
    assert "!*.ts" in source
    assert "!*.tsx" in source
    assert "node_modules/" in source
    assert ".DS_Store" in source
