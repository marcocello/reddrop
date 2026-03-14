from pathlib import Path


def test_jobs_model_and_api_include_social_platform_field() -> None:
    models_source = Path('backend/models.py').read_text(encoding='utf-8')
    api_source = Path('backend/api/main.py').read_text(encoding='utf-8')

    assert 'social: Literal["reddit", "linkedin"] = "reddit"' in models_source
    assert 'social: Literal["reddit", "linkedin"] = "reddit"' in api_source


def test_jobs_ui_and_auths_ui_expose_linkedin_controls() -> None:
    jobs_source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    auths_source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')

    assert "Label htmlFor={`${prefix}-job-social`}>Social</Label>" in jobs_source
    assert "SelectItem value='reddit'>reddit</SelectItem>" in jobs_source
    assert "SelectItem value='linkedin'>linkedin</SelectItem>" in jobs_source

    assert 'LinkedIn credentials' in auths_source
    assert 'linkedin-email' not in auths_source
    assert 'linkedin-password' not in auths_source
    assert 'reddit-username' in auths_source
    assert 'reddit-password' in auths_source
    assert 'reddit-client-id' not in auths_source
    assert 'reddit-client-secret' not in auths_source


def test_runtime_logging_bridge_is_present() -> None:
    runtime_source = Path('backend/api/job_runtime.py').read_text(encoding='utf-8')
    api_source = Path('backend/api/main.py').read_text(encoding='utf-8')

    assert 'def append_log(self, name: str, message: str) -> None:' in runtime_source
    assert 'logger.info("[job=%s] %s", normalized_name, message)' in runtime_source
    assert 'class _JobBoundLogger:' in api_source


def test_sidebar_navigation_groups_personas_under_workflows() -> None:
    source = Path('frontend/src/components/layout/data/sidebar-data.ts').read_text(encoding='utf-8')
    workflows_start = source.find("title: 'Workflows'")
    configuration_start = source.find("title: 'Configuration'")

    assert workflows_start != -1
    assert configuration_start != -1
    assert workflows_start < configuration_start

    workflows_section = source[workflows_start:configuration_start]
    configuration_section = source[configuration_start:]

    assert "title: 'Jobs'" in workflows_section
    assert "title: 'Threads'" in workflows_section
    assert "title: 'Personas'" in workflows_section
    assert "title: 'Auths'" in configuration_section
    assert "title: 'Settings'" in configuration_section
    assert "title: 'Personas'" not in configuration_section
