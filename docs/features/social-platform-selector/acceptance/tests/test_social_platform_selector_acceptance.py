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
    assert "import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'" in jobs_source
    assert "import { faLinkedinIn, faRedditAlien } from '@fortawesome/free-brands-svg-icons'" in jobs_source
    assert "return value === 'linkedin' ? 'LinkedIn' : 'Reddit'" in jobs_source
    assert "<FontAwesomeIcon icon={socialIcon('reddit')} className='size-3.5' />" in jobs_source
    assert "<FontAwesomeIcon icon={socialIcon('linkedin')} className='size-3.5' />" in jobs_source

    assert "import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'" in auths_source
    assert "import { faLinkedinIn, faRedditAlien } from '@fortawesome/free-brands-svg-icons'" in auths_source
    assert "return value === 'linkedin' ? 'LinkedIn' : 'Reddit'" in auths_source
    assert "<FontAwesomeIcon icon={socialIcon('reddit')} className='size-3.5' />" in auths_source
    assert "<FontAwesomeIcon icon={socialIcon('linkedin')} className='size-3.5' />" in auths_source
    assert 'Create an auth object for Reddit or LinkedIn.' in auths_source
    assert 'LinkedIn credentials' in auths_source
    assert 'linkedin-email' not in auths_source
    assert 'linkedin-password' not in auths_source
    assert 'linkedin-headless' not in auths_source
    assert 'linkedin-wait-seconds' not in auths_source
    assert 'reddit-username' in auths_source
    assert 'reddit-password' in auths_source
    assert 'reddit-client-id' in auths_source
    assert 'reddit-client-secret' in auths_source
    assert 'reddit-user-agent' in auths_source
    assert 'Client ID, client secret, username, and password are required for Reddit validation.' in auths_source
    assert "accessorKey: 'id'" in auths_source
    assert "Label htmlFor={`${prefix}-name`}>Name</Label>" not in auths_source
    assert "title='Username'" in auths_source
    assert "title='Status'" in auths_source


def test_runtime_logging_bridge_is_present() -> None:
    runtime_source = Path('backend/api/job_runtime.py').read_text(encoding='utf-8')
    api_source = Path('backend/api/main.py').read_text(encoding='utf-8')

    assert 'def append_log(self, name: str, message: str) -> None:' in runtime_source
    assert 'logger.info("[job=%s] %s", normalized_name, message)' in runtime_source
    assert 'class _JobBoundLogger:' in api_source


def test_auth_catalog_refresh_is_non_validating_and_still_resolves_linkedin_username() -> None:
    api_source = Path('backend/api/main.py').read_text(encoding='utf-8')
    assert 'def _resolve_linkedin_username_from_profile(auth: AuthProfile) -> str:' in api_source
    assert 'def _auth_catalog_entry(item: AuthProfile) -> dict[str, Any]:' in api_source
    assert 'auths = [_auth_catalog_entry(item) for item in store.list_auths()]' in api_source
    assert 'Auth validation start:' not in api_source
    assert 'Auth validation result:' not in api_source


def test_sidebar_navigation_groups_personas_under_workflows() -> None:
    source = Path('frontend/src/components/layout/data/sidebar-data.ts').read_text(encoding='utf-8')
    workflows_start = source.find("title: 'Workflows'")
    results_start = source.find("title: 'Results'")
    configuration_start = source.find("title: 'Configuration'")

    assert workflows_start != -1
    assert results_start != -1
    assert configuration_start != -1
    assert workflows_start < results_start < configuration_start

    workflows_section = source[workflows_start:results_start]
    results_section = source[results_start:configuration_start]
    configuration_section = source[configuration_start:]

    assert "title: 'Personas'" in workflows_section
    assert "title: 'Search Queries'" in workflows_section
    assert "title: 'Jobs'" in workflows_section
    assert "title: 'Conversations'" not in workflows_section
    assert "title: 'Conversations'" in results_section
    assert "title: 'Jobs'" not in results_section
    assert "title: 'Auths'" in configuration_section
    assert "title: 'Settings'" in configuration_section
    assert "title: 'Personas'" not in configuration_section
