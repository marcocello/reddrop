from pathlib import Path


def test_jobs_page_has_social_selector_and_visibility() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert "Label htmlFor={`${prefix}-job-social`}>Social</Label>" in source
    assert "SelectTrigger id={`${prefix}-job-social`}" in source
    assert "SelectItem value='reddit'>reddit</SelectItem>" in source
    assert "SelectItem value='linkedin'>linkedin</SelectItem>" in source
    assert "title='Social'" in source


def test_frontend_api_contract_includes_social_and_auth_contracts() -> None:
    source = Path('frontend/src/lib/reddrop-api.ts').read_text(encoding='utf-8')
    assert "export type SocialPlatform = 'reddit' | 'linkedin'" in source
    assert 'social: SocialPlatform' in source
    assert 'auth_id?: string | null' in source
    assert 'export type AuthProfile = {' in source
    assert "runLinkedinAuthLogin: (authRef: string) =>" in source


def test_settings_page_is_llm_only() -> None:
    source = Path('frontend/src/features/settings/index.tsx').read_text(encoding='utf-8')
    assert 'OpenRouter configuration' in source
    assert 'LinkedIn configuration' not in source
    assert 'Reddit configuration' not in source


def test_auth_page_supports_reddit_and_linkedin_auth_objects() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert "SelectItem value='reddit'>reddit</SelectItem>" in source
    assert "SelectItem value='linkedin'>linkedin</SelectItem>" in source
    assert 'Reddit credentials' in source
    assert 'reddit-username' in source
    assert 'reddit-password' in source
    assert 'reddit-client-id' not in source
    assert 'reddit-client-secret' not in source
    assert 'reddit-user-agent' not in source
    assert 'LinkedIn credentials' in source
    assert 'Run LinkedIn login' in source
    assert 'linkedin-email' not in source
    assert 'linkedin-password' not in source
    assert 'linkedin-headless' not in source
    assert 'linkedin-wait-seconds' not in source
    assert 'User data dir' not in source


def test_auth_page_hides_manual_id_field_and_auto_login_on_linkedin_create() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert "Label htmlFor={`${prefix}-id`}>ID</Label>" not in source
    assert "if (selectedAuth === null && normalized.social === 'linkedin')" in source
    assert 'await reddropApi.runLinkedinAuthLogin(normalized.id)' in source
    assert 'crypto.randomUUID()' in source
    assert 'buildLinkedinProfileDir' in source
    assert '.reddrop/auth/linkedin/' in source
    assert '~/.reddrop/linkedin/' not in source


def test_auth_create_form_orders_social_before_name() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    social_index = source.find("Label htmlFor={`${prefix}-social`}>Social</Label>")
    name_index = source.find("Label htmlFor={`${prefix}-name`}>Name</Label>")
    assert social_index != -1
    assert name_index != -1
    assert social_index < name_index


def test_linkedin_create_rolls_back_auth_when_login_fails() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert "await reddropApi.saveAuths({ auths })" in source
    assert 'LinkedIn login failed. Auth was not created.' in source


def test_jobs_auth_column_shows_name_only_without_id_noise() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert 'title=\'Auth\'' in source
    assert "font-mono text-xs text-muted-foreground'>{authId}" not in source


def test_auth_table_shows_id_column() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert "accessorKey: 'id'" in source
    assert "title='ID'" in source
