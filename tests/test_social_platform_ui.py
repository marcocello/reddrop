from pathlib import Path


def test_jobs_page_has_social_selector_and_visibility() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert "Label htmlFor={`${prefix}-job-social`}>Social</Label>" in source
    assert "SelectTrigger id={`${prefix}-job-social`}" in source
    assert "import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'" in source
    assert "import { faLinkedinIn, faRedditAlien } from '@fortawesome/free-brands-svg-icons'" in source
    assert 'function socialIcon(platform: SocialPlatform): IconDefinition {' in source
    assert "function formatSocial(value: SocialPlatform): string {" in source
    assert "return value === 'linkedin' ? 'LinkedIn' : 'Reddit'" in source
    assert "<FontAwesomeIcon icon={socialIcon(row.original.social)} className='size-3.5' />" in source
    assert "<FontAwesomeIcon icon={socialIcon('reddit')} className='size-3.5' />" in source
    assert "<FontAwesomeIcon icon={socialIcon('linkedin')} className='size-3.5' />" in source
    assert 'Reddit' in source
    assert 'LinkedIn' in source
    assert "title='Social'" in source


def test_jobs_create_form_uses_search_path_only_without_inline_plan_builder() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert "Label htmlFor={`${prefix}-search-query-id`}>Search query</Label>" in source
    assert "SelectValue placeholder='Select a saved search query'" in source
    assert 'Generate search query' not in source
    assert "Label htmlFor={`${prefix}-job-topic`}>Topic</Label>" not in source
    assert "Label htmlFor={`${prefix}-reddit-plan-topics`}>Topics</Label>" not in source
    assert "Label htmlFor={`${prefix}-reddit-plan-subreddits`}>Subreddits</Label>" not in source
    assert "Label htmlFor={`${prefix}-reddit-plan-queries`}>Queries</Label>" not in source
    assert "Label htmlFor={`${prefix}-linkedin-plan-query`}>LinkedIn search query</Label>" not in source
    assert 'search_plan:' in source


def test_linkedin_search_form_exposes_time_filter_sort_and_posts_controls() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert "const isLinkedinSearch = form.job_type === 'search' && form.social === 'linkedin'" in source
    assert "const showRedditSearchFilters = form.job_type === 'search' && form.social === 'reddit'" in source
    assert "const showLinkedinSearchFilters = isLinkedinSearch" in source
    assert "showRedditSearchFilters\n              ? 'grid gap-3 sm:grid-cols-3'" in source
    assert "showLinkedinSearchFilters\n                ? 'grid gap-3 sm:grid-cols-7'" in source
    assert "{showRedditSearchFilters ? (" in source
    assert "{showLinkedinSearchFilters ? (" in source
    assert "Label htmlFor={`${prefix}-linkedin-time-filter`}>Time filter</Label>" in source
    assert "time_filter: form.time_filter," in source
    assert "Label htmlFor={`${prefix}-linkedin-sort-by`}>Sort by</Label>" in source
    assert "sort_by: form.sort_by," in source
    assert "Label htmlFor={`${prefix}-scroll-rounds`}>Scroll rounds</Label>" in source
    assert "scroll_rounds: form.scroll_rounds," in source
    assert "Label htmlFor={`${prefix}-selector-wait-timeout-seconds`}>Selector wait (s)</Label>" in source
    assert "selector_wait_timeout_seconds: form.selector_wait_timeout_seconds," in source
    assert "Label htmlFor={`${prefix}-detail-wait-timeout-seconds`}>Detail wait (s)</Label>" in source
    assert "detail_wait_timeout_seconds: form.detail_wait_timeout_seconds," in source
    assert "Label htmlFor={`${prefix}-max-comment-expand-clicks`}>Expand comments clicks</Label>" in source
    assert "max_comment_expand_clicks: form.max_comment_expand_clicks," in source
    assert "isLinkedinSearch ? 'Posts limit' : 'Threads limit'" in source
    assert "subreddit_limit: form.subreddit_limit," in source
    assert (
        "...(selectedSearchQuery?.social === 'reddit'" in source
        or "...(form.social === 'reddit'" in source
    )


def test_frontend_api_contract_includes_social_and_auth_contracts() -> None:
    source = Path('frontend/src/lib/reddrop-api.ts').read_text(encoding='utf-8')
    assert "export type SocialPlatform = 'reddit' | 'linkedin'" in source
    assert 'export type SearchPlan = {' in source
    assert 'social: SocialPlatform' in source
    assert 'auth_id?: string | null' in source
    assert 'sort_by?: string' in source
    assert 'search_plan?: SearchPlan' in source
    assert "buildSearchPlan: (payload: BuildSearchPlanPayload) =>" not in source
    assert "request<BuildSearchPlanResponse>('/jobs/search-plan'" not in source
    assert "generateSearchQuery: (payload: GenerateSearchQueryPayload) =>" in source
    assert "request<SearchQueryResponse>('/search-queries/generate'" in source
    assert 'export type AuthProfile = {' in source
    assert "runLinkedinAuthLogin: (authRef: string) =>" in source


def test_jobs_create_form_auth_dropdown_uses_display_name() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert '{authDisplayName(auth)}' in source


def test_settings_page_is_llm_only() -> None:
    source = Path('frontend/src/features/settings/index.tsx').read_text(encoding='utf-8')
    assert 'OpenRouter configuration' in source
    assert 'Reply generation' not in source
    assert 'System prompt' not in source
    assert 'LinkedIn configuration' not in source
    assert 'Reddit configuration' not in source


def test_auth_page_supports_reddit_and_linkedin_auth_objects() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert "import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'" in source
    assert "import { faLinkedinIn, faRedditAlien } from '@fortawesome/free-brands-svg-icons'" in source
    assert 'function formatSocial(value: SocialPlatform): string {' in source
    assert "return value === 'linkedin' ? 'LinkedIn' : 'Reddit'" in source
    assert "<FontAwesomeIcon icon={socialIcon(row.original.social)} className='size-3.5' />" in source
    assert "<FontAwesomeIcon icon={socialIcon('reddit')} className='size-3.5' />" in source
    assert "<FontAwesomeIcon icon={socialIcon('linkedin')} className='size-3.5' />" in source
    assert "function socialBadgeClass(_value: SocialPlatform): string {" in source
    assert "return 'bg-muted text-muted-foreground'" in source
    assert "bg-sky-600 text-white" not in source
    assert "bg-orange-600 text-white" not in source
    assert 'Create an auth object for Reddit or LinkedIn.' in source
    assert 'Reddit credentials' in source
    assert 'reddit-username' in source
    assert 'reddit-password' in source
    assert 'reddit-client-id' in source
    assert 'reddit-client-secret' in source
    assert 'reddit-user-agent' in source
    assert 'LinkedIn credentials' in source
    assert 'Run LinkedIn login' in source
    assert 'linkedin-email' not in source
    assert 'linkedin-password' not in source
    assert 'linkedin-headless' not in source
    assert 'linkedin-wait-seconds' not in source
    assert 'User data dir' not in source
    assert 'Client ID, client secret, username, and password are required for Reddit validation.' in source
    assert 'Client ID is required for Reddit auth.' in source
    assert 'Client secret is required for Reddit auth.' in source
    assert 'Username is required for Reddit auth.' in source
    assert 'Password is required for Reddit auth.' in source
    assert "Label htmlFor={`${prefix}-name`}>Name</Label>" not in source


def test_auth_page_hides_manual_id_field_and_auto_login_on_linkedin_create() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert "Label htmlFor={`${prefix}-id`}>ID</Label>" not in source
    assert "if (selectedAuth === null && normalized.social === 'linkedin')" in source
    assert 'await reddropApi.runLinkedinAuthLogin(normalized.id)' in source
    assert 'crypto.randomUUID()' in source
    assert 'buildLinkedinProfileDir' in source
    assert '.reddrop/auth/linkedin/' in source
    assert '~/.reddrop/linkedin/' not in source


def test_auth_create_form_removes_name_field() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert "Label htmlFor={`${prefix}-social`}>Social</Label>" in source
    assert "Label htmlFor={`${prefix}-name`}>Name</Label>" not in source


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
    assert "function shortId(value: string): string" in source
    assert "return normalized.slice(0, 8)" in source


def test_auth_table_uses_username_and_status_columns() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert "title='Username'" in source
    assert "title='Status'" in source
    assert "title='Summary'" not in source


def test_auth_page_does_not_call_validation_api_directly() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert 'await reddropApi.validateAuths()' not in source


def test_auth_username_uses_linkedin_username_when_available() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    assert "if (item.social === 'linkedin')" in source
    assert 'item.linkedin_username' in source
