from pathlib import Path


def test_search_paths_feature_and_jobs_form_contract() -> None:
    api_source = Path('frontend/src/lib/reddrop-api.ts').read_text(encoding='utf-8')
    jobs_source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    search_paths_source = Path('frontend/src/features/search-queries/index.tsx').read_text(encoding='utf-8')
    sidebar_source = Path('frontend/src/components/layout/data/sidebar-data.ts').read_text(encoding='utf-8')

    assert 'export type SearchQuery = {' in api_source
    assert 'listSearchQueries:' in api_source
    assert 'generateSearchQuery:' in api_source
    assert 'saveSearchQuery:' in api_source
    assert 'auth_id: string' not in api_source.split('export type SearchQuery = {', 1)[1].split('}', 1)[0]

    assert 'search_query_id' in jobs_source
    assert 'Search query' in jobs_source
    assert 'createJob(payload)' in jobs_source

    assert 'export function SearchQueriesPage()' in search_paths_source
    assert 'Generate search query' in search_paths_source
    assert 'Save query' in search_paths_source
    assert 'Auth profile' not in search_paths_source
    assert '<TableHead>Auth</TableHead>' not in search_paths_source
    assert 'Build and freeze deterministic query inputs before creating Search jobs.' not in search_paths_source
    assert "placeholder='Search by name, topic, social or status...'" in search_paths_source

    assert "title: 'Search Queries'" in sidebar_source
    assert "url: '/search-queries'" in sidebar_source


def test_search_path_draft_sheet_header_actions_do_not_clip() -> None:
    search_paths_source = Path('frontend/src/features/search-queries/index.tsx').read_text(encoding='utf-8')

    assert "className='sticky top-0 z-10 mt-3 border-y bg-background px-4 py-3'" in search_paths_source
    assert "className='flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between'" in search_paths_source
    assert "className='flex min-w-0 flex-wrap items-center gap-1.5'" in search_paths_source
    assert "className='max-w-[16rem] truncate bg-muted text-muted-foreground'" in search_paths_source
    assert "className='flex w-full flex-wrap items-center justify-end gap-2 sm:w-auto'" in search_paths_source


def test_search_path_sidebar_uses_compact_topic_and_larger_plan_inputs() -> None:
    search_paths_source = Path('frontend/src/features/search-queries/index.tsx').read_text(encoding='utf-8')

    assert "const DEFAULT_LINKEDIN_QUERY_INPUT_COUNT = 5" in search_paths_source
    assert "id='sheet-query-topic'" in search_paths_source
    assert "className='mt-3 grid gap-3'" in search_paths_source
    assert "className='grid gap-2 sm:grid-cols-[7rem_minmax(0,1fr)] sm:items-center'" in search_paths_source
    assert "rows={3}" in search_paths_source
    assert "min-h-20 resize-none" in search_paths_source
    assert "placeholder='Source topic used to generate this plan'" in search_paths_source
    assert "id='sheet-plan-topics'" in search_paths_source
    assert "id='sheet-plan-subreddits'" in search_paths_source
    assert "id='sheet-plan-queries'" in search_paths_source
    assert "rows={6}" in search_paths_source
    assert "rows={8}" in search_paths_source
    assert 'sheet-plan-linkedin-query-${index}' in search_paths_source
    assert "LinkedIn queries" in search_paths_source
    assert "addLinkedinQuery" in search_paths_source
    assert "removeLinkedinQuery(index)" in search_paths_source
    assert "Add query" in search_paths_source
    assert "Remove query" in search_paths_source
    assert "updateLinkedinQuery(index, event.target.value)" in search_paths_source
    assert "Query {index + 1}" in search_paths_source
    assert "<Textarea" in search_paths_source
