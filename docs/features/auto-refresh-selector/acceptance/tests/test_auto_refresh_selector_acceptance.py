from pathlib import Path


def test_jobs_and_threads_have_refresh_selector_options() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    for source in (jobs, threads):
        assert 'const AUTO_REFRESH_OPTIONS' in source
        assert "{ value: 0, label: 'Off', menuLabel: 'Auto-refresh off' }" in source
        assert "{ value: 30, label: 'Every 30s', menuLabel: 'Every 30s' }" in source
        assert "{ value: 60, label: 'Every 1m', menuLabel: 'Every 1m' }" in source
        assert "{ value: 300, label: 'Every 5m', menuLabel: 'Every 5m' }" in source
        assert "{ value: 900, label: 'Every 15m', menuLabel: 'Every 15m' }" in source
        assert "const [autoRefreshSeconds, setAutoRefreshSeconds] = useState<number>(30)" in source
        assert "autoRefreshSeconds <= 0" in source
        assert '}, autoRefreshSeconds * 1000)' in source


def test_auto_refresh_label_removed_and_refresh_button_kept() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    assert 'Auto-refresh:' not in jobs
    assert 'Auto-refresh:' not in threads
    assert "inline-flex h-8 items-center overflow-hidden rounded-md border" in jobs
    assert "inline-flex h-8 items-center overflow-hidden rounded-md border" in threads
    assert "aria-label='Manual refresh jobs'" in jobs
    assert "aria-label='Manual refresh threads'" in threads
    assert "aria-label='Auto-refresh jobs'" in jobs
    assert "aria-label='Auto-refresh threads'" in threads


def test_jobs_header_has_create_and_threads_has_refresh_only() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    assert 'Create job' in jobs
    assert 'Create job' not in threads
    assert jobs.index("aria-label='Manual refresh jobs'") < jobs.index('Create job')
    assert "<Plus className='size-4' />" in jobs
    assert "<Plus className='size-4' />" not in threads


def test_personas_auths_and_search_paths_use_refresh_then_create() -> None:
    personas = Path('frontend/src/features/personas/index.tsx').read_text(encoding='utf-8')
    auths = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    search_queries = Path('frontend/src/features/search-queries/index.tsx').read_text(encoding='utf-8')

    assert 'Create persona' in personas
    assert "onClick={() => void loadPersonas()}" in personas
    assert personas.index('RefreshCw') < personas.index('Create persona')
    assert "<Plus className='size-4' />" in personas

    assert 'Create auth' in auths
    assert "onClick={() => void loadAuths()}" in auths
    assert auths.index('RefreshCw') < auths.index('Create auth')
    assert "<Plus className='size-4' />" in auths

    assert 'Create search query' in search_queries
    assert "onClick={() => void load()}" in search_queries
    assert search_queries.index('RefreshCw') < search_queries.index('Create search query')
    assert "<Plus className='size-4' />" in search_queries
