from pathlib import Path


def test_jobs_and_threads_use_refresh_selector_with_30s_default() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    assert 'AUTO_REFRESH_INTERVAL_MS = 2000' not in jobs
    assert 'const AUTO_REFRESH_OPTIONS' in jobs
    assert "{ value: 0, label: 'Off', menuLabel: 'Auto-refresh off' }" in jobs
    assert "{ value: 30, label: 'Every 30s', menuLabel: 'Every 30s' }" in jobs
    assert "{ value: 60, label: 'Every 1m', menuLabel: 'Every 1m' }" in jobs
    assert "{ value: 300, label: 'Every 5m', menuLabel: 'Every 5m' }" in jobs
    assert "{ value: 900, label: 'Every 15m', menuLabel: 'Every 15m' }" in jobs
    assert "const [autoRefreshSeconds, setAutoRefreshSeconds] = useState<number>(30)" in jobs
    assert "autoRefreshSeconds <= 0" in jobs
    assert "aria-label='Auto-refresh jobs'" in jobs
    assert "menuLabel" in jobs

    assert 'AUTO_REFRESH_INTERVAL_MS = 2000' not in threads
    assert 'const AUTO_REFRESH_OPTIONS' in threads
    assert "{ value: 0, label: 'Off', menuLabel: 'Auto-refresh off' }" in threads
    assert "{ value: 30, label: 'Every 30s', menuLabel: 'Every 30s' }" in threads
    assert "{ value: 60, label: 'Every 1m', menuLabel: 'Every 1m' }" in threads
    assert "{ value: 300, label: 'Every 5m', menuLabel: 'Every 5m' }" in threads
    assert "{ value: 900, label: 'Every 15m', menuLabel: 'Every 15m' }" in threads
    assert "const [autoRefreshSeconds, setAutoRefreshSeconds] = useState<number>(30)" in threads
    assert "autoRefreshSeconds <= 0" in threads
    assert "aria-label='Auto-refresh threads'" in threads
    assert "menuLabel" in threads


def test_refresh_buttons_have_visible_labels_everywhere() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    personas = Path('frontend/src/features/personas/index.tsx').read_text(encoding='utf-8')
    auths = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    search_queries = Path('frontend/src/features/search-queries/index.tsx').read_text(encoding='utf-8')
    assert "onClick={() => void load()}" in jobs
    assert "onClick={() => void loadThreads()}" in threads
    assert "onClick={() => void loadPersonas()}" in personas
    assert "onClick={() => void loadAuths()}" in auths
    assert "onClick={() => void load()}" in search_queries
    assert "size='sm'" in personas
    assert "aria-label='Manual refresh jobs'" in jobs
    assert "aria-label='Manual refresh threads'" in threads
    assert "aria-label='Refresh jobs'" not in jobs
    assert "aria-label='Refresh threads'" not in threads
    assert "Refresh\n          </Button>" in personas
    assert "Refresh\n          </Button>" in auths
    assert 'Refresh' in search_queries
    assert "inline-flex h-8 items-center overflow-hidden rounded-md border" in jobs
    assert "inline-flex h-8 items-center overflow-hidden rounded-md border" in threads


def test_auto_refresh_pages_use_animated_refresh_icon() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "'size-4', autoRefreshSeconds > 0 && 'animate-spin [animation-duration:2s]'" in jobs
    assert "'size-4', autoRefreshSeconds > 0 && 'animate-spin [animation-duration:2s]'" in threads


def test_jobs_split_refresh_control_matches_create_button_height() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert "inline-flex h-8 items-center overflow-hidden rounded-md border" in jobs
    assert "className='h-full w-8 rounded-none border-r'" in jobs
    assert "className='h-full rounded-none gap-2 px-3'" in jobs


def test_jobs_buttons_order_refresh_before_create() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert jobs.index("aria-label='Manual refresh jobs'") < jobs.index('Create job')
    assert "<Plus className='size-4' />" in jobs


def test_threads_header_has_refresh_controls_without_create_action() -> None:
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert 'Create job' not in threads
    assert "aria-label='Manual refresh threads'" in threads
    assert "aria-label='Auto-refresh threads'" in threads
    assert "<Plus className='size-4' />" not in threads
