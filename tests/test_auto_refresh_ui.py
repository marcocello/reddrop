from pathlib import Path


def test_jobs_and_threads_use_fixed_auto_refresh_interval() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert 'AutoRefreshMenu' not in jobs
    assert 'autoRefreshSeconds' not in jobs
    assert 'AUTO_REFRESH_INTERVAL_MS = 2000' in jobs
    assert 'Auto-refresh: 2s' in jobs
    assert 'window.setInterval(() => {' in jobs
    assert '}, AUTO_REFRESH_INTERVAL_MS)' in jobs

    assert 'AutoRefreshMenu' not in threads
    assert 'autoRefreshSeconds' not in threads
    assert 'AUTO_REFRESH_INTERVAL_MS = 2000' in threads
    assert 'Auto-refresh: 2s' in threads
    assert 'window.setInterval(() => {' in threads
    assert '}, AUTO_REFRESH_INTERVAL_MS)' in threads


def test_refresh_buttons_have_visible_labels_everywhere() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    personas = Path('frontend/src/features/personas/index.tsx').read_text(encoding='utf-8')
    assert "onClick={() => void load()}" in jobs
    assert "onClick={() => void loadThreads()}" in threads
    assert "onClick={() => void loadPersonas()}" in personas
    assert "size='sm'" in jobs
    assert "size='sm'" in threads
    assert "size='sm'" in personas
    assert "Refresh\n          </Button>" in jobs
    assert "Refresh\n          </Button>" in threads
    assert "Refresh\n          </Button>" in personas
    assert "aria-label='Refresh jobs'" in jobs
    assert "aria-label='Refresh threads'" in threads
    assert "aria-label='Refresh personas'" in personas


def test_auto_refresh_pages_use_animated_refresh_icon() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "RefreshCw className='size-4 animate-spin [animation-duration:2s]'" in jobs
    assert "RefreshCw className='size-4 animate-spin [animation-duration:2s]'" in threads
