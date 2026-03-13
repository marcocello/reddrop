from pathlib import Path


def test_jobs_and_threads_use_fixed_auto_refresh_interval() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert 'AutoRefreshMenu' not in jobs
    assert 'autoRefreshSeconds' not in jobs
    assert 'AUTO_REFRESH_INTERVAL_MS = 2000' in jobs
    assert 'window.setInterval(() => {' in jobs
    assert '}, AUTO_REFRESH_INTERVAL_MS)' in jobs

    assert 'AutoRefreshMenu' not in threads
    assert 'autoRefreshSeconds' not in threads
    assert 'AUTO_REFRESH_INTERVAL_MS = 2000' in threads
    assert 'window.setInterval(() => {' in threads
    assert '}, AUTO_REFRESH_INTERVAL_MS)' in threads


def test_jobs_and_threads_refresh_buttons_are_compact_icon_buttons() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "onClick={() => void load()}" in jobs
    assert "onClick={() => void loadThreads()}" in threads
    assert "size='icon'" in jobs
    assert "size='icon'" in threads
