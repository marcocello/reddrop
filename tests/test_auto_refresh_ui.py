from pathlib import Path


def test_auto_refresh_menu_has_required_options() -> None:
    source = Path('frontend/src/components/auto-refresh-menu.tsx').read_text(encoding='utf-8')
    assert 'off' in source
    assert 'every 5 seconds' in source
    assert 'every 30 seconds' in source
    assert 'every 1 minute' in source
    assert 'Every 5m' not in source
    assert 'Every 15m' not in source


def test_jobs_and_threads_use_auto_refresh_menu() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert 'AutoRefreshMenu' in jobs
    assert 'autoRefreshSeconds' in jobs
    assert 'AutoRefreshMenu' in threads
    assert 'autoRefreshSeconds' in threads


def test_jobs_and_threads_refresh_buttons_are_compact_icon_buttons() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "onClick={() => void load()}" in jobs
    assert "onClick={() => void loadThreads()}" in threads
    assert "size='icon'" in jobs
    assert "size='icon'" in threads
