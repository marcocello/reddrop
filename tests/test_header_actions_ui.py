from pathlib import Path


def _header_section(source: str) -> str:
    start = source.find('<Header fixed>')
    if start == -1:
        return source
    end = source.find('</Header>', start)
    if end == -1:
        return source[start:]
    return source[start:end]


def test_header_actions_use_refresh_then_create_with_plus_icon() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    personas = Path('frontend/src/features/personas/index.tsx').read_text(encoding='utf-8')
    auths = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    search_queries = Path('frontend/src/features/search-queries/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    jobs_header = _header_section(jobs)
    assert 'RefreshCw' in jobs_header
    assert "Create job" in jobs_header
    assert "<Plus className='size-4' />" in jobs_header
    assert jobs_header.index('RefreshCw') < jobs_header.index('Create job')

    personas_header = _header_section(personas)
    assert 'RefreshCw' in personas_header
    assert "Create persona" in personas_header
    assert "<Plus className='size-4' />" in personas_header
    assert personas_header.index('RefreshCw') < personas_header.index('Create persona')

    auths_header = _header_section(auths)
    assert 'RefreshCw' in auths_header
    assert "Create auth" in auths_header
    assert "<Plus className='size-4' />" in auths_header
    assert auths_header.index('RefreshCw') < auths_header.index('Create auth')

    search_paths_header = _header_section(search_queries)
    assert 'RefreshCw' in search_paths_header
    assert "Create search query" in search_paths_header
    assert "<Plus className='size-4' />" in search_paths_header
    assert search_paths_header.index('RefreshCw') < search_paths_header.index('Create search query')

    threads_header = _header_section(threads)
    assert 'RefreshCw' in threads_header
    assert "Create job" not in threads_header
    assert "<Plus className='size-4' />" not in threads_header
