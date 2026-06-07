from pathlib import Path


def test_jobs_personas_auths_and_search_paths_support_row_selection_bulk_delete() -> None:
    jobs = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    personas = Path('frontend/src/features/personas/index.tsx').read_text(encoding='utf-8')
    auths = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    search_queries = Path('frontend/src/features/search-queries/index.tsx').read_text(encoding='utf-8')
    threads = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    selection_bar = Path('frontend/src/components/selection-action-bar.tsx').read_text(encoding='utf-8')

    assert 'rowSelection' in jobs
    assert 'Delete selected' in jobs
    assert 'Checkbox' in jobs
    assert 'SelectionActionBar' in jobs

    assert 'rowSelection' in personas
    assert 'Delete selected' in personas
    assert 'Checkbox' in personas
    assert 'SelectionActionBar' in personas

    assert 'rowSelection' in auths
    assert 'Delete selected' in auths
    assert 'Checkbox' in auths
    assert 'SelectionActionBar' in auths

    assert 'selectedQueryIds' in search_queries
    assert 'Delete selected' in search_queries
    assert 'Checkbox' in search_queries
    assert 'SelectionActionBar' in search_queries

    assert 'selectedConversationIds' in threads
    assert 'Delete selected' in threads
    assert 'SelectionActionBar' in threads

    assert 'fixed bottom-6 left-1/2' in selection_bar
    assert '-translate-x-1/2' in selection_bar
