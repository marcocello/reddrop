from pathlib import Path


def test_draft_detail_header_wraps_actions_and_truncates_long_id() -> None:
    source = Path('frontend/src/features/search-queries/index.tsx').read_text(encoding='utf-8')

    assert "className='sticky top-0 z-10 mt-3 border-y bg-background px-4 py-3'" in source
    assert "className='flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between'" in source
    assert "className='flex min-w-0 flex-wrap items-center gap-1.5'" in source
    assert "className='max-w-[16rem] truncate bg-muted text-muted-foreground'" in source
    assert "className='flex w-full flex-wrap items-center justify-end gap-2 sm:w-auto'" in source


def test_draft_sidebar_uses_compact_source_topic_and_larger_plan_fields() -> None:
    source = Path('frontend/src/features/search-queries/index.tsx').read_text(encoding='utf-8')

    assert "id='sheet-query-topic'" in source
    assert "rows={2}" in source
    assert "id='sheet-plan-topics'" in source
    assert "id='sheet-plan-subreddits'" in source
    assert "id='sheet-plan-queries'" in source
    assert "rows={6}" in source
    assert "rows={8}" in source
    assert "id='sheet-plan-query'" in source
    assert "<Textarea" in source


def test_create_search_job_uses_only_frozen_search_path_selector() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')

    assert "Label htmlFor={`${prefix}-search-query-id`}>Search query</Label>" in source
    assert "SelectValue placeholder='Select a saved search query'" in source
    assert 'Generate search query' not in source
    assert "Label htmlFor={`${prefix}-job-topic`}>Topic</Label>" not in source
    assert "Label htmlFor={`${prefix}-reddit-plan-topics`}>Topics</Label>" not in source
    assert "Label htmlFor={`${prefix}-reddit-plan-subreddits`}>Subreddits</Label>" not in source
    assert "Label htmlFor={`${prefix}-reddit-plan-queries`}>Queries</Label>" not in source
    assert "Label htmlFor={`${prefix}-linkedin-plan-query`}>LinkedIn search query</Label>" not in source
