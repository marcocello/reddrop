from pathlib import Path


def test_threads_table_has_job_column_with_short_id_and_name() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "title='Job'" in source
    assert 'slice(0, 8)' in source
    assert '(${row.job_name})' in source
    assert '({row.original.job_name})' in source
    assert "block font-mono text-xs font-medium leading-tight" in source
    assert "block truncate text-xs text-muted-foreground leading-tight" in source


def test_job_column_is_before_thread_column() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    job_index = source.find("title='Job'")
    thread_index = source.find("title='Thread'")
    subreddit_index = source.find("title='Subreddit'")
    status_index = source.find("title='Status'")
    similarity_index = source.find("title='Similarity'")
    assert job_index != -1
    assert thread_index != -1
    assert subreddit_index != -1
    assert status_index != -1
    assert similarity_index != -1
    assert job_index < thread_index
    assert thread_index < subreddit_index < status_index < similarity_index


def test_threads_search_placeholder_mentions_job_id() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert 'Search by thread, job id, subreddit or URL...' in source

def test_thread_sidebar_url_is_clamped_to_two_lines() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert 'WebkitLineClamp: 2' in source
    assert "parts.join('\\n')" not in source

def test_thread_cell_only_contains_two_line_title() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "{row.original.job_name} · r/{row.original.subreddit}" not in source
    assert "block max-w-[28rem] line-clamp-2 font-medium leading-snug" in source


def test_job_column_is_pinned_first_in_column_order_logic() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')
    assert "const JOB_COLUMN_ID = 'job'" in source
    assert "next = next.filter((id) => id !== JOB_COLUMN_ID)" in source
    assert "next.unshift(JOB_COLUMN_ID)" in source
