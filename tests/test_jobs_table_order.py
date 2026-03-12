from pathlib import Path


def test_jobs_table_puts_status_after_active() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')

    active_index = source.find("id: 'active'")
    status_index = source.find("id: 'status'")
    topic_index = source.find("accessorKey: 'topic'")

    assert active_index != -1
    assert status_index != -1
    assert topic_index != -1
    assert active_index < status_index < topic_index


def test_jobs_table_includes_required_summary_columns() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert "title='ID'" in source
    assert "title='Name'" in source
    assert "title='Active'" in source
    assert "title='Status'" in source
    assert "title='Topic'" in source
    assert "title='Last Run'" in source
    assert "title='Successful Runs'" in source
    assert "title='Threads Found'" in source
    assert "title='Threads Replied'" in source
