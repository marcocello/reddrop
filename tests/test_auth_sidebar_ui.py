from pathlib import Path


def test_auth_sidebar_matches_jobs_personas_sheet_structure() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')

    assert "<SheetContent side='right' className='sm:max-w-xl p-0'>" in source
    assert "<div className='flex h-full flex-col'>" in source
    assert "<SheetHeader className='px-4 pt-4'>" in source
    assert "sticky top-0 z-10 mt-3 flex items-center gap-2 border-y bg-background px-4 py-3" in source
    assert "className='flex-1 space-y-4 overflow-y-auto px-4 py-4'" in source


def test_auth_sidebar_action_row_keeps_primary_controls() -> None:
    source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')

    assert 'Run LinkedIn login' in source
    assert "{saving ? 'Saving...' : 'Save changes'}" in source
    assert 'Delete' in source


def test_auth_main_vertical_spacing_matches_personas_layout_pattern() -> None:
    auths_source = Path('frontend/src/features/auths/index.tsx').read_text(encoding='utf-8')
    personas_source = Path('frontend/src/features/personas/index.tsx').read_text(encoding='utf-8')

    assert "<Main className='flex flex-1 flex-col gap-4 sm:gap-6'>" in personas_source
    assert "<Main className='flex flex-1 flex-col gap-4 sm:gap-6'>" in auths_source
    assert "searchPlaceholder='Search by auth id, username, social or status...'" in auths_source
    assert "<div className='space-y-4'>\n          <DataTableToolbar" not in auths_source


def test_jobs_main_vertical_spacing_matches_threads_layout_pattern() -> None:
    jobs_source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    threads_source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    assert "<Main className='flex flex-1 flex-col gap-4 sm:gap-6'>" in threads_source
    assert "<Main className='flex flex-1 flex-col gap-4 sm:gap-6'>" in jobs_source
    assert "searchPlaceholder='Search by id, name, topic or status...'" in jobs_source
    assert "<div className='space-y-4'>\n          <DataTableToolbar" not in jobs_source


def test_jobs_right_sidebar_uses_grouped_information_sections() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')

    assert 'Next best action' not in source
    assert 'Overview' in source
    assert 'State' in source
    assert 'Metrics' in source
    assert 'Context' in source
    assert 'Scheduling' in source
    assert 'Search configuration' in source
    assert 'Reply configuration' in source
    assert 'Run timeline' in source
    assert 'Show errors only' in source
    assert 'Show warnings only' in source
    assert 'Personas:' not in source
    assert 'Minimum similarity score:' not in source
    assert "rounded-lg border bg-muted/20 p-3" in source
    assert 'text-xs font-semibold uppercase tracking-wide text-muted-foreground' in source


def test_jobs_right_sidebar_uses_overview_and_activity_tabs() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')

    assert "<TabsTrigger value='overview'>Overview</TabsTrigger>" in source
    assert "<TabsTrigger value='activity'>Activity</TabsTrigger>" in source
    assert "<TabsContent value='overview'" in source
    assert "<TabsContent value='activity'" in source
    assert "<TabsTrigger value='configuration'>Configuration</TabsTrigger>" not in source
    assert "<TabsContent value='configuration'" not in source


def test_jobs_right_sidebar_shows_title_meta_badges_and_taller_activity_log() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')

    assert "ID {shortId(selectedJob.id)}" in source
    assert 'Type {formatJobType(selectedJob.job_type)}' in source
    assert "{formatSocial(selectedJob.social)}" in source
    assert "{selectedJob.active ? 'Active' : 'Inactive'}" in source
    assert "{formatStatusLabel(selectedJob.runtime?.status)}" in source
    assert 'max-h-[70vh]' in source


def test_threads_right_sidebar_prioritizes_post_content_and_compact_summary() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    post_index = source.find('>\n                        Post\n')
    summary_index = source.find('>\n                        Summary\n')
    link_index = source.find('>\n                        Link\n')
    assert post_index != -1
    assert summary_index != -1
    assert link_index != -1
    assert post_index < summary_index < link_index

    assert 'Next best action' not in source
    assert '<span className=\'font-medium\'>Thread ID:</span>' not in source
    assert source.count('>Overview<') == 1
    assert '>State<' not in source
    assert '>Metrics<' not in source
    assert '>Context<' not in source
    assert 'Compose reply' in source
    assert 'Activity timeline' in source
    assert 'Reply generated' in source
    assert 'Reply sent' in source
    assert "rounded-lg border bg-muted/20 p-3" in source
    assert 'text-xs font-semibold uppercase tracking-wide text-muted-foreground' in source


def test_threads_right_sidebar_uses_overview_compose_activity_tabs() -> None:
    source = Path('frontend/src/features/threads/index.tsx').read_text(encoding='utf-8')

    assert "<TabsTrigger value='overview'>Overview</TabsTrigger>" in source
    assert "<TabsTrigger value='compose'>Compose</TabsTrigger>" in source
    assert "<TabsTrigger value='activity'>Activity</TabsTrigger>" in source
    assert "<TabsContent value='overview'" in source
    assert "<TabsContent value='compose'" in source
    assert "<TabsContent value='activity'" in source
