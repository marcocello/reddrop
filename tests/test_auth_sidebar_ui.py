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
    assert "searchPlaceholder='Search by auth id, name, social...'" in auths_source
    assert "<div className='space-y-4'>\n          <DataTableToolbar" not in auths_source
