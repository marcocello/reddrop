from pathlib import Path


def test_jobs_context_menu_has_duplicate_action_with_icon() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert 'openDuplicateJobDialog' in source
    assert 'Duplicate' in source
    assert '<Copy className=' in source
    assert "onSelect={() => openDuplicateJobDialog(job)}" in source


def test_jobs_duplicate_opens_prefilled_create_modal() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert 'setForm({' in source
    assert 'name: duplicateName' in source
    assert "topic: job.job_type === 'reply' ? '' : job.topic" in source
    assert 'min_similarity_score: job.min_similarity_score' in source
    assert 'active: job.active' in source
    assert 'time_filter: job.time_filter' in source
    assert 'setCreateDialogOpen(true)' in source


def test_jobs_context_menu_has_icons_for_actions() -> None:
    source = Path('frontend/src/features/jobs/index.tsx').read_text(encoding='utf-8')
    assert '<Eye className=' in source
    assert '<Play className=' in source
    assert '<Square className=' in source
    assert '<Trash2 className=' in source


def test_personas_context_menu_has_duplicate_action_with_icon() -> None:
    source = Path('frontend/src/features/personas/index.tsx').read_text(encoding='utf-8')
    assert 'duplicatePersona' in source
    assert 'Duplicate' in source
    assert '<Copy className=' in source


def test_personas_context_menu_has_icons_for_actions() -> None:
    source = Path('frontend/src/features/personas/index.tsx').read_text(encoding='utf-8')
    assert '<Pencil className=' in source
    assert '<Trash2 className=' in source
