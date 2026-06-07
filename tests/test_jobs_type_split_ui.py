from pathlib import Path


def test_jobs_page_has_job_type_controls_for_create_form() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "Job type" in source
    assert "value='search'" in source
    assert "value='reply'" in source
    assert "source_job_id" in source
    assert "Label htmlFor={`${prefix}-job-type`}>Job type</Label>" in source
    assert "Label htmlFor={`${prefix}-job-name`}>Name</Label>" in source
    assert source.index("Label htmlFor={`${prefix}-job-type`}>Job type</Label>") < source.index(
        "Label htmlFor={`${prefix}-job-name`}>Name</Label>"
    )


def test_modal_does_not_show_active_selector() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "Label htmlFor={`${prefix}-job-active`}>Active</Label>" not in source


def test_jobs_table_and_sheet_show_job_type_and_source() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "title='Type'" in source
    assert "title='Source'" in source
    assert "Source search job" in source
    assert "Reply jobs read threads from this linked Search job." in source


def test_job_type_badge_uses_grey_style() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "bg-muted text-muted-foreground" in source


def test_job_type_uses_icons_in_table_select_and_sheet() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "function jobTypeIcon(value: JobType) {" in source
    assert "return value === 'reply' ? MessageSquareReply : Search" in source
    assert "<Search className='size-3.5' />" in source
    assert "<MessageSquareReply className='size-3.5' />" in source
    assert "<span className='inline-flex items-center gap-1.5'>" in source


def test_reply_form_uses_dropdown_for_persona_selection() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "Label htmlFor={`${prefix}-reply-persona`}>Persona</Label>" in source
    assert "SelectTrigger id={`${prefix}-reply-persona`}" in source
    assert "Select personas" not in source


def test_reply_name_is_prefixed_from_source_job() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "buildReplyJobName(" in source
    assert "reply_" in source
    assert "sourceSearchNameById.get(value)" in source


def test_reply_name_uses_editable_prefix_with_locked_suffix() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "Label htmlFor={`${prefix}-job-name`}>Name</Label>" in source
    assert "id={`${prefix}-job-name-suffix`}" in source
    assert "readOnly" in source
    assert "flex items-center gap-2" in source
    assert "Final name:" in source
    assert "Name prefix" not in source


def test_reply_form_has_minimum_similarity_score_input() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "Label htmlFor={`${prefix}-min-similarity-score`}>Minimum similarity score</Label>" in source
    assert "id={`${prefix}-min-similarity-score`}" in source
    assert "step='0.01'" in source


def test_reply_modal_places_source_before_name() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert source.index("Label htmlFor={`${prefix}-source-job-id`}>Source search job</Label>") < source.index(
        "Label htmlFor={`${prefix}-job-name`}>Name</Label>"
    )


def test_reply_jobs_show_na_topic_in_table_and_sheet() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "function displayJobTopic(job: Pick<Job, 'job_type' | 'topic'>): string {" in source
    assert "if (job.job_type === 'reply') return 'N/A'" in source
    assert "const topicLabel = displayJobTopic(row.original)" in source
    assert "selectedJob.job_type === 'reply' ? 'N/A' : (selectedJob.topic ?? '-')" in source


def test_search_form_hides_replies_per_iteration_field() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "Label htmlFor={`${prefix}-replies-per-iteration`}>Replies per iteration</Label>" in source
    assert "{form.job_type === 'reply' ? (" in source
    assert "id={`${prefix}-replies-per-iteration`}" in source


def test_run_interval_uses_simple_frequency_dropdown() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "Label htmlFor={`${prefix}-max-runtime-minutes`}>Run frequency</Label>" in source
    assert "SelectTrigger id={`${prefix}-max-runtime-minutes`}" in source
    assert "SelectItem value='60'>Every 1 hour</SelectItem>" in source
    assert "SelectItem value='360'>Every 6 hours</SelectItem>" in source
    assert "SelectItem value='720'>Every 12 hours</SelectItem>" in source
    assert "SelectItem value='1440'>Every 24 hours</SelectItem>" in source
    assert "Run interval minutes" not in source
