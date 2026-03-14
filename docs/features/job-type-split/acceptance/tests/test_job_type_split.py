from pathlib import Path


def test_backend_job_model_has_type_and_source_link() -> None:
    source = Path("backend/models.py").read_text(encoding="utf-8")
    assert 'job_type: Literal["search", "reply"] = "search"' in source
    assert "source_job_id: str | None = None" in source
    assert "min_similarity_score: float | None = None" in source


def test_jobs_ui_has_type_and_source_controls() -> None:
    source = Path("frontend/src/features/jobs/index.tsx").read_text(encoding="utf-8")
    assert "Job type" in source
    assert "value='search'" in source
    assert "value='reply'" in source
    assert "title='Source'" in source
    assert "Source search job" in source
    assert "Label htmlFor={`${prefix}-reply-persona`}>Persona</Label>" in source
    assert "SelectTrigger id={`${prefix}-reply-persona`}" in source
    assert "buildReplyJobName(" in source
    assert "Label htmlFor={`${prefix}-job-name`}>Name</Label>" in source
    assert "id={`${prefix}-job-name-suffix`}" in source
    assert "readOnly" in source
    assert "Final name:" in source
    assert "if (job.job_type === 'reply') return 'N/A'" in source
    assert "Label htmlFor={`${prefix}-min-similarity-score`}>Minimum similarity score</Label>" in source
    assert "{form.job_type === 'reply' ? (" in source
    assert "Label htmlFor={`${prefix}-replies-per-iteration`}>Replies per iteration</Label>" in source
    assert "Label htmlFor={`${prefix}-max-runtime-minutes`}>Run frequency</Label>" in source
    assert "SelectItem value='60'>Every 1 hour</SelectItem>" in source
    assert "SelectItem value='360'>Every 6 hours</SelectItem>" in source
    assert "SelectItem value='720'>Every 12 hours</SelectItem>" in source
    assert "SelectItem value='1440'>Every 24 hours</SelectItem>" in source


def test_reply_jobs_do_not_persist_search_only_fields() -> None:
    source = Path("backend/api/main.py").read_text(encoding="utf-8")
    assert "topic_value: str | None = None" in source
    assert "time_filter_value: str | None = None" in source
    assert "subreddit_limit_value: int | None = None" in source
    assert "threads_limit_value: int | None = None" in source
    assert "min_similarity_score=min_similarity_score_value" in source


def test_reddit_discovery_still_logs_step_5_for_empty_candidates() -> None:
    source = Path("backend/services/reddit_service.py").read_text(encoding="utf-8")
    assert 'step 5/5: Ranking %s candidate thread(s).' in source
