from pathlib import Path


def test_backend_job_model_has_type_and_source_link() -> None:
    source = Path("backend/models.py").read_text(encoding="utf-8")
    assert 'job_type: Literal["search", "reply"] = "search"' in source
    assert "source_job_id: str | None = None" in source
    assert "min_similarity_score: float = 0.35" in source


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


def test_reply_jobs_store_empty_topic() -> None:
    source = Path("backend/api/main.py").read_text(encoding="utf-8")
    assert 'topic_value = request.topic.strip() if request.job_type == "search" else ""' in source
    assert "min_similarity_score=request.min_similarity_score" in source


def test_reddit_discovery_still_logs_step_5_for_empty_candidates() -> None:
    source = Path("backend/services/reddit_service.py").read_text(encoding="utf-8")
    assert 'step 5/5: Ranking %s candidate thread(s).' in source
