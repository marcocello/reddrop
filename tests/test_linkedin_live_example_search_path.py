from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.models import AuthProfile
from backend.services.linkedin_service import LinkedinService


def _load_example_job() -> dict:
    config_path = Path("backend/.reddrop/jobs_config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing jobs config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("Invalid jobs config: jobs is not a list")

    job_name = os.getenv("LIVE_LINKEDIN_JOB_NAME", "example").strip() or "example"
    for job in jobs:
        if not isinstance(job, dict):
            continue
        if str(job.get("name", "")).strip() == job_name:
            return job
    raise ValueError(f"Job not found in local config: {job_name}")


def _load_auth(auth_id: str) -> AuthProfile:
    auth_path = Path("backend/.reddrop/auth/linkedin/auths.json")
    if not auth_path.exists():
        raise FileNotFoundError(f"Missing LinkedIn auth config: {auth_path}")
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    auths = payload.get("auths", [])
    if not isinstance(auths, list):
        raise ValueError("Invalid auth config: auths is not a list")
    for raw in auths:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("id", "")).strip() == auth_id:
            return AuthProfile.model_validate(raw)
    raise ValueError(f"Auth not found: {auth_id}")


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LINKEDIN_EXAMPLE", "").strip() != "1",
    reason="Set RUN_LIVE_LINKEDIN_EXAMPLE=1 to run live LinkedIn example smoke test.",
)
def test_live_example_search_path_can_return_more_than_three_with_all_time_filter() -> None:
    job = _load_example_job()
    if str(job.get("social", "")).strip().lower() != "linkedin":
        pytest.skip("Selected live job is not LinkedIn.")

    auth_id = str(job.get("auth_id", "")).strip()
    if not auth_id:
        pytest.skip("Selected live job has no auth_id.")

    search_plan = job.get("search_plan") if isinstance(job.get("search_plan"), dict) else {}
    query = str(search_plan.get("query", "")).strip()
    if not query:
        queries = search_plan.get("queries")
        if isinstance(queries, list):
            query = next((str(item).strip() for item in queries if str(item).strip()), "")
    if not query:
        query = str(job.get("topic", "")).strip()
    if not query:
        pytest.skip("No query/topic available on selected live job.")

    auth = _load_auth(auth_id)
    service = LinkedinService(auth=auth)

    threads_limit = max(1, int(job.get("threads_limit", 50) or 50))
    scroll_rounds = max(1, int(job.get("scroll_rounds", 50) or 50))
    sort_by = str(job.get("sort_by", "relevance") or "relevance")
    configured_time_filter = str(job.get("time_filter", "week") or "week")

    rows_week = service.discover_relevant_threads(
        content=query,
        time_filter=configured_time_filter,
        threads_limit=threads_limit,
        scroll_rounds=scroll_rounds,
        sort_by=sort_by,
        search_plan={"query": query},
        job_id="live-example-week",
    )
    rows_all = service.discover_relevant_threads(
        content=query,
        time_filter="all",
        threads_limit=threads_limit,
        scroll_rounds=scroll_rounds,
        sort_by=sort_by,
        search_plan={"query": query},
        job_id="live-example-all",
    )

    assert len(rows_week) >= 1
    assert len(rows_all) >= len(rows_week)
    assert len(rows_all) > 3
