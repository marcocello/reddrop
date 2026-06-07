from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from ...services.job_config_store import JobConfigStore


def handle(
    args,
    store: JobConfigStore,
    error: Callable[[str], int],
    emit: Callable[[dict], None],
    job_name_pattern,
    id_factory: Callable[[], object] = uuid4,
) -> int:
    if not job_name_pattern.fullmatch(args.name):
        return error(
            "Invalid job name: spaces are not allowed. Use letters, numbers, '-' or '_'."
        )

    target_job_id = str(id_factory())
    job = store.upsert_job(
        job_id=target_job_id,
        name=args.name,
        job_type=getattr(args, "job_type", "search"),
        social=getattr(args, "social", "reddit"),
        auth_id=getattr(args, "auth_id", None),
        source_job_id=getattr(args, "source_job_id", None),
        topic=getattr(args, "topic", None),
        min_similarity_score=getattr(args, "min_similarity_score", None),
        active=bool(getattr(args, "active", False)),
        time_filter=getattr(args, "time_filter", None),
        subreddit_limit=getattr(args, "subreddit_limit", None),
        threads_limit=getattr(args, "threads_limit", None),
        scroll_rounds=getattr(args, "scroll_rounds", None),
        sort_by=getattr(args, "sort_by", None),
        selector_wait_timeout_seconds=getattr(
            args, "selector_wait_timeout_seconds", None
        ),
        detail_wait_timeout_seconds=getattr(args, "detail_wait_timeout_seconds", None),
        max_comment_expand_clicks=getattr(args, "max_comment_expand_clicks", None),
        detail_enrich_top_k=getattr(args, "detail_enrich_top_k", None),
        search_plan=getattr(args, "search_plan", None),
        search_query_id=getattr(args, "search_query_id", None),
        search_query_version=getattr(args, "search_query_version", None),
        replies_per_iteration=getattr(args, "replies_per_iteration", None),
        max_runtime_minutes=getattr(args, "max_runtime_minutes", 1440),
        personas=getattr(args, "personas", None),
    )
    emit(job.model_dump(mode="json", exclude_none=True))
    return 0
