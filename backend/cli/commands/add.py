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
        return error("Invalid job name: spaces are not allowed. Use letters, numbers, '-' or '_'.")

    target_job_id = str(id_factory())
    job = store.upsert_job(
        job_id=target_job_id,
        name=args.name,
        job_type=getattr(args, "job_type", "search"),
        source_job_id=getattr(args, "source_job_id", None),
        topic=args.topic,
        min_similarity_score=getattr(args, "min_similarity_score", 0.35),
        active=bool(getattr(args, "active", False)),
        time_filter=args.time_filter,
        subreddit_limit=args.subreddit_limit,
        threads_limit=args.threads_limit,
        replies_per_iteration=getattr(args, "replies_per_iteration", 3),
        max_runtime_minutes=getattr(args, "max_runtime_minutes", 1440),
        personas=getattr(args, "personas", []),
    )
    emit(job.model_dump(mode="json"))
    return 0
