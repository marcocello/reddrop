from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from ...models import PeriodicSearchJob
from ...services.job_config_store import JobConfigStore


def _never_stop() -> bool:
    return False


@dataclass(frozen=True)
class SearchDependencies:
    logger: Any
    search_storage_factory: Any
    reddit_service_cls: Any
    conversation_model: Any
    search_artifact_model: Any
    now_fn: Any
    should_stop: Callable[[], bool] = _never_stop


def _search_job(job: PeriodicSearchJob, storage, deps: SearchDependencies):
    if deps.should_stop():
        raise RuntimeError("Run stopped by user request.")

    normalized_topic = job.topic.strip()
    if not normalized_topic:
        raise ValueError("Topic must be configured before running autonomous analysis.")

    now = deps.now_fn()
    existing_artifact = storage.get_search(job.id)
    created_at = existing_artifact.created_at if existing_artifact else now
    artifact = deps.search_artifact_model(
        id=job.id,
        name=job.name,
        topic=normalized_topic,
        created_at=created_at,
        updated_at=now,
        conversations=[],
    )
    storage.save_search(artifact)

    deps.logger.info("starting: %s - %s", job.name, job.id)
    deps.logger.info(
        "%s - searching Reddit conversations for topic %r (time_filter=%s, subreddit_limit=%s, threads_limit=%s).",
        job.id,
        normalized_topic,
        job.time_filter,
        job.subreddit_limit,
        job.threads_limit,
    )
    threads = deps.reddit_service_cls().discover_relevant_threads(
        content=normalized_topic,
        time_filter=job.time_filter,
        subreddit_limit=job.subreddit_limit,
        threads_limit=job.threads_limit,
        job_id=job.id,
        stop_requested=deps.should_stop,
    )
    if deps.should_stop():
        raise RuntimeError("Run stopped by user request.")
    conversations = [
        deps.conversation_model(
            thread_id=str(thread.get("id", "")),
            title=str(thread.get("title", "")),
            subreddit=str(thread.get("subreddit", "")),
            url=str(thread.get("url", "")),
            score=int(thread.get("score", 0) or 0),
            num_comments=int(thread.get("num_comments", 0) or 0),
            created_utc=thread.get("created_utc"),
            selftext=str(thread.get("selftext", "") or ""),
            semantic_similarity=float(thread.get("semantic_similarity", 0.0) or 0.0),
            user_has_commented=bool(thread.get("user_has_commented", False)),
        )
        for thread in (threads or [])
    ]
    artifact = deps.search_artifact_model(
        id=job.id,
        name=job.name,
        topic=normalized_topic,
        created_at=created_at,
        updated_at=deps.now_fn(),
        conversations=conversations,
    )
    storage.save_search(artifact)
    persisted = storage.get_search(job.id)
    return persisted if persisted is not None else artifact


def handle(
    args,
    store: JobConfigStore,
    emit,
    error,
    deps: SearchDependencies,
) -> int:
    deps.logger.info("Starting search execution.")
    jobs = [job for job in store.list_jobs() if getattr(job, "job_type", "search") == "search"]
    if not jobs:
        return error("Job not found")

    selected_jobs = jobs
    if args.name:
        selected_jobs = [job for job in jobs if job.name == args.name]
        if not selected_jobs:
            exists_any = store.get_job_by_name(args.name)
            if exists_any is not None and getattr(exists_any, "job_type", "search") != "search":
                return error(f"Job is not a search job: {args.name}")
            return error(f"Job not found: {args.name}")
        deps.logger.info("Running one job by name: %s", args.name)
    else:
        deps.logger.info("Running all configured jobs: count=%s.", len(selected_jobs))

    storage = deps.search_storage_factory()
    search_results: list[dict] = []
    try:
        for job in selected_jobs:
            if deps.should_stop():
                raise RuntimeError("Run stopped by user request.")
            deps.logger.info("Running job: name=%s id=%s.", job.name, job.id)
            artifact = _search_job(job, storage, deps)
            if deps.should_stop():
                raise RuntimeError("Run stopped by user request.")
            deps.logger.info(
                "Job search completed: name=%s id=%s conversations=%s.",
                artifact.name,
                artifact.id,
                len(artifact.conversations),
            )
            search_results.append(artifact.model_dump(mode="json"))
    except (ValueError, RuntimeError) as exc:
        return error(str(exc))

    deps.logger.info("Search completed successfully. jobs=%s.", len(search_results))
    emit({"runs": search_results})
    return 0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
