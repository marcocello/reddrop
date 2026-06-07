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
    linkedin_service_cls: Any | None = None
    should_stop: Callable[[], bool] = _never_stop
    prepare_job: Callable[[PeriodicSearchJob], None] | None = None


def _resolve_social(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if value == "linkedin":
        return "linkedin"
    return "reddit"


def _normalize_plan_items(
    value: object, *, fallback: list[str] | None = None
) -> list[str]:
    items: list[str] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if normalized:
                items.append(normalized)
    if items:
        return items
    return list(fallback or [])


def _search_query_label(job: PeriodicSearchJob) -> str:
    path_id = str(getattr(job, "search_query_id", "") or "").strip()
    if not path_id:
        return "none"
    raw_version = getattr(job, "search_query_version", None)
    if raw_version is None:
        return path_id
    try:
        version = max(1, int(raw_version))
    except Exception:
        return path_id
    return f"{path_id}@v{version}"


def _search_job(job: PeriodicSearchJob, storage, deps: SearchDependencies):
    if deps.should_stop():
        raise RuntimeError("Run stopped by user request.")
    if deps.prepare_job is not None:
        deps.prepare_job(job)

    normalized_topic = str(job.topic or "").strip()
    if not normalized_topic:
        raise ValueError(
            "Search topic must be configured from a saved search query before runtime."
        )
    time_filter = str(job.time_filter or "week").strip() or "week"
    subreddit_limit = max(1, int(job.subreddit_limit or 5))
    threads_limit = max(1, int(job.threads_limit or 10))
    scroll_rounds = max(1, int(getattr(job, "scroll_rounds", None) or 15))
    sort_by = str(getattr(job, "sort_by", None) or "relevance").strip() or "relevance"
    selector_wait_timeout_seconds = max(
        1, int(getattr(job, "selector_wait_timeout_seconds", None) or 20)
    )
    detail_wait_timeout_seconds = max(
        1, int(getattr(job, "detail_wait_timeout_seconds", None) or 8)
    )
    max_comment_expand_clicks = max(
        1, int(getattr(job, "max_comment_expand_clicks", None) or 30)
    )
    detail_enrich_top_k = max(0, int(getattr(job, "detail_enrich_top_k", None) or 12))
    search_plan = (
        job.search_plan.model_dump(mode="python")
        if getattr(job, "search_plan", None) is not None
        else None
    )
    if not isinstance(search_plan, dict):
        raise ValueError("Locked search_plan is required for search runtime.")

    now = deps.now_fn()
    existing_artifact = storage.get_search(job.id)
    created_at = existing_artifact.created_at if existing_artifact else now
    existing_conversations = (
        list(existing_artifact.conversations) if existing_artifact else []
    )
    artifact = deps.search_artifact_model(
        id=job.id,
        name=job.name,
        topic=normalized_topic,
        created_at=created_at,
        updated_at=now,
        conversations=existing_conversations,
    )
    storage.save_search(artifact)

    deps.logger.info("starting: %s - %s", job.name, job.id)
    social = _resolve_social(getattr(job, "social", "reddit"))
    if social == "linkedin":
        if deps.linkedin_service_cls is None:
            raise RuntimeError("LinkedIn service is not configured.")
        discovery_service = deps.linkedin_service_cls()
    else:
        discovery_service = deps.reddit_service_cls()

    search_query = _search_query_label(job)
    if social == "linkedin":
        query = str(search_plan.get("query", "")).strip()
        if not query:
            queries = _normalize_plan_items(search_plan.get("queries"))
            query = queries[0] if queries else ""
        if not query:
            raise ValueError(
                "LinkedIn locked search plan must include at least one query."
            )
        deps.logger.info(
            "%s - searching linkedin conversations with search_query=%s query=%r "
            "(time_filter=%s, threads_limit=%s, scroll_rounds=%s, sort_by=%s, selector_wait_timeout_seconds=%s, "
            "detail_wait_timeout_seconds=%s, max_comment_expand_clicks=%s, detail_enrich_top_k=%s).",
            job.id,
            search_query,
            query,
            time_filter,
            threads_limit,
            scroll_rounds,
            sort_by,
            selector_wait_timeout_seconds,
            detail_wait_timeout_seconds,
            max_comment_expand_clicks,
            detail_enrich_top_k,
        )
    else:
        topics = _normalize_plan_items(search_plan.get("topics"))
        subreddits = _normalize_plan_items(search_plan.get("subreddits"))
        queries = _normalize_plan_items(search_plan.get("queries"))
        if not subreddits:
            raise ValueError(
                "Reddit locked search plan must include at least one subreddit."
            )
        if not queries:
            raise ValueError(
                "Reddit locked search plan must include at least one query."
            )
        deps.logger.info(
            "%s - searching reddit conversations with search_query=%s topics=%s subreddits=%s queries=%s "
            "(time_filter=%s, subreddit_limit=%s, threads_limit=%s).",
            job.id,
            search_query,
            topics,
            subreddits,
            queries,
            time_filter,
            subreddit_limit,
            threads_limit,
        )
    if social == "linkedin":
        threads = discovery_service.discover_relevant_threads(
            content=normalized_topic,
            time_filter=time_filter,
            subreddit_limit=subreddit_limit,
            threads_limit=threads_limit,
            scroll_rounds=scroll_rounds,
            sort_by=sort_by,
            selector_wait_timeout_seconds=selector_wait_timeout_seconds,
            detail_wait_timeout_seconds=detail_wait_timeout_seconds,
            max_comment_expand_clicks=max_comment_expand_clicks,
            detail_enrich_top_k=detail_enrich_top_k,
            job_id=job.id,
            stop_requested=deps.should_stop,
            search_plan=search_plan,
        )
    else:
        threads = discovery_service.discover_relevant_threads(
            content=normalized_topic,
            time_filter=time_filter,
            subreddit_limit=subreddit_limit,
            threads_limit=threads_limit,
            job_id=job.id,
            stop_requested=deps.should_stop,
            search_plan=search_plan,
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
            published_at=thread.get("published_at"),
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
    jobs = [
        job
        for job in store.list_jobs()
        if getattr(job, "job_type", "search") == "search"
    ]
    if not jobs:
        return error("Job not found")

    selected_jobs = jobs
    if args.name:
        selected_jobs = [job for job in jobs if job.name == args.name]
        if not selected_jobs:
            exists_any = store.get_job_by_name(args.name)
            if (
                exists_any is not None
                and getattr(exists_any, "job_type", "search") != "search"
            ):
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
