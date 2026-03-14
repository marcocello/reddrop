from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Support both import styles:
# - python -m uvicorn backend.api.main:app  (repo root cwd)
# - python -m uvicorn api.main:app          (backend/ cwd)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.api.job_runtime import JobRuntimeManager
from backend.cli.commands import add as add_command
from backend.cli.commands import reply as reply_command
from backend.cli.commands import search as search_command
from backend.cli.commands import send as send_command
from backend.config.log import get_app_log_config, setup_logging
from backend.models import (
    AuthProfile,
    Conversation,
    PeriodicSearchJob,
    PersonaProfile,
    SearchArtifact,
)
from backend.services.auth_store import AuthStore
from backend.services.job_config_store import JobConfigStore
from backend.services.linkedin_service import LinkedinService
from backend.services.llm_service import llm_service
from backend.services.persona_store import PersonaStore
from backend.services.reply_generation_service import ReplyGenerationService
from backend.services.reddit_service import RedditService
from backend.services.search_storage import SearchStorage
from backend.services.settings_store import SettingsStore

DEFAULT_HOME = _BACKEND_ROOT / ".reddrop"
DEFAULT_JOBS_CONFIG = DEFAULT_HOME / "jobs_config.json"
DEFAULT_RUNS_DIR = DEFAULT_HOME / "runs"
DEFAULT_TIME_FILTER = "week"
DEFAULT_SUBREDDIT_LIMIT = 5
DEFAULT_THREADS_LIMIT = 10
DEFAULT_REPLIES_PER_ITERATION = 3
DEFAULT_MIN_SIMILARITY_SCORE = 0.35
DEFAULT_MAX_RUNTIME_MINUTES = 1440
JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
logger = setup_logging(__name__)


def _resolve_social(raw: object) -> Literal["reddit", "linkedin"]:
    value = str(raw or "").strip().lower()
    if value == "linkedin":
        return "linkedin"
    return "reddit"


def _slug_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    token = re.sub(r"-+", "-", token).strip("-")
    return token


def _default_linkedin_user_data_dir(name: str, auth_id: str) -> str:
    name_part = _slug_token(name)[:24] or "linkedin"
    id_part = auth_id.strip()[:8] or "auth"
    return f".reddrop/auth/linkedin/{name_part}-{id_part}"


def _resolve_linkedin_user_data_dir(raw_dir: str) -> str:
    raw = (raw_dir or "").strip() or ".reddrop/auth/linkedin"
    candidate = Path(raw).expanduser()
    resolved = candidate if candidate.is_absolute() else (_BACKEND_ROOT / candidate)
    try:
        resolved = resolved.resolve()
    except Exception:
        pass

    home_reddrop = (Path.home() / ".reddrop").resolve()
    if resolved == home_reddrop or home_reddrop in resolved.parents:
        try:
            relative_tail = resolved.relative_to(home_reddrop)
        except Exception:
            relative_tail = Path("auth/linkedin")
        resolved = (_BACKEND_ROOT / ".reddrop" / relative_tail).resolve()
    return str(resolved)


app = FastAPI(title="Reddrop API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
runtime_manager = JobRuntimeManager()
_threads_cache_lock = Lock()
_threads_cache_signature: tuple[tuple[str, int, int], ...] | None = None
_threads_cache_rows: list[dict[str, Any]] = []


class AddJobRequest(BaseModel):
    name: str
    job_type: Literal["search", "reply"] = "search"
    social: Literal["reddit", "linkedin"] = "reddit"
    auth_id: str | None = None
    source_job_id: str | None = None
    topic: str | None = None
    active: bool | None = None
    time_filter: str | None = None
    subreddit_limit: int | None = None
    threads_limit: int | None = None
    replies_per_iteration: int | None = None
    min_similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_runtime_minutes: int = DEFAULT_MAX_RUNTIME_MINUTES
    personas: list[str] | None = None


class SearchRequest(BaseModel):
    name: str | None = None


class ReplyRequest(BaseModel):
    search_file: str
    personas: str
    replies: int = 3
    min_similarity: float = Field(default=DEFAULT_MIN_SIMILARITY_SCORE, ge=0.0, le=1.0)


class SendRequest(BaseModel):
    search_file: str | None = None
    replies_file: str | None = None


class ThreadReplyRequest(BaseModel):
    persona: str
    reply: str | None = None


class PersonaCatalogResponse(BaseModel):
    personas: list[PersonaProfile]
    source: str | None = None


class UpsertPersonasRequest(BaseModel):
    personas: list[PersonaProfile]


class AuthCatalogResponse(BaseModel):
    auths: list[AuthProfile]
    source: str | None = None


class UpsertAuthsRequest(BaseModel):
    auths: list[AuthProfile]


class AuthValidationResult(BaseModel):
    auth_id: str
    social: Literal["reddit", "linkedin"]
    status: Literal["ok", "invalid"]


class AuthValidationResultsResponse(BaseModel):
    results: list[AuthValidationResult]


class JobActionResponse(BaseModel):
    name: str
    active: bool | None = None
    status: str
    last_run_status: str | None = None
    successful_runs: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    last_error: str | None = None
    last_output: str | None = None
    logs: list[dict[str, str]] = Field(default_factory=list)


class OpenRouterSettings(BaseModel):
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "x-ai/grok-4.1-fast"
    http_referer: str = ""
    x_title: str = "reddrop"
    timeout_seconds: str = "20"


class SettingsPayload(BaseModel):
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)


class _JobBoundLogger:
    def __init__(self, *, job_name: str, runtime: JobRuntimeManager) -> None:
        self._job_name = job_name
        self._runtime = runtime

    @staticmethod
    def _render(message: object, args: tuple[object, ...]) -> str:
        template = str(message)
        if not args:
            return template
        try:
            return template % args
        except Exception:
            return " ".join([template, *[str(item) for item in args]])

    def debug(self, message: object, *args: object) -> None:
        self._runtime.append_log(self._job_name, f"DEBUG: {self._render(message, args)}")

    def info(self, message: object, *args: object) -> None:
        self._runtime.append_log(self._job_name, f"INFO: {self._render(message, args)}")

    def warning(self, message: object, *args: object) -> None:
        self._runtime.append_log(self._job_name, f"WARN: {self._render(message, args)}")

    def error(self, message: object, *args: object) -> None:
        self._runtime.append_log(self._job_name, f"ERROR: {self._render(message, args)}")

    def exception(self, message: object, *args: object) -> None:
        self._runtime.append_log(self._job_name, f"ERROR: {self._render(message, args)}")


def _job_store() -> JobConfigStore:
    return JobConfigStore(DEFAULT_JOBS_CONFIG)


def _search_storage() -> SearchStorage:
    return SearchStorage(DEFAULT_RUNS_DIR)


def _settings_store() -> SettingsStore:
    return SettingsStore(DEFAULT_HOME / "settings.yaml")


def _auth_store() -> AuthStore:
    return AuthStore(DEFAULT_HOME / "auth", legacy_candidates=[DEFAULT_HOME / "auths.json"])


def _persona_store() -> PersonaStore:
    try:
        return PersonaStore([DEFAULT_HOME / "personas.json"])
    except TypeError:
        # Test doubles may expose a zero-arg constructor only.
        return PersonaStore()


def _auth_by_ref(auth_ref: str) -> AuthProfile:
    normalized = auth_ref.strip()
    if not normalized:
        raise HTTPException(status_code=404, detail="Auth not found")
    store = _auth_store()
    for auth in store.list_auths():
        if (
            auth.id == normalized
            or auth.name == normalized
            or auth.reddit_username.strip() == normalized
        ):
            return auth
    raise HTTPException(status_code=404, detail=f"Auth not found: {auth_ref}")


def _apply_auth_to_env(auth: AuthProfile) -> None:
    social = _resolve_social(auth.social)
    if social == "reddit":
        client_id = auth.reddit_client_id.strip()
        client_secret = auth.reddit_client_secret.strip()
        user_agent = auth.reddit_user_agent.strip()
        if client_id:
            os.environ["REDDIT_CLIENT_ID"] = client_id
        if client_secret:
            os.environ["REDDIT_CLIENT_SECRET"] = client_secret
        if user_agent:
            os.environ["REDDIT_USER_AGENT"] = user_agent
        elif not (os.getenv("REDDIT_USER_AGENT") or "").strip():
            os.environ["REDDIT_USER_AGENT"] = "Reddrop:v1.0"
        os.environ["REDDIT_USERNAME"] = auth.reddit_username.strip()
        os.environ["REDDIT_PASSWORD"] = auth.reddit_password.strip()
        return

    os.environ["LINKEDIN_EMAIL"] = auth.linkedin_email.strip()
    os.environ["LINKEDIN_PASSWORD"] = auth.linkedin_password.strip()
    os.environ["LINKEDIN_HEADLESS"] = auth.linkedin_headless.strip() or "true"
    os.environ["LINKEDIN_USER_DATA_DIR"] = _resolve_linkedin_user_data_dir(auth.linkedin_user_data_dir.strip())
    os.environ["LINKEDIN_WAIT_SECONDS"] = auth.linkedin_wait_seconds.strip() or "120"


def _auth_for_job(job: PeriodicSearchJob) -> AuthProfile | None:
    auth_id = (job.auth_id or "").strip()
    if not auth_id:
        return None
    return _auth_store().get_auth(auth_id)


def _prepare_auth_for_job(job: PeriodicSearchJob) -> None:
    auth = _auth_for_job(job)
    if auth is None:
        raise RuntimeError(f"Auth not found for job {job.name}: {job.auth_id}")
    if _resolve_social(auth.social) != _resolve_social(job.social):
        raise RuntimeError(f"Auth social mismatch for job {job.name}: {auth.social} != {job.social}")
    _apply_auth_to_env(auth)


def _error(message: str) -> int:
    logger.error("%s", message)
    raise HTTPException(status_code=400, detail=message)


def _job_by_ref(job_ref: str) -> PeriodicSearchJob:
    store = _job_store()
    by_id = store.get_job_by_id(job_ref)
    if by_id is not None:
        return by_id
    by_name = store.get_job_by_name(job_ref)
    if by_name is not None:
        return by_name
    raise HTTPException(status_code=404, detail=f"Job not found: {job_ref}")


def _job_status_payload(job: PeriodicSearchJob) -> dict[str, Any]:
    runtime = runtime_manager.status(job.name)
    artifact_job_id = job.id
    if job.job_type == "reply" and job.source_job_id:
        artifact_job_id = job.source_job_id
    search = _search_storage().get_search(artifact_job_id)
    conversations = search.conversations if search is not None else []
    threads_found_total = len(conversations)
    threads_replied_total = sum(1 for item in conversations if bool(item.user_has_commented))
    payload = job.model_dump(mode="json", exclude_none=True)
    payload["runtime"] = runtime
    payload["last_run_status"] = str(runtime.get("last_run_status", "never"))
    payload["successful_runs"] = int(runtime.get("successful_runs", 0) or 0)
    payload["threads_found_total"] = threads_found_total
    payload["threads_replied_total"] = threads_replied_total
    return payload


def _register_runtime_job(job: PeriodicSearchJob) -> None:
    runtime_manager.register(
        job.name,
        interval_minutes=max(1, int(job.max_runtime_minutes)),
        runner=_build_job_runner(job.name),
    )


def _resolve_reply_source_search_file(job: PeriodicSearchJob, store: JobConfigStore) -> str:
    source_job_id = (job.source_job_id or "").strip()
    if not source_job_id:
        raise RuntimeError(f"Reply job {job.name} is missing source_job_id.")
    source_job = store.get_job_by_id(source_job_id)
    if source_job is None:
        raise RuntimeError(f"Reply job source not found: {source_job_id}")
    if source_job.job_type != "search":
        raise RuntimeError(f"Reply job source must be a search job: {source_job.name}")
    source_path = _search_storage().path_for_job(source_job.id)
    if not source_path.exists():
        raise RuntimeError(f"Source search artifact not found. Run Search job first: {source_job.name}")
    return source_path.name


def _build_job_runner(job_name: str):
    def _stop_if_requested() -> None:
        if runtime_manager.should_stop(job_name):
            raise RuntimeError("Run stopped by user request.")

    def _runner(set_phase) -> str:
        store = _job_store()
        job = store.get_job_by_name(job_name)
        if job is None:
            raise RuntimeError(f"Job not found: {job_name}")
        should_stop = lambda: runtime_manager.should_stop(job_name)
        job_logger = _JobBoundLogger(job_name=job_name, runtime=runtime_manager)

        if job.job_type == "search":
            _stop_if_requested()
            set_phase("searching")
            search_file = _run_search_for_job(job.name, should_stop=should_stop, job_logger=job_logger)
            _stop_if_requested()
            return f"search_file={search_file}"

        if job.job_type == "reply":
            personas = [item.strip() for item in (job.personas or []) if item and item.strip()]
            if not personas:
                raise RuntimeError(f"Reply job {job.name} requires at least one persona.")
            source_search_file = _resolve_reply_source_search_file(job, store)
            _stop_if_requested()
            set_phase("replying")
            _run_reply_for_job(
                search_file=source_search_file,
                persona=personas[0],
                replies=max(1, int(job.replies_per_iteration or DEFAULT_REPLIES_PER_ITERATION)),
                min_similarity=max(
                    0.0,
                    min(1.0, float(job.min_similarity_score if job.min_similarity_score is not None else DEFAULT_MIN_SIMILARITY_SCORE)),
                ),
                should_stop=should_stop,
                job_logger=job_logger,
            )
            _stop_if_requested()
            return f"reply_file={source_search_file}"

        raise RuntimeError(f"Unsupported job type: {job.job_type}")

    return _runner


def _run_search_for_job(job_name: str, *, should_stop=None, job_logger=None) -> str:
    args = argparse.Namespace(name=job_name)
    deps = search_command.SearchDependencies(
        logger=job_logger or logger,
        search_storage_factory=_search_storage,
        reddit_service_cls=RedditService,
        linkedin_service_cls=LinkedinService,
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=search_command.utc_now,
        should_stop=should_stop or (lambda: False),
        prepare_job=_prepare_auth_for_job,
    )
    payload: dict[str, Any] = {}
    errors: list[str] = []

    def _emit(data: dict) -> None:
        payload.update(data)

    def _collect_error(message: str) -> int:
        errors.append(message)
        return 1

    rc = search_command.handle(args, _job_store(), _emit, _collect_error, deps)
    if rc != 0:
        raise RuntimeError(errors[-1] if errors else f"Search failed for job {job_name}")

    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise RuntimeError(f"Search produced no run payload for job {job_name}")

    job_id = str(runs[0].get("id", "")).strip()
    if not job_id:
        raise RuntimeError(f"Search payload missing job id for job {job_name}")
    return _search_storage().path_for_job(job_id).name


def _run_reply_for_job(
    *,
    search_file: str,
    persona: str,
    replies: int,
    min_similarity: float = DEFAULT_MIN_SIMILARITY_SCORE,
    should_stop=None,
    job_logger=None,
) -> None:
    args = argparse.Namespace(
        search_file=search_file,
        personas=persona,
        replies=replies,
        min_similarity=min_similarity,
    )
    deps = reply_command.ReplyDependencies(
        logger=job_logger or logger,
        runs_dir=DEFAULT_RUNS_DIR,
        persona_store_cls=_persona_store,
        reply_generation_service_cls=ReplyGenerationService,
        search_storage_cls=SearchStorage,
        search_artifact_model=SearchArtifact,
        conversation_model=Conversation,
        now_fn=reply_command.utc_now,
        should_stop=should_stop or (lambda: False),
    )
    errors: list[str] = []
    rc = reply_command.handle(args, lambda _data: None, lambda message: errors.append(message) or 1, deps)
    if rc != 0:
        raise RuntimeError(errors[-1] if errors else f"Reply failed for {search_file}")


def _run_send_for_job(*, search_file: str, should_stop=None, job_logger=None) -> None:
    args = argparse.Namespace(search_file=search_file)
    deps = send_command.SendDependencies(
        logger=job_logger or logger,
        runs_dir=DEFAULT_RUNS_DIR,
        search_storage_cls=SearchStorage,
        reddit_service_cls=RedditService,
        linkedin_service_cls=LinkedinService,
        job_store_factory=_job_store,
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=send_command.utc_now,
        should_stop=should_stop or (lambda: False),
        prepare_job=_prepare_auth_for_job,
    )
    errors: list[str] = []
    rc = send_command.handle(args, lambda _data: None, lambda message: errors.append(message) or 1, deps)
    if rc != 0:
        raise RuntimeError(errors[-1] if errors else f"Send failed for {search_file}")


def _thread_key_match(item: Conversation, *, thread_id: str, subreddit: str) -> bool:
    return item.thread_id == thread_id and item.subreddit == subreddit


def _thread_payload(
    *,
    search_job: PeriodicSearchJob,
    conversation: Conversation,
    reply: str = "",
    reply_job: PeriodicSearchJob | None = None,
) -> dict[str, Any]:
    persona_names: list[str] = []
    if reply_job is not None:
        seen: set[str] = set()
        for item in (reply_job.personas or []):
            value = item.strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            persona_names.append(value)

    return {
        "job_id": search_job.id,
        "job_name": search_job.name,
        "social": _resolve_social(getattr(search_job, "social", "reddit")),
        "topic": search_job.topic or "",
        "search_job_id": search_job.id,
        "search_job_name": search_job.name,
        "reply_job_id": None if reply_job is None else reply_job.id,
        "reply_job_name": None if reply_job is None else reply_job.name,
        "reply_personas": persona_names,
        "thread_id": conversation.thread_id,
        "subreddit": conversation.subreddit,
        "title": conversation.title,
        "url": conversation.url,
        "score": conversation.score,
        "num_comments": conversation.num_comments,
        "created_utc": conversation.created_utc,
        "selftext": conversation.selftext,
        "semantic_similarity": conversation.semantic_similarity,
        "user_has_commented": bool(conversation.user_has_commented),
        "reply": reply or conversation.reply,
        "has_reply": bool((reply or conversation.reply).strip()),
    }


def _reply_jobs_by_source_from_jobs(jobs: list[PeriodicSearchJob]) -> dict[str, list[PeriodicSearchJob]]:
    mapping: dict[str, list[PeriodicSearchJob]] = {}
    for job in jobs:
        if job.job_type != "reply":
            continue
        source_id = (job.source_job_id or "").strip()
        if not source_id:
            continue
        mapping.setdefault(source_id, []).append(job)

    for source_id, reply_jobs in mapping.items():
        mapping[source_id] = sorted(
            reply_jobs,
            key=lambda item: (not item.active, item.name.lower(), item.id),
        )
    return mapping


def _reply_jobs_by_source_id() -> dict[str, list[PeriodicSearchJob]]:
    return _reply_jobs_by_source_from_jobs(_job_store().list_jobs())


def _collect_reply_personas(reply_jobs: list[PeriodicSearchJob]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for job in reply_jobs:
        for raw in (job.personas or []):
            value = raw.strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            names.append(value)
    return names


def _resolve_thread_job_context(job_ref: str) -> tuple[PeriodicSearchJob, PeriodicSearchJob | None]:
    store = _job_store()
    target = _job_by_ref(job_ref)

    if target.job_type == "reply":
        source_id = (target.source_job_id or "").strip()
        if not source_id:
            raise HTTPException(status_code=400, detail=f"Reply job {target.name} is missing source_job_id.")
        source_job = store.get_job_by_id(source_id)
        if source_job is None:
            raise HTTPException(status_code=404, detail=f"Source search job not found: {source_id}")
        if source_job.job_type != "search":
            raise HTTPException(status_code=400, detail=f"Source job must be search type: {source_job.name}")
        return source_job, target

    linked_reply_jobs = _reply_jobs_by_source_from_jobs(store.list_jobs()).get(target.id, [])
    return target, (linked_reply_jobs[0] if linked_reply_jobs else None)


def _resolve_artifact_path(prefix: str, artifact_name: str) -> Path:
    if "/" in artifact_name or "\\" in artifact_name:
        raise HTTPException(status_code=400, detail="Invalid artifact name.")
    expected_prefix = f"{prefix}_"
    if not artifact_name.startswith(expected_prefix) or not artifact_name.endswith(".json"):
        raise HTTPException(status_code=400, detail=f"Artifact must match {expected_prefix}*.json")
    path = DEFAULT_RUNS_DIR / artifact_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_name}")
    return path


def _read_search_artifact(path: Path) -> SearchArtifact | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SearchArtifact.model_validate(payload)
    except Exception:
        return None


def _thread_sort_key(item: dict[str, Any]) -> tuple[float, int, str, str]:
    return (
        float(item.get("semantic_similarity", 0.0) or 0.0),
        int(item.get("score", 0) or 0),
        str(item.get("thread_id", "")),
        str(item.get("subreddit", "")),
    )


def _runs_snapshot_signature() -> tuple[tuple[str, int, int], ...]:
    DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot: list[tuple[str, int, int]] = []
    try:
        jobs_stat = DEFAULT_JOBS_CONFIG.stat()
        snapshot.append(("__jobs_config__", int(jobs_stat.st_mtime_ns), int(jobs_stat.st_size)))
    except OSError:
        snapshot.append(("__jobs_config__", 0, 0))
    for search_path in sorted(DEFAULT_RUNS_DIR.glob("search_*.json"), key=lambda item: item.name):
        try:
            stat = search_path.stat()
        except OSError:
            continue
        snapshot.append((search_path.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(snapshot)


def _build_threads_cache_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reply_jobs_by_source = _reply_jobs_by_source_id()
    jobs_by_id = {job.id: job for job in _job_store().list_jobs()}

    for search_path in sorted(DEFAULT_RUNS_DIR.glob("search_*.json"), key=lambda item: item.name):
        artifact = _read_search_artifact(search_path)
        if artifact is None:
            continue

        linked_reply_jobs = reply_jobs_by_source.get(artifact.id, [])
        primary_reply_job = linked_reply_jobs[0] if linked_reply_jobs else None
        search_job = jobs_by_id.get(artifact.id)
        social = _resolve_social(getattr(search_job, "social", "reddit"))
        reply_personas = _collect_reply_personas(linked_reply_jobs)
        reply_job_ids = [item.id for item in linked_reply_jobs]
        reply_job_names = [item.name for item in linked_reply_jobs]

        for conversation in artifact.conversations:
            reply_text = conversation.reply
            has_reply_value = bool(reply_text.strip())
            rows.append(
                {
                    "job_id": artifact.id,
                    "job_name": artifact.name,
                    "social": social,
                    "topic": artifact.topic,
                    "search_job_id": artifact.id,
                    "search_job_name": artifact.name,
                    "reply_job_id": None if primary_reply_job is None else primary_reply_job.id,
                    "reply_job_name": None if primary_reply_job is None else primary_reply_job.name,
                    "linked_reply_job_ids": reply_job_ids,
                    "linked_reply_job_names": reply_job_names,
                    "reply_personas": reply_personas,
                    "search_artifact": search_path.name,
                    "search_updated_at": artifact.updated_at,
                    "thread_id": conversation.thread_id,
                    "subreddit": conversation.subreddit,
                    "title": conversation.title,
                    "url": conversation.url,
                    "score": int(conversation.score),
                    "num_comments": int(conversation.num_comments),
                    "created_utc": conversation.created_utc,
                    "selftext": conversation.selftext,
                    "semantic_similarity": float(conversation.semantic_similarity or 0.0),
                    "user_has_commented": bool(conversation.user_has_commented),
                    "reply": reply_text,
                    "has_reply": has_reply_value,
                }
            )
    return rows


def _cached_thread_rows() -> list[dict[str, Any]]:
    global _threads_cache_signature, _threads_cache_rows
    signature = _runs_snapshot_signature()
    with _threads_cache_lock:
        if _threads_cache_signature == signature:
            return _threads_cache_rows

    rebuilt_rows = _build_threads_cache_rows()

    with _threads_cache_lock:
        _threads_cache_signature = signature
        _threads_cache_rows = rebuilt_rows
        return _threads_cache_rows


def _thread_row_matches_filters(
    item: dict[str, Any],
    *,
    job_id: str,
    job_name: str,
    subreddit: str,
    min_similarity: float | None,
    only_open: bool,
    has_reply: bool | None,
) -> bool:
    if job_id and str(item.get("job_id", "")) != job_id:
        return False
    if job_name and str(item.get("job_name", "")) != job_name:
        return False
    if subreddit and str(item.get("subreddit", "")) != subreddit:
        return False

    similarity = float(item.get("semantic_similarity", 0.0) or 0.0)
    if min_similarity is not None and similarity < float(min_similarity):
        return False

    user_has_commented = bool(item.get("user_has_commented"))
    has_reply_value = bool(item.get("has_reply"))
    if only_open and user_has_commented:
        return False
    if has_reply is True and not has_reply_value:
        return False
    if has_reply is False and has_reply_value:
        return False
    return True


def _list_threads_from_artifacts(
    *,
    job_id: str | None,
    job_name: str | None,
    subreddit: str | None,
    min_similarity: float | None,
    only_open: bool,
    has_reply: bool | None,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    normalized_job_id = (job_id or "").strip()
    normalized_job_name = (job_name or "").strip()
    normalized_subreddit = (subreddit or "").strip()
    max_ranked = offset + limit

    matched_total = 0
    heap: list[tuple[tuple[float, int, str, str], int, dict[str, Any]]] = []

    for item in _cached_thread_rows():
        if not _thread_row_matches_filters(
            item,
            job_id=normalized_job_id,
            job_name=normalized_job_name,
            subreddit=normalized_subreddit,
            min_similarity=min_similarity,
            only_open=only_open,
            has_reply=has_reply,
        ):
            continue

        matched_total += 1
        if max_ranked <= 0:
            continue

        sort_key = _thread_sort_key(item)
        entry = (sort_key, matched_total, item)
        if len(heap) < max_ranked:
            heapq.heappush(heap, entry)
            continue
        if sort_key > heap[0][0]:
            heapq.heapreplace(heap, entry)

    ranked = [item[2] for item in sorted(heap, key=lambda row: (row[0], row[1]), reverse=True)]
    return ranked[offset : offset + limit], matched_total


def _list_artifacts(prefix: str) -> dict[str, Any]:
    DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DEFAULT_RUNS_DIR.glob(f"{prefix}_*.json"), key=lambda item: item.name)
    artifacts: list[dict[str, str | int]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            conversations = payload.get("conversations", [])
            artifacts.append(
                {
                    "name": path.name,
                    "job_id": str(payload.get("id", "")),
                    "job_name": str(payload.get("name", "")),
                    "topic": str(payload.get("topic", "")),
                    "updated_at": str(payload.get("updated_at", "")),
                    "conversations_count": len(conversations) if isinstance(conversations, list) else 0,
                }
            )
        except Exception:
            artifacts.append(
                {
                    "name": path.name,
                    "job_id": "",
                    "job_name": "",
                    "topic": "",
                    "updated_at": "",
                    "conversations_count": 0,
                }
            )
    return {"artifacts": artifacts}


def _persona_catalog_payload(store: PersonaStore) -> dict[str, Any]:
    source = store.resolve_file()
    personas = [item.model_dump(mode="json") for item in store.list_personas()]
    return {"personas": personas, "source": None if source is None else str(source)}


def _auth_catalog_payload(store: AuthStore) -> dict[str, Any]:
    source = store.resolve_file()
    auths: list[dict[str, Any]] = []
    for item in store.list_auths():
        payload = item.model_dump(mode="json")
        payload["status"] = _auth_status(item)
        auths.append(payload)
    return {"auths": auths, "source": None if source is None else str(source)}


def _auth_validation_payload(auth: AuthProfile) -> dict[str, str]:
    social = _resolve_social(auth.social)
    valid = False
    if social == "reddit":
        valid = RedditService.validate_credentials(
            client_id=(auth.reddit_client_id.strip() or (os.getenv("REDDIT_CLIENT_ID") or "").strip()),
            client_secret=(auth.reddit_client_secret.strip() or (os.getenv("REDDIT_CLIENT_SECRET") or "").strip()),
            user_agent=(
                auth.reddit_user_agent.strip()
                or (os.getenv("REDDIT_USER_AGENT") or "").strip()
                or "Reddrop:v1.0"
            ),
            username=auth.reddit_username.strip(),
            password=auth.reddit_password.strip(),
        )
    else:
        resolved_dir = _resolve_linkedin_user_data_dir(auth.linkedin_user_data_dir.strip())
        validator = getattr(LinkedinService, "validate_saved_session", None)
        if callable(validator):
            try:
                valid = bool(validator(user_data_dir=resolved_dir))
            except Exception:
                valid = False
        else:
            profile_dir = Path(resolved_dir)
            valid = profile_dir.exists() and profile_dir.is_dir()

    status: Literal["ok", "invalid"] = "ok" if valid else "invalid"
    return {
        "auth_id": auth.id,
        "social": social,
        "status": status,
    }


def _auth_status(auth: AuthProfile) -> Literal["ok", "invalid"]:
    return "ok" if _auth_validation_payload(auth)["status"] == "ok" else "invalid"


@app.on_event("startup")
def _startup_runtime_scheduler() -> None:
    store = _job_store()
    for job in store.list_jobs():
        _register_runtime_job(job)
        if job.active:
            try:
                runtime_manager.activate(job.name, run_now=False)
            except RuntimeError:
                continue


@app.on_event("shutdown")
def _shutdown_runtime_scheduler() -> None:
    runtime_manager.shutdown()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/personas", response_model=PersonaCatalogResponse)
def list_personas() -> dict[str, Any]:
    try:
        return _persona_catalog_payload(_persona_store())
    except FileNotFoundError:
        return {"personas": [], "source": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load personas file: {exc}") from exc


@app.put("/personas", response_model=PersonaCatalogResponse)
def upsert_personas(request: UpsertPersonasRequest) -> dict[str, Any]:
    store = _persona_store()
    try:
        persisted_path = store.save_personas(request.personas)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save personas file: {exc}") from exc
    payload = _persona_catalog_payload(store)
    payload["source"] = str(persisted_path)
    return payload


@app.get("/auths", response_model=AuthCatalogResponse)
def list_auths() -> dict[str, Any]:
    try:
        return _auth_catalog_payload(_auth_store())
    except FileNotFoundError:
        return {"auths": [], "source": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load auths file: {exc}") from exc


@app.put("/auths", response_model=AuthCatalogResponse)
def upsert_auths(request: UpsertAuthsRequest) -> dict[str, Any]:
    normalized_auths: list[AuthProfile] = []
    seen_ids: set[str] = set()
    for auth in request.auths:
        auth_id = auth.id.strip() or str(uuid4())
        social = _resolve_social(auth.social)
        reddit_username = auth.reddit_username.strip()
        auth_name = auth.name.strip()
        if social == "reddit":
            auth_name = auth_name or reddit_username or auth_id
        else:
            auth_name = auth_name or "linkedin"
        lowered = auth_id.lower()
        if lowered in seen_ids:
            raise HTTPException(status_code=400, detail=f"Duplicate auth id: {auth_id}")
        seen_ids.add(lowered)
        linkedin_user_data_dir = auth.linkedin_user_data_dir.strip()
        linkedin_headless = auth.linkedin_headless.strip() or "true"
        linkedin_wait_seconds = auth.linkedin_wait_seconds.strip() or "120"
        if social == "linkedin":
            linkedin_headless = "true"
            linkedin_user_data_dir = _default_linkedin_user_data_dir(auth_name, auth_id)
            linkedin_wait_seconds = "120"

        normalized_auths.append(
            AuthProfile(
                id=auth_id,
                name=auth_name,
                social=social,
                reddit_client_id=auth.reddit_client_id,
                reddit_client_secret=auth.reddit_client_secret,
                reddit_user_agent=auth.reddit_user_agent,
                reddit_username=reddit_username,
                reddit_password=auth.reddit_password,
                linkedin_email=auth.linkedin_email,
                linkedin_password=auth.linkedin_password,
                linkedin_headless=linkedin_headless,
                linkedin_user_data_dir=linkedin_user_data_dir,
                linkedin_wait_seconds=linkedin_wait_seconds,
                status="invalid",
            )
        )

    store = _auth_store()
    try:
        persisted_path = store.save_auths(normalized_auths)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save auths file: {exc}") from exc
    payload = _auth_catalog_payload(store)
    payload["source"] = str(persisted_path)
    return payload


@app.post("/auths/{auth_ref}/validate", response_model=AuthValidationResult)
def validate_auth(auth_ref: str) -> dict[str, str]:
    auth = _auth_by_ref(auth_ref)
    return _auth_validation_payload(auth)


@app.post("/auths/validate", response_model=AuthValidationResultsResponse)
def validate_auths() -> dict[str, list[dict[str, str]]]:
    results = [_auth_validation_payload(item) for item in _auth_store().list_auths()]
    return {"results": results}


@app.get("/settings", response_model=SettingsPayload)
def list_settings() -> dict[str, Any]:
    return _settings_store().load()


@app.put("/settings", response_model=SettingsPayload)
def save_settings(request: SettingsPayload) -> dict[str, Any]:
    payload = _settings_store().save(
        openrouter=request.openrouter.model_dump(mode="json"),
    )
    llm_service.reload_from_env()
    return payload


@app.post("/auths/{auth_ref}/linkedin/login")
def run_linkedin_login(auth_ref: str) -> dict[str, Any]:
    auth = _auth_by_ref(auth_ref)
    if _resolve_social(auth.social) != "linkedin":
        raise HTTPException(status_code=400, detail=f"Auth is not LinkedIn: {auth_ref}")

    raw_user_data_dir = auth.linkedin_user_data_dir.strip() or ".reddrop/auth/linkedin"
    user_data_dir = _resolve_linkedin_user_data_dir(raw_user_data_dir)
    raw_wait = auth.linkedin_wait_seconds.strip() or "120"
    try:
        wait_seconds = max(30, int(raw_wait))
    except Exception:
        wait_seconds = 120

    try:
        payload = LinkedinService.login_and_save_session(
            user_data_dir=user_data_dir,
            wait_seconds=wait_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload["auth_id"] = auth.id
    return payload


@app.get("/jobs")
def list_jobs() -> dict[str, Any]:
    jobs = [_job_status_payload(job) for job in _job_store().list_jobs()]
    return {"jobs": jobs}


@app.get("/jobs/status")
def list_job_statuses() -> dict[str, Any]:
    names = sorted(job.name for job in _job_store().list_jobs())
    return {"statuses": runtime_manager.list_statuses(names)}


@app.get("/jobs/{name}")
def get_job(name: str) -> dict[str, Any]:
    return _job_status_payload(_job_by_ref(name))


@app.post("/jobs")
def create_job(request: AddJobRequest) -> dict[str, Any]:
    store = _job_store()
    auth_store = _auth_store()
    existing = store.get_job_by_name(request.name)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Job name already exists: {request.name}")

    requested_social = _resolve_social(request.social)
    requested_auth_id = (request.auth_id or "").strip() or None
    request_job_type = "reply" if request.job_type == "reply" else "search"

    source_job: PeriodicSearchJob | None = None
    if request_job_type == "reply":
        source_job_id = (request.source_job_id or "").strip()
        if not source_job_id:
            raise HTTPException(status_code=400, detail="source_job_id is required for reply jobs.")
        personas = [item.strip() for item in (request.personas or []) if item and item.strip()]
        if not personas:
            raise HTTPException(status_code=400, detail="At least one persona is required for reply jobs.")
        source_job = store.get_job_by_id(source_job_id)
        if source_job is None:
            raise HTTPException(status_code=400, detail=f"Source search job not found: {source_job_id}")
        if source_job.job_type != "search":
            raise HTTPException(status_code=400, detail=f"Source job must be search type: {source_job.name}")
        source_social = _resolve_social(getattr(source_job, "social", "reddit"))
        requested_social = source_social

        source_auth_id = (source_job.auth_id or "").strip()
        if not source_auth_id:
            raise HTTPException(
                status_code=400,
                detail=f"Source search job {source_job.name} has no auth_id configured.",
            )
        if requested_auth_id and requested_auth_id != source_auth_id:
            raise HTTPException(
                status_code=400,
                detail=f"Reply job auth_id must match source search job auth_id: {source_auth_id}",
            )
        requested_auth_id = source_auth_id
        topic_value: str | None = None
        time_filter_value: str | None = None
        subreddit_limit_value: int | None = None
        threads_limit_value: int | None = None
        min_similarity_score_value = (
            request.min_similarity_score
            if request.min_similarity_score is not None
            else DEFAULT_MIN_SIMILARITY_SCORE
        )
        replies_per_iteration_value = (
            request.replies_per_iteration
            if request.replies_per_iteration is not None and request.replies_per_iteration > 0
            else DEFAULT_REPLIES_PER_ITERATION
        )
        personas_value: list[str] | None = personas
    else:
        topic_text = (request.topic or "").strip()
        if not topic_text:
            raise HTTPException(status_code=400, detail="topic is required for search jobs.")
        if not requested_auth_id:
            raise HTTPException(status_code=400, detail="auth_id is required for search jobs.")
        topic_value = topic_text
        time_filter_value = (request.time_filter or DEFAULT_TIME_FILTER).strip() or DEFAULT_TIME_FILTER
        subreddit_limit_value = (
            request.subreddit_limit
            if request.subreddit_limit is not None and request.subreddit_limit > 0
            else DEFAULT_SUBREDDIT_LIMIT
        )
        threads_limit_value = (
            request.threads_limit
            if request.threads_limit is not None and request.threads_limit > 0
            else DEFAULT_THREADS_LIMIT
        )
        min_similarity_score_value = None
        replies_per_iteration_value = None
        personas_value = None

    if not requested_auth_id:
        raise HTTPException(status_code=400, detail="auth_id is required.")

    auth = auth_store.get_auth(requested_auth_id)
    if auth is None:
        raise HTTPException(status_code=400, detail=f"Auth not found: {requested_auth_id}")
    auth_social = _resolve_social(auth.social)
    if auth_social != requested_social:
        raise HTTPException(
            status_code=400,
            detail=f"Auth social must match job social: {auth_social} != {requested_social}",
        )

    max_runtime_minutes = request.max_runtime_minutes if request.max_runtime_minutes > 0 else DEFAULT_MAX_RUNTIME_MINUTES

    args = argparse.Namespace(
        name=request.name,
        job_type=request_job_type,
        social=requested_social,
        auth_id=requested_auth_id,
        source_job_id=((request.source_job_id or "").strip() or None) if request_job_type == "reply" else None,
        topic=topic_value,
        min_similarity_score=min_similarity_score_value,
        active=bool(request.active),
        time_filter=time_filter_value,
        subreddit_limit=subreddit_limit_value,
        threads_limit=threads_limit_value,
        replies_per_iteration=replies_per_iteration_value,
        max_runtime_minutes=max_runtime_minutes,
        personas=personas_value,
    )

    payload: dict[str, Any] = {}

    def _emit(data: dict) -> None:
        payload.update(data)

    add_command.handle(
        args,
        store,
        _error,
        _emit,
        JOB_NAME_PATTERN,
    )
    created = _job_by_ref(str(payload.get("id", request.name)))
    _register_runtime_job(created)
    if created.active:
        runtime_manager.activate(created.name, run_now=False)
    return _job_status_payload(created)


@app.get("/jobs/{name}/status", response_model=JobActionResponse)
def get_job_status(name: str) -> dict[str, Any]:
    job = _job_by_ref(name)
    status = runtime_manager.status(job.name)
    status["id"] = job.id
    return status


@app.post("/jobs/add")
def add_job(request: AddJobRequest) -> dict[str, Any]:
    return create_job(request)


@app.put("/jobs/{name}")
def update_job(name: str, request: AddJobRequest) -> dict[str, Any]:
    _ = request
    _job_by_ref(name)
    raise HTTPException(
        status_code=405,
        detail="Jobs cannot be edited after creation. Use activate/deactivate/start/stop controls.",
    )


@app.delete("/jobs/{name}")
def delete_job(name: str) -> dict[str, Any]:
    target = _job_by_ref(name)
    runtime_manager.deactivate(target.name)
    runtime_manager.unregister(target.name)
    removed = _job_store().delete_job(target.id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Job not found: {name}")

    search_path = _search_storage().path_for_job(target.id)
    if search_path.exists():
        search_path.unlink()

    return {"deleted": True, "id": target.id, "name": target.name}


@app.post("/jobs/{name}/start", response_model=JobActionResponse)
def start_job(name: str) -> dict[str, Any]:
    job = _job_by_ref(name)
    try:
        status = runtime_manager.start(job.name)
        status["id"] = job.id
        return status
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/jobs/{name}/stop", response_model=JobActionResponse)
def stop_job(name: str) -> dict[str, Any]:
    job = _job_by_ref(name)
    status = runtime_manager.stop(job.name)
    status["id"] = job.id
    return status


@app.post("/jobs/{name}/activate", response_model=JobActionResponse)
def activate_job(name: str) -> dict[str, Any]:
    job = _job_by_ref(name)
    updated = _job_store().set_active(job.id, True) or job
    _register_runtime_job(updated)
    try:
        status = runtime_manager.activate(updated.name, run_now=False)
        status["id"] = updated.id
        return status
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/jobs/{name}/deactivate", response_model=JobActionResponse)
def deactivate_job(name: str) -> dict[str, Any]:
    job = _job_by_ref(name)
    updated = _job_store().set_active(job.id, False) or job
    status = runtime_manager.deactivate(updated.name, stop_processing=True)
    status["id"] = updated.id
    return status


@app.post("/jobs/{name}/restart", response_model=JobActionResponse)
def restart_job(name: str) -> dict[str, Any]:
    job = _job_by_ref(name)
    try:
        status = runtime_manager.restart(job.name)
        status["id"] = job.id
        return status
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/search")
def run_search(request: SearchRequest) -> dict[str, Any]:
    args = argparse.Namespace(name=request.name)
    deps = search_command.SearchDependencies(
        logger=logger,
        search_storage_factory=_search_storage,
        reddit_service_cls=RedditService,
        linkedin_service_cls=LinkedinService,
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=search_command.utc_now,
        prepare_job=_prepare_auth_for_job,
    )
    payload: dict[str, Any] = {}

    def _emit(data: dict) -> None:
        payload.update(data)

    search_command.handle(args, _job_store(), _emit, _error, deps)
    return payload


@app.post("/reply")
def generate_reply(request: ReplyRequest) -> dict[str, Any]:
    args = argparse.Namespace(
        search_file=request.search_file,
        personas=request.personas,
        replies=request.replies,
        min_similarity=request.min_similarity,
    )
    deps = reply_command.ReplyDependencies(
        logger=logger,
        runs_dir=DEFAULT_RUNS_DIR,
        persona_store_cls=_persona_store,
        reply_generation_service_cls=ReplyGenerationService,
        search_storage_cls=SearchStorage,
        search_artifact_model=SearchArtifact,
        conversation_model=Conversation,
        now_fn=reply_command.utc_now,
    )
    payload: dict[str, Any] = {}

    def _emit(data: dict) -> None:
        payload.update(data)

    reply_command.handle(args, _emit, _error, deps)
    return payload


@app.post("/send")
def send_reply(request: SendRequest) -> dict[str, Any]:
    search_file = (request.search_file or request.replies_file or "").strip()
    if not search_file:
        raise HTTPException(status_code=400, detail="search_file is required.")
    args = argparse.Namespace(search_file=search_file)
    deps = send_command.SendDependencies(
        logger=logger,
        runs_dir=DEFAULT_RUNS_DIR,
        search_storage_cls=SearchStorage,
        reddit_service_cls=RedditService,
        linkedin_service_cls=LinkedinService,
        job_store_factory=_job_store,
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=send_command.utc_now,
        prepare_job=_prepare_auth_for_job,
    )
    payload: dict[str, Any] = {}

    def _emit(data: dict) -> None:
        payload.update(data)

    send_command.handle(args, _emit, _error, deps)
    return payload


@app.get("/threads")
def list_threads(
    job_id: str | None = None,
    job_name: str | None = None,
    subreddit: str | None = None,
    min_similarity: float | None = None,
    only_open: bool = False,
    has_reply: bool | None = Query(default=None),
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    threads, total = _list_threads_from_artifacts(
        job_id=job_id,
        job_name=job_name,
        subreddit=subreddit,
        min_similarity=min_similarity,
        only_open=only_open,
        has_reply=has_reply,
        limit=limit,
        offset=offset,
    )
    return {
        "threads": threads,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.post("/threads/{job_ref}/{subreddit}/{thread_id}/reply")
def create_reply_for_thread(
    job_ref: str,
    subreddit: str,
    thread_id: str,
    request: ThreadReplyRequest,
) -> dict[str, Any]:
    persona_name = request.persona.strip()
    if not persona_name:
        raise HTTPException(status_code=400, detail="persona is required")

    search_job, reply_job = _resolve_thread_job_context(job_ref)
    search_storage = _search_storage()
    search_artifact = search_storage.get_search(search_job.id)
    if search_artifact is None:
        raise HTTPException(status_code=404, detail=f"Search artifact not found for job: {search_job.id}")

    conversation = next(
        (
            item
            for item in search_artifact.conversations
            if _thread_key_match(item, thread_id=thread_id, subreddit=subreddit)
        ),
        None,
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Thread not found for job: {thread_id}")

    try:
        persona = _persona_store().get_persona(persona_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load personas file: {exc}") from exc
    if persona is None:
        raise HTTPException(status_code=404, detail=f"Persona not found: {persona_name}")

    if request.reply and request.reply.strip():
        generated_text = request.reply.strip()
    else:
        generated_text = ReplyGenerationService().generate_reply(
            topic=search_artifact.topic,
            conversation=conversation,
            persona=persona,
        )
    if not generated_text.strip():
        raise HTTPException(status_code=500, detail="Reply generation returned empty output.")

    search_path = search_storage.path_for_job(search_job.id)
    timestamp = datetime.now(timezone.utc)
    updated_conversations: list[Conversation] = []
    for item in search_artifact.conversations:
        if _thread_key_match(item, thread_id=thread_id, subreddit=subreddit):
            payload = item.model_dump(mode="python")
            payload["reply"] = generated_text.strip()
            updated_conversations.append(
                Conversation(**payload)
            )
        else:
            updated_conversations.append(item)

    updated_search = SearchArtifact(
        id=search_artifact.id,
        name=search_artifact.name,
        topic=search_artifact.topic,
        created_at=search_artifact.created_at,
        updated_at=timestamp,
        conversations=updated_conversations,
    )
    search_storage.save_search_at_path(search_path, updated_search)
    persisted = search_storage.get_search(search_job.id)
    if persisted is None:
        raise HTTPException(status_code=500, detail="Persisted search artifact not found after reply generation.")

    updated = next(
        (
            item
            for item in persisted.conversations
            if _thread_key_match(item, thread_id=thread_id, subreddit=subreddit)
        ),
        None,
    )
    if updated is None:
        raise HTTPException(status_code=500, detail="Persisted reply not found after save.")

    return {
        **_thread_payload(
            search_job=search_job,
            conversation=updated,
            reply=updated.reply,
            reply_job=reply_job,
        ),
        "search_file": search_path.name,
        "updated_at": persisted.updated_at,
    }


@app.post("/threads/{job_ref}/{subreddit}/{thread_id}/send")
def send_reply_for_thread(job_ref: str, subreddit: str, thread_id: str) -> dict[str, Any]:
    search_job, reply_job = _resolve_thread_job_context(job_ref)
    search_storage = _search_storage()
    search_artifact = search_storage.get_search(search_job.id)
    if search_artifact is None:
        raise HTTPException(status_code=404, detail=f"Search artifact not found for job: {search_job.id}")
    search_path = search_storage.path_for_job(search_job.id)

    selected = next(
        (
            item
            for item in search_artifact.conversations
            if _thread_key_match(item, thread_id=thread_id, subreddit=subreddit)
        ),
        None,
    )
    if selected is None:
        raise HTTPException(status_code=404, detail=f"Reply not found for thread: {thread_id}")
    if not selected.reply.strip():
        raise HTTPException(status_code=400, detail="Reply text is empty.")

    social = _resolve_social(getattr(search_job, "social", "reddit"))
    _prepare_auth_for_job(search_job)
    post_service: RedditService | LinkedinService
    if social == "linkedin":
        post_service = LinkedinService()
    else:
        post_service = RedditService()

    if not selected.user_has_commented:
        posted = post_service.post_comment(thread_id=selected.thread_id, comment_text=selected.reply)
        if not posted:
            raise HTTPException(status_code=502, detail=f"Failed to send reply to {social}.")

    now = datetime.now(timezone.utc)

    updated_search_conversations: list[Conversation] = []
    for item in search_artifact.conversations:
        if _thread_key_match(item, thread_id=thread_id, subreddit=subreddit):
            item.user_has_commented = True
        updated_search_conversations.append(item)
    updated_search = SearchArtifact(
        id=search_artifact.id,
        name=search_artifact.name,
        topic=search_artifact.topic,
        created_at=search_artifact.created_at,
        updated_at=now,
        conversations=updated_search_conversations,
    )
    search_storage.save_search_at_path(search_path, updated_search)

    sent = next(
        (
            item
            for item in updated_search.conversations
            if _thread_key_match(item, thread_id=thread_id, subreddit=subreddit)
        ),
        None,
    )
    if sent is None:
        raise HTTPException(status_code=500, detail="Persisted sent reply not found.")

    return {
        **_thread_payload(
            search_job=search_job,
            conversation=sent,
            reply=sent.reply,
            reply_job=reply_job,
        ),
        "search_file": search_path.name,
        "updated_at": now,
    }


@app.get("/artifacts/search")
def list_search_artifacts() -> dict[str, Any]:
    return _list_artifacts("search")


@app.get("/artifacts/search/{artifact_name}")
def get_search_artifact(artifact_name: str) -> dict[str, Any]:
    path = _resolve_artifact_path("search", artifact_name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SearchArtifact.model_validate(payload).model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invalid search artifact: {artifact_name}") from exc


@app.get("/artifacts/reply")
def list_reply_artifacts() -> dict[str, Any]:
    base = _list_artifacts("search")
    artifacts: list[dict[str, Any]] = []
    for item in base["artifacts"]:
        path = DEFAULT_RUNS_DIR / str(item["name"])
        artifact = _read_search_artifact(path)
        if artifact is None:
            continue
        replied_count = sum(1 for conversation in artifact.conversations if conversation.reply.strip())
        if replied_count == 0:
            continue
        enriched = dict(item)
        enriched["replied_count"] = replied_count
        artifacts.append(enriched)
    return {"artifacts": artifacts}


@app.get("/artifacts/reply/{artifact_name}")
def get_reply_artifact(artifact_name: str) -> dict[str, Any]:
    normalized_name = artifact_name
    if normalized_name.startswith("reply_"):
        normalized_name = normalized_name.replace("reply_", "search_", 1)
    path = _resolve_artifact_path("search", normalized_name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifact = SearchArtifact.model_validate(payload)
        artifact.conversations = [item for item in artifact.conversations if item.reply.strip()]
        return artifact.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invalid reply artifact: {artifact_name}") from exc


def run_api_server(*, host: str = "0.0.0.0", port: int = 8000, reload: bool = True) -> None:
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_config=get_app_log_config(),
    )


if __name__ == "__main__":
    run_api_server()
