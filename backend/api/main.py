from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Support both import styles:
# - python -m uvicorn backend.api.main:app  (repo root cwd)
# - python -m uvicorn api.main:app          (backend/ cwd)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.api.job_runtime import JobRuntimeManager
from backend.cli.commands import add as add_command
from backend.cli.commands import reply as reply_command
from backend.cli.commands import search as search_command
from backend.cli.commands import send as send_command
from backend.config.log import get_app_log_config, setup_logging
from backend.models import (
    Conversation,
    PeriodicSearchJob,
    PersonaProfile,
    SearchArtifact,
)
from backend.services.job_config_store import JobConfigStore
from backend.services.llm_service import llm_service
from backend.services.persona_store import PersonaStore
from backend.services.reply_generation_service import ReplyGenerationService
from backend.services.reddit_service import RedditService
from backend.services.search_storage import SearchStorage
from backend.services.settings_store import SettingsStore

DEFAULT_HOME = Path(".reddrop")
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

app = FastAPI(title="Reddrop API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
runtime_manager = JobRuntimeManager()


class AddJobRequest(BaseModel):
    name: str
    job_type: Literal["search", "reply"] = "search"
    source_job_id: str | None = None
    topic: str = ""
    active: bool | None = None
    time_filter: str = DEFAULT_TIME_FILTER
    subreddit_limit: int = DEFAULT_SUBREDDIT_LIMIT
    threads_limit: int = DEFAULT_THREADS_LIMIT
    replies_per_iteration: int = DEFAULT_REPLIES_PER_ITERATION
    min_similarity_score: float = Field(default=DEFAULT_MIN_SIMILARITY_SCORE, ge=0.0, le=1.0)
    max_runtime_minutes: int = DEFAULT_MAX_RUNTIME_MINUTES
    personas: list[str] = Field(default_factory=list)


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


class RedditSettings(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "Reddrop:v1.0"
    username: str = ""
    password: str = ""


class OpenRouterSettings(BaseModel):
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "x-ai/grok-4.1-fast"
    http_referer: str = ""
    x_title: str = "reddrop"
    timeout_seconds: str = "20"


class SettingsPayload(BaseModel):
    reddit: RedditSettings = Field(default_factory=RedditSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)


def _job_store() -> JobConfigStore:
    return JobConfigStore(DEFAULT_JOBS_CONFIG)


def _search_storage() -> SearchStorage:
    return SearchStorage(DEFAULT_RUNS_DIR)


def _settings_store() -> SettingsStore:
    return SettingsStore(DEFAULT_HOME / "settings.yaml")


def _error(message: str) -> int:
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
    payload = job.model_dump(mode="json")
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

        if job.job_type == "search":
            _stop_if_requested()
            set_phase("searching")
            search_file = _run_search_for_job(job.name, should_stop=should_stop)
            _stop_if_requested()
            return f"search_file={search_file}"

        if job.job_type == "reply":
            if not job.personas:
                raise RuntimeError(f"Reply job {job.name} requires at least one persona.")
            source_search_file = _resolve_reply_source_search_file(job, store)
            _stop_if_requested()
            set_phase("replying")
            _run_reply_for_job(
                search_file=source_search_file,
                persona=job.personas[0],
                replies=max(1, int(job.replies_per_iteration)),
                min_similarity=max(0.0, min(1.0, float(job.min_similarity_score))),
                should_stop=should_stop,
            )
            _stop_if_requested()
            return f"reply_file={source_search_file}"

        raise RuntimeError(f"Unsupported job type: {job.job_type}")

    return _runner


def _run_search_for_job(job_name: str, *, should_stop=None) -> str:
    args = argparse.Namespace(name=job_name)
    deps = search_command.SearchDependencies(
        logger=logger,
        search_storage_factory=_search_storage,
        reddit_service_cls=RedditService,
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=search_command.utc_now,
        should_stop=should_stop or (lambda: False),
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
) -> None:
    args = argparse.Namespace(
        search_file=search_file,
        personas=persona,
        replies=replies,
        min_similarity=min_similarity,
    )
    deps = reply_command.ReplyDependencies(
        logger=logger,
        runs_dir=DEFAULT_RUNS_DIR,
        persona_store_cls=PersonaStore,
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


def _run_send_for_job(*, search_file: str, should_stop=None) -> None:
    args = argparse.Namespace(search_file=search_file)
    deps = send_command.SendDependencies(
        logger=logger,
        runs_dir=DEFAULT_RUNS_DIR,
        search_storage_cls=SearchStorage,
        reddit_service_cls=RedditService,
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=send_command.utc_now,
        should_stop=should_stop or (lambda: False),
    )
    errors: list[str] = []
    rc = send_command.handle(args, lambda _data: None, lambda message: errors.append(message) or 1, deps)
    if rc != 0:
        raise RuntimeError(errors[-1] if errors else f"Send failed for {search_file}")


def _thread_key_match(item: Conversation, *, thread_id: str, subreddit: str) -> bool:
    return item.thread_id == thread_id and item.subreddit == subreddit


def _thread_payload(
    *,
    job: PeriodicSearchJob,
    conversation: Conversation,
    reply: str = "",
) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "job_name": job.name,
        "topic": job.topic,
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


def _list_threads_from_artifacts(
    *,
    job_id: str | None,
    job_name: str | None,
    subreddit: str | None,
    min_similarity: float | None,
    only_open: bool,
    has_reply: bool | None,
) -> list[dict[str, Any]]:
    DEFAULT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    normalized_job_id = (job_id or "").strip()
    normalized_job_name = (job_name or "").strip()
    normalized_subreddit = (subreddit or "").strip()

    for search_path in sorted(DEFAULT_RUNS_DIR.glob("search_*.json"), key=lambda item: item.name):
        artifact = _read_search_artifact(search_path)
        if artifact is None:
            continue
        if normalized_job_id and artifact.id != normalized_job_id:
            continue
        if normalized_job_name and artifact.name != normalized_job_name:
            continue

        for conversation in artifact.conversations:
            if normalized_subreddit and conversation.subreddit != normalized_subreddit:
                continue
            similarity = float(conversation.semantic_similarity or 0.0)
            if min_similarity is not None and similarity < float(min_similarity):
                continue

            reply_text = conversation.reply
            user_has_commented = bool(conversation.user_has_commented)
            has_reply_value = bool(reply_text.strip())

            if only_open and user_has_commented:
                continue
            if has_reply is True and not has_reply_value:
                continue
            if has_reply is False and has_reply_value:
                continue

            rows.append(
                {
                    "job_id": artifact.id,
                    "job_name": artifact.name,
                    "topic": artifact.topic,
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
                    "semantic_similarity": similarity,
                    "user_has_commented": user_has_commented,
                    "reply": reply_text,
                    "has_reply": has_reply_value,
                }
            )

    rows.sort(key=lambda item: (item["semantic_similarity"], item["score"]), reverse=True)
    return rows


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
        return _persona_catalog_payload(PersonaStore())
    except FileNotFoundError:
        return {"personas": [], "source": None}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load personas file: {exc}") from exc


@app.put("/personas", response_model=PersonaCatalogResponse)
def upsert_personas(request: UpsertPersonasRequest) -> dict[str, Any]:
    store = PersonaStore()
    try:
        persisted_path = store.save_personas(request.personas)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save personas file: {exc}") from exc
    payload = _persona_catalog_payload(store)
    payload["source"] = str(persisted_path)
    return payload


@app.get("/settings", response_model=SettingsPayload)
def list_settings() -> dict[str, Any]:
    return _settings_store().load()


@app.put("/settings", response_model=SettingsPayload)
def save_settings(request: SettingsPayload) -> dict[str, Any]:
    payload = _settings_store().save(
        reddit=request.reddit.model_dump(mode="json"),
        openrouter=request.openrouter.model_dump(mode="json"),
    )
    llm_service.reload_from_env()
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
    existing = store.get_job_by_name(request.name)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Job name already exists: {request.name}")
    source_job: PeriodicSearchJob | None = None
    if request.job_type == "reply":
        source_job_id = (request.source_job_id or "").strip()
        if not source_job_id:
            raise HTTPException(status_code=400, detail="source_job_id is required for reply jobs.")
        if not request.personas:
            raise HTTPException(status_code=400, detail="At least one persona is required for reply jobs.")
        source_job = store.get_job_by_id(source_job_id)
        if source_job is None:
            raise HTTPException(status_code=400, detail=f"Source search job not found: {source_job_id}")
        if source_job.job_type != "search":
            raise HTTPException(status_code=400, detail=f"Source job must be search type: {source_job.name}")
    else:
        if not request.topic.strip():
            raise HTTPException(status_code=400, detail="topic is required for search jobs.")

    topic_value = request.topic.strip() if request.job_type == "search" else ""

    args = argparse.Namespace(
        name=request.name,
        job_type=request.job_type,
        source_job_id=(request.source_job_id or "").strip() or None,
        topic=topic_value,
        min_similarity_score=request.min_similarity_score,
        active=bool(request.active),
        time_filter=request.time_filter,
        subreddit_limit=request.subreddit_limit,
        threads_limit=request.threads_limit,
        replies_per_iteration=request.replies_per_iteration,
        max_runtime_minutes=request.max_runtime_minutes,
        personas=request.personas,
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
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=search_command.utc_now,
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
        persona_store_cls=PersonaStore,
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
        conversation_model=Conversation,
        search_artifact_model=SearchArtifact,
        now_fn=send_command.utc_now,
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
    rows = _list_threads_from_artifacts(
        job_id=job_id,
        job_name=job_name,
        subreddit=subreddit,
        min_similarity=min_similarity,
        only_open=only_open,
        has_reply=has_reply,
    )
    total = len(rows)
    threads = rows[offset : offset + limit]
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

    job = _job_by_ref(job_ref)
    search_storage = _search_storage()
    search_artifact = search_storage.get_search(job.id)
    if search_artifact is None:
        raise HTTPException(status_code=404, detail=f"Search artifact not found for job: {job.id}")

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
        persona = PersonaStore().get_persona(persona_name)
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

    search_path = search_storage.path_for_job(job.id)
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
    persisted = search_storage.get_search(job.id)
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
        **_thread_payload(job=job, conversation=updated, reply=updated.reply),
        "search_file": search_path.name,
        "updated_at": persisted.updated_at,
    }


@app.post("/threads/{job_ref}/{subreddit}/{thread_id}/send")
def send_reply_for_thread(job_ref: str, subreddit: str, thread_id: str) -> dict[str, Any]:
    job = _job_by_ref(job_ref)
    search_storage = _search_storage()
    search_artifact = search_storage.get_search(job.id)
    if search_artifact is None:
        raise HTTPException(status_code=404, detail=f"Search artifact not found for job: {job.id}")
    search_path = search_storage.path_for_job(job.id)

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

    if not selected.user_has_commented:
        posted = RedditService().post_comment(thread_id=selected.thread_id, comment_text=selected.reply)
        if not posted:
            raise HTTPException(status_code=502, detail="Failed to send reply to Reddit.")

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
        **_thread_payload(job=job, conversation=sent, reply=sent.reply),
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
