from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Literal

from ..models import JobsConfig, PeriodicSearchJob


class JobConfigStore:
    def __init__(self, config_path: str | Path = ".reddrop/jobs_config.json") -> None:
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path = self.config_path.parent / "jobs" / "jobs_config.json"
        if not self.config_path.exists() and legacy_path.exists():
            self.config_path.write_text(legacy_path.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                legacy_path.unlink()
                legacy_dir = legacy_path.parent
                if legacy_dir.exists() and not any(legacy_dir.iterdir()):
                    legacy_dir.rmdir()
            except Exception:
                pass
        self._lock = Lock()
        self._config = self._load()

    def list_jobs(self) -> list[PeriodicSearchJob]:
        with self._lock:
            return [PeriodicSearchJob.model_validate(item.model_dump(mode="python")) for item in self._config.jobs]

    def get_job_by_id(self, job_id: str) -> PeriodicSearchJob | None:
        normalized_id = job_id.strip()
        if not normalized_id:
            return None
        with self._lock:
            for job in self._config.jobs:
                if job.id == normalized_id:
                    return PeriodicSearchJob.model_validate(job.model_dump(mode="python"))
        return None

    def get_job_by_name(self, name: str) -> PeriodicSearchJob | None:
        normalized_name = name.strip()
        if not normalized_name:
            return None
        with self._lock:
            for job in self._config.jobs:
                if job.name == normalized_name:
                    return PeriodicSearchJob.model_validate(job.model_dump(mode="python"))
        return None

    def upsert_job(
        self,
        *,
        job_id: str,
        name: str,
        job_type: Literal["search", "reply"] = "search",
        source_job_id: str | None = None,
        topic: str,
        min_similarity_score: float = 0.35,
        active: bool = False,
        time_filter: str,
        subreddit_limit: int,
        threads_limit: int,
        replies_per_iteration: int = 3,
        max_runtime_minutes: int = 1440,
        personas: list[str] | None = None,
    ) -> PeriodicSearchJob:
        normalized_id = job_id.strip()
        if not normalized_id:
            raise ValueError("job_id is required")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("name is required")

        with self._lock:
            existing = None
            for index, job in enumerate(self._config.jobs):
                if job.name.lower() == normalized_name.lower():
                    existing = index
                    break

            payload = PeriodicSearchJob(
                id=self._config.jobs[existing].id if existing is not None else normalized_id,
                name=normalized_name,
                job_type=self._normalize_job_type(job_type),
                source_job_id=self._normalize_source_job_id(source_job_id),
                topic=self._normalize_topic_for_job_type(job_type=job_type, topic=topic),
                min_similarity_score=self._normalize_similarity_score(min_similarity_score, fallback=0.35),
                active=bool(active),
                time_filter=time_filter,
                subreddit_limit=subreddit_limit,
                threads_limit=threads_limit,
                replies_per_iteration=self._positive_int(replies_per_iteration, fallback=3),
                max_runtime_minutes=self._positive_int(max_runtime_minutes, fallback=1440),
                personas=self._normalize_persona_names(personas),
            )

            if existing is None:
                self._config.jobs.append(payload)
            else:
                self._config.jobs[existing] = payload
            self._persist()
            return PeriodicSearchJob.model_validate(payload.model_dump(mode="python"))

    def update_job(
        self,
        *,
        job_id: str,
        name: str,
        job_type: Literal["search", "reply"] = "search",
        source_job_id: str | None = None,
        topic: str,
        min_similarity_score: float = 0.35,
        active: bool = False,
        time_filter: str,
        subreddit_limit: int,
        threads_limit: int,
        replies_per_iteration: int = 3,
        max_runtime_minutes: int = 1440,
        personas: list[str] | None = None,
    ) -> PeriodicSearchJob:
        normalized_id = job_id.strip()
        if not normalized_id:
            raise ValueError("job_id is required")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("name is required")

        with self._lock:
            target_index: int | None = None
            for index, job in enumerate(self._config.jobs):
                if job.id == normalized_id:
                    target_index = index
                    break
            if target_index is None:
                raise ValueError(f"Job not found: {normalized_id}")

            for index, job in enumerate(self._config.jobs):
                if index != target_index and job.name.lower() == normalized_name.lower():
                    raise ValueError(f"Job name already exists: {normalized_name}")

            payload = PeriodicSearchJob(
                id=normalized_id,
                name=normalized_name,
                job_type=self._normalize_job_type(job_type),
                source_job_id=self._normalize_source_job_id(source_job_id),
                topic=self._normalize_topic_for_job_type(job_type=job_type, topic=topic),
                min_similarity_score=self._normalize_similarity_score(min_similarity_score, fallback=0.35),
                active=bool(active),
                time_filter=time_filter,
                subreddit_limit=subreddit_limit,
                threads_limit=threads_limit,
                replies_per_iteration=self._positive_int(replies_per_iteration, fallback=3),
                max_runtime_minutes=self._positive_int(max_runtime_minutes, fallback=1440),
                personas=self._normalize_persona_names(personas),
            )
            self._config.jobs[target_index] = payload
            self._persist()
            return PeriodicSearchJob.model_validate(payload.model_dump(mode="python"))

    def delete_job(self, job_id: str) -> bool:
        normalized_id = job_id.strip()
        if not normalized_id:
            return False
        with self._lock:
            original_len = len(self._config.jobs)
            self._config.jobs = [job for job in self._config.jobs if job.id != normalized_id]
            deleted = len(self._config.jobs) != original_len
            if deleted:
                self._persist()
            return deleted

    def set_active(self, job_id: str, active: bool) -> PeriodicSearchJob | None:
        normalized_id = job_id.strip()
        if not normalized_id:
            return None
        with self._lock:
            for index, job in enumerate(self._config.jobs):
                if job.id != normalized_id:
                    continue
                updated = PeriodicSearchJob(
                    id=job.id,
                    name=job.name,
                    job_type=job.job_type,
                    source_job_id=job.source_job_id,
                    topic=job.topic,
                    min_similarity_score=job.min_similarity_score,
                    active=bool(active),
                    time_filter=job.time_filter,
                    subreddit_limit=job.subreddit_limit,
                    threads_limit=job.threads_limit,
                    replies_per_iteration=job.replies_per_iteration,
                    max_runtime_minutes=job.max_runtime_minutes,
                    personas=list(job.personas),
                )
                self._config.jobs[index] = updated
                self._persist()
                return PeriodicSearchJob.model_validate(updated.model_dump(mode="python"))
        return None

    def _load(self) -> JobsConfig:
        if not self.config_path.exists():
            return JobsConfig()
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            payload = self._normalize_payload(payload)
            return JobsConfig.model_validate(payload)
        except Exception:
            return JobsConfig()

    @staticmethod
    def _normalize_payload(payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {"jobs": []}

        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            return {"jobs": []}

        normalized_jobs: list[dict] = []
        for index, item in enumerate(jobs, start=1):
            if not isinstance(item, dict):
                continue
            record = dict(item)
            if "id" not in record and "job_id" in record:
                record["id"] = record.get("job_id")
            if not record.get("name"):
                source_id = str(record.get("id", "")).strip()
                source_topic = str(record.get("topic", "")).strip()
                if source_topic:
                    record["name"] = source_topic[:32]
                elif source_id:
                    record["name"] = f"job-{source_id[:8]}"
                else:
                    record["name"] = f"job-{index}"
            if not record.get("time_filter"):
                record["time_filter"] = "week"
            record["job_type"] = JobConfigStore._normalize_job_type(record.get("job_type"))
            record["source_job_id"] = JobConfigStore._normalize_source_job_id(record.get("source_job_id"))
            record["topic"] = JobConfigStore._normalize_topic_for_job_type(
                job_type=record.get("job_type"),
                topic=record.get("topic"),
            )
            record["min_similarity_score"] = JobConfigStore._normalize_similarity_score(
                record.get("min_similarity_score"),
                fallback=0.35,
            )
            if "active" not in record:
                record["active"] = False
            if "subreddit_limit" not in record:
                record["subreddit_limit"] = 5
            if "threads_limit" not in record:
                record["threads_limit"] = 10
            record["replies_per_iteration"] = JobConfigStore._positive_int(
                record.get("replies_per_iteration"),
                fallback=3,
            )
            fallback_minutes = 1440
            if "max_runtime_hours" in record:
                fallback_minutes = JobConfigStore._positive_int(record.get("max_runtime_hours"), fallback=24) * 60
            record["max_runtime_minutes"] = JobConfigStore._positive_int(
                record.get("max_runtime_minutes"),
                fallback=fallback_minutes,
            )
            record.pop("max_runtime_hours", None)
            record["personas"] = JobConfigStore._normalize_persona_names(record.get("personas"))
            normalized_jobs.append(record)

        return {"jobs": normalized_jobs}

    def _persist(self) -> None:
        self.config_path.write_text(
            json.dumps(self._config.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _positive_int(value: object, *, fallback: int) -> int:
        try:
            numeric = int(value)  # type: ignore[arg-type]
        except Exception:
            return fallback
        if numeric <= 0:
            return fallback
        return numeric

    @staticmethod
    def _normalize_persona_names(personas: object) -> list[str]:
        if not isinstance(personas, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in personas:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(value)
        return normalized

    @staticmethod
    def _normalize_job_type(raw: object) -> Literal["search", "reply"]:
        value = str(raw or "").strip().lower()
        if value in {"reply", "replying"}:
            return "reply"
        return "search"

    @staticmethod
    def _normalize_source_job_id(raw: object) -> str | None:
        if raw is None:
            return None
        value = str(raw).strip()
        return value or None

    @staticmethod
    def _normalize_topic_for_job_type(*, job_type: object, topic: object) -> str:
        if JobConfigStore._normalize_job_type(job_type) == "reply":
            return ""
        return str(topic or "").strip()

    @staticmethod
    def _normalize_similarity_score(raw: object, *, fallback: float) -> float:
        try:
            value = float(raw)  # type: ignore[arg-type]
        except Exception:
            return fallback
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
