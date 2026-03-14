from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


class Conversation(BaseModel):
    thread_id: str
    title: str
    subreddit: str
    url: str
    score: int
    num_comments: int
    created_utc: float | int | str | None = None
    selftext: str = ""
    semantic_similarity: float = 0.0
    user_has_commented: bool = False
    reply: str = ""


class SearchArtifact(BaseModel):
    id: str = Field(default="", validation_alias=AliasChoices("id", "job_id"))
    name: str
    topic: str
    created_at: datetime
    updated_at: datetime
    conversations: list[Conversation]


class ReplyConversation(Conversation):
    reply: str


class ReplyArtifact(BaseModel):
    id: str = Field(default="", validation_alias=AliasChoices("id", "job_id"))
    name: str
    topic: str
    created_at: datetime
    updated_at: datetime
    conversations: list[ReplyConversation]


class PersonaProfile(BaseModel):
    name: str
    description: str
    objective: str


class AuthProfile(BaseModel):
    id: str = Field(validation_alias=AliasChoices("id", "auth_id"))
    name: str = ""
    social: Literal["reddit", "linkedin"] = "reddit"
    status: Literal["ok", "invalid"] = "invalid"
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "Reddrop:v1.0"
    reddit_username: str = ""
    reddit_password: str = ""
    linkedin_email: str = ""
    linkedin_password: str = ""
    linkedin_headless: str = "true"
    linkedin_user_data_dir: str = ".reddrop/auth/linkedin"
    linkedin_wait_seconds: str = "120"


class PeriodicSearchJob(BaseModel):
    id: str = Field(validation_alias=AliasChoices("id", "job_id"))
    name: str
    job_type: Literal["search", "reply"] = "search"
    social: Literal["reddit", "linkedin"] = "reddit"
    auth_id: str | None = None
    source_job_id: str | None = None
    topic: str | None = None
    min_similarity_score: float | None = None
    active: bool = False
    time_filter: str | None = None
    subreddit_limit: int | None = None
    threads_limit: int | None = None
    replies_per_iteration: int | None = None
    max_runtime_minutes: int = 1440
    personas: list[str] | None = None

    @staticmethod
    def _positive_int(value: object, fallback: int) -> int:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except Exception:
            return fallback
        if parsed <= 0:
            return fallback
        return parsed

    @staticmethod
    def _bounded_float(value: object, fallback: float) -> float:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except Exception:
            return fallback
        if parsed < 0.0:
            return 0.0
        if parsed > 1.0:
            return 1.0
        return parsed

    @staticmethod
    def _normalized_personas(personas: object) -> list[str]:
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

    @model_validator(mode="after")
    def _normalize_for_job_type(self) -> "PeriodicSearchJob":
        self.max_runtime_minutes = self._positive_int(self.max_runtime_minutes, 1440)
        if self.job_type == "search":
            self.topic = str(self.topic or "").strip()
            self.source_job_id = None
            self.time_filter = str(self.time_filter or "week").strip() or "week"
            self.subreddit_limit = self._positive_int(self.subreddit_limit, 5)
            self.threads_limit = self._positive_int(self.threads_limit, 10)
            self.min_similarity_score = None
            self.replies_per_iteration = None
            self.personas = None
            return self

        self.topic = None
        self.time_filter = None
        self.subreddit_limit = None
        self.threads_limit = None
        self.min_similarity_score = self._bounded_float(self.min_similarity_score, 0.35)
        self.replies_per_iteration = self._positive_int(self.replies_per_iteration, 3)
        self.personas = self._normalized_personas(self.personas)
        return self


class JobsConfig(BaseModel):
    jobs: list[PeriodicSearchJob] = Field(default_factory=list)


class AuthsConfig(BaseModel):
    auths: list[AuthProfile] = Field(default_factory=list)
