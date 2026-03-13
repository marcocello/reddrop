from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


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


class PeriodicSearchJob(BaseModel):
    id: str = Field(validation_alias=AliasChoices("id", "job_id"))
    name: str
    job_type: Literal["search", "reply"] = "search"
    source_job_id: str | None = None
    topic: str
    min_similarity_score: float = 0.35
    active: bool = False
    time_filter: str = "week"
    subreddit_limit: int = 5
    threads_limit: int = 10
    replies_per_iteration: int = 3
    max_runtime_minutes: int = 1440
    personas: list[str] = Field(default_factory=list)


class JobsConfig(BaseModel):
    jobs: list[PeriodicSearchJob] = Field(default_factory=list)
