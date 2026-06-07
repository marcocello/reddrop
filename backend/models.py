from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import AliasChoices, BaseModel, Field, model_validator


class Conversation(BaseModel):
    id: UUID = Field(
        default_factory=uuid4, validation_alias=AliasChoices("id", "conversation_id")
    )
    thread_id: str
    title: str
    subreddit: str
    url: str
    score: int
    num_comments: int
    created_utc: float | int | str | None = None
    published_at: str | None = None
    selftext: str = ""
    semantic_similarity: float = 0.0
    user_has_commented: bool = False
    reply: str = ""


class SearchArtifact(BaseModel):
    id: str = Field(default="", validation_alias=AliasChoices("id", "job_id"))
    name: str
    topic: str = ""
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
    linkedin_username: str = ""
    linkedin_email: str = ""
    linkedin_password: str = ""
    linkedin_headless: str = "true"
    linkedin_user_data_dir: str = ".reddrop/auth/linkedin"
    linkedin_wait_seconds: str = "120"


class SearchPlan(BaseModel):
    topics: list[str] = Field(default_factory=list)
    subreddits: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    query: str | None = None


_DEFAULT_REDDIT_SUBREDDITS = [
    "startups",
    "entrepreneur",
    "smallbusiness",
    "business",
    "marketing",
]


def normalize_search_plan_for_social(
    *,
    social: Literal["reddit", "linkedin"],
    topic: str,
    raw: object,
) -> SearchPlan:
    if isinstance(raw, SearchPlan):
        payload = raw.model_dump(mode="python")
    elif isinstance(raw, dict):
        payload = raw
    elif raw is None:
        payload = {}
    else:
        payload = {}

    def _normalize_plan_items(
        values: object, *, lowercase: bool = False, strip_r_prefix: bool = False
    ) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not value:
                continue
            if strip_r_prefix and value.lower().startswith("r/"):
                value = value[2:].strip()
                if not value:
                    continue
            compare = value.lower() if lowercase else value
            if compare in seen:
                continue
            seen.add(compare)
            normalized.append(compare if lowercase else value)
        return normalized

    plan = SearchPlan.model_validate(payload)
    topic_text = str(topic or "").strip()

    if social == "linkedin":
        queries = _normalize_plan_items(plan.queries)
        query = str(plan.query or "").strip()
        if not query and queries:
            query = queries[0]
        if not query:
            query = topic_text
        if query:
            queries = [query] + [item for item in queries if item != query]
        if not queries and query:
            queries = [query]
        return SearchPlan(
            topics=[],
            subreddits=[],
            queries=queries,
            query=query or None,
        )

    topics = _normalize_plan_items(plan.topics)
    if not topics and topic_text:
        topics = [topic_text]

    queries = _normalize_plan_items(plan.queries)
    if not queries and topic_text:
        queries = [topic_text]

    subreddits = _normalize_plan_items(
        plan.subreddits,
        lowercase=True,
        strip_r_prefix=True,
    )
    if not subreddits:
        subreddits = list(_DEFAULT_REDDIT_SUBREDDITS)

    return SearchPlan(
        topics=topics,
        subreddits=subreddits,
        queries=queries,
        query=None,
    )


class SearchQuery(BaseModel):
    id: str = Field(validation_alias=AliasChoices("id", "search_query_id"))
    name: str
    social: Literal["reddit", "linkedin"] = "reddit"
    topic: str
    plan: SearchPlan
    status: Literal["draft", "saved"] = "draft"
    version: int = 0
    created_at: datetime
    updated_at: datetime
    saved_at: datetime | None = None

    @model_validator(mode="after")
    def _normalize(self) -> "SearchQuery":
        self.name = str(self.name or "").strip()
        self.topic = str(self.topic or "").strip()
        self.plan = normalize_search_plan_for_social(
            social=self.social,
            topic=self.topic,
            raw=self.plan,
        )
        if self.status == "saved":
            self.version = max(1, int(self.version or 1))
        else:
            self.version = max(0, int(self.version or 0))
            self.saved_at = None
        return self


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
    scroll_rounds: int | None = None
    sort_by: str | None = None
    selector_wait_timeout_seconds: int | None = None
    detail_wait_timeout_seconds: int | None = None
    max_comment_expand_clicks: int | None = None
    detail_enrich_top_k: int | None = None
    search_plan: SearchPlan | None = None
    search_query_id: str | None = None
    search_query_version: int | None = None
    replies_per_iteration: int | None = None
    max_runtime_minutes: int = 1440
    personas: list[str] | None = None

    @staticmethod
    def _positive_int(value: object, fallback: int) -> int:
        try:
            parsed = int(str(value))
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

    @staticmethod
    def _normalized_plan_items(
        values: object, *, lowercase: bool = False, strip_r_prefix: bool = False
    ) -> list[str]:
        if not isinstance(values, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not value:
                continue
            if strip_r_prefix and value.lower().startswith("r/"):
                value = value[2:].strip()
                if not value:
                    continue
            compare = value.lower() if lowercase else value
            if compare in seen:
                continue
            seen.add(compare)
            normalized.append(compare if lowercase else value)
        return normalized

    def _normalized_search_plan(self, raw: object) -> SearchPlan:
        return normalize_search_plan_for_social(
            social=self.social,
            topic=str(self.topic or "").strip(),
            raw=raw,
        )

    @model_validator(mode="after")
    def _normalize_for_job_type(self) -> "PeriodicSearchJob":
        self.max_runtime_minutes = self._positive_int(self.max_runtime_minutes, 1440)
        if self.job_type == "search":
            normalized_topic = str(self.topic or "").strip()
            self.topic = normalized_topic or None
            self.source_job_id = None
            self.threads_limit = self._positive_int(self.threads_limit, 10)
            if self.social == "linkedin":
                self.time_filter = (
                    str(self.time_filter or "week").strip().lower() or "week"
                )
                if self.time_filter not in {
                    "hour",
                    "day",
                    "week",
                    "month",
                    "year",
                    "all",
                }:
                    self.time_filter = "week"
                self.subreddit_limit = None
                self.scroll_rounds = self._positive_int(self.scroll_rounds, 15)
                self.sort_by = (
                    str(self.sort_by or "relevance").strip().lower() or "relevance"
                )
                if self.sort_by not in {"relevance", "recency", "engagement"}:
                    self.sort_by = "relevance"
                self.selector_wait_timeout_seconds = self._positive_int(
                    self.selector_wait_timeout_seconds, 20
                )
                self.detail_wait_timeout_seconds = self._positive_int(
                    self.detail_wait_timeout_seconds, 8
                )
                self.max_comment_expand_clicks = self._positive_int(
                    self.max_comment_expand_clicks, 30
                )
                self.detail_enrich_top_k = self._positive_int(
                    self.detail_enrich_top_k, 12
                )
            else:
                self.time_filter = str(self.time_filter or "week").strip() or "week"
                self.subreddit_limit = self._positive_int(self.subreddit_limit, 5)
                self.scroll_rounds = None
                self.sort_by = None
                self.selector_wait_timeout_seconds = None
                self.detail_wait_timeout_seconds = None
                self.max_comment_expand_clicks = None
                self.detail_enrich_top_k = None
            if self.search_plan is None:
                self.search_plan = None
            else:
                self.search_plan = self._normalized_search_plan(self.search_plan)
            self.search_query_id = str(self.search_query_id or "").strip() or None
            if self.search_query_version is None:
                self.search_query_version = None
            else:
                self.search_query_version = self._positive_int(
                    self.search_query_version, 1
                )
            self.min_similarity_score = None
            self.replies_per_iteration = None
            self.personas = None
            return self

        self.topic = None
        self.time_filter = None
        self.subreddit_limit = None
        self.threads_limit = None
        self.scroll_rounds = None
        self.sort_by = None
        self.selector_wait_timeout_seconds = None
        self.detail_wait_timeout_seconds = None
        self.max_comment_expand_clicks = None
        self.detail_enrich_top_k = None
        self.search_plan = None
        self.search_query_id = None
        self.search_query_version = None
        self.min_similarity_score = self._bounded_float(self.min_similarity_score, 0.35)
        self.replies_per_iteration = self._positive_int(self.replies_per_iteration, 3)
        self.personas = self._normalized_personas(self.personas)
        return self


class JobsConfig(BaseModel):
    jobs: list[PeriodicSearchJob] = Field(default_factory=list)


class SearchQueriesConfig(BaseModel):
    search_queries: list[SearchQuery] = Field(default_factory=list)


class AuthsConfig(BaseModel):
    auths: list[AuthProfile] = Field(default_factory=list)
