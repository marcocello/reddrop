from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import praw

from ..config.log import setup_logging
from .llm_service import LLMServiceError, llm_service
from .settings_store import SettingsStore

logger = setup_logging(__name__)


def _never_stop() -> bool:
    return False


class RedditService:
    """Service for Reddit discovery and posting."""

    _DEFAULT_SUBREDDITS = ["startups", "entrepreneur", "smallbusiness", "business", "marketing"]

    @staticmethod
    def validate_credentials(
        *,
        client_id: str,
        client_secret: str,
        user_agent: str,
        username: str,
        password: str,
    ) -> bool:
        if not client_id.strip() or not client_secret.strip():
            return False
        if not username.strip() or not password.strip():
            return False
        try:
            reddit = praw.Reddit(
                client_id=client_id.strip(),
                client_secret=client_secret.strip(),
                user_agent=user_agent.strip() or "Reddrop:v1.0",
                username=username.strip(),
                password=password.strip(),
                check_for_async=False,
            )
            me = reddit.user.me()
            if me is None:
                return False
            return str(me).strip().lower() == username.strip().lower()
        except Exception:
            return False

    def __init__(self, auth: Any | None = None) -> None:
        self.initialization_error = ""
        try:
            SettingsStore().load()
            if auth is None:
                client_id = (os.getenv("REDDIT_CLIENT_ID") or "").strip()
                client_secret = (os.getenv("REDDIT_CLIENT_SECRET") or "").strip()
                user_agent = (os.getenv("REDDIT_USER_AGENT") or "Reddrop:v1.0").strip()
                username = (os.getenv("REDDIT_USERNAME") or "").strip()
                password = (os.getenv("REDDIT_PASSWORD") or "").strip()
            else:
                client_id = str(getattr(auth, "reddit_client_id", "")).strip()
                client_secret = str(getattr(auth, "reddit_client_secret", "")).strip()
                user_agent = str(getattr(auth, "reddit_user_agent", "")).strip() or "Reddrop:v1.0"
                username = str(getattr(auth, "reddit_username", "")).strip()
                password = str(getattr(auth, "reddit_password", "")).strip()

            if not all([client_id, client_secret]):
                raise ValueError("Missing Reddit API credentials.")

            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                username=(username or None),
                password=(password or None),
                check_for_async=False,
            )
            self.reddit.user.me()
        except Exception:
            self.reddit = None
            self.initialization_error = (
                "Reddit API credentials not configured. "
                "Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT."
            )

    def discover_relevant_threads(
        self,
        content: str,
        time_filter: str = "week",
        subreddit_limit: int = 30,
        threads_limit: int = 20,
        job_id: str | None = None,
        stop_requested: Callable[[], bool] | None = None,
        search_plan: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.reddit:
            raise RuntimeError(self.initialization_error)
        should_stop = stop_requested or _never_stop
        log_prefix = f"{job_id} - " if job_id else ""

        def _log_info(message: str, *args) -> None:
            logger.info(f"{log_prefix}{message}", *args)

        try:
            self._raise_if_stopped(should_stop)
            locked_plan = self._normalize_locked_search_plan(search_plan=search_plan, topic=content)
            locked_mode = search_plan is not None
            if locked_mode:
                content_analysis = {
                    "topics": list(locked_plan["topics"]),
                    "intent": "locked",
                    "domain": "locked",
                    "suggested_subreddits": list(locked_plan["subreddits"]),
                    "content_type": "locked",
                    "semantic_keywords": list(locked_plan["queries"]),
                }
                subreddit_names = list(locked_plan["subreddits"])
                search_queries = list(locked_plan["queries"])
                _log_info("step 1/5: Using locked search plan inputs from job configuration.")
                _log_info(
                    "step 1/5 output: topics=%s, subreddits=%s, search_queries=%s.",
                    self._preview_values(content_analysis.get("topics"), max_items=10),
                    self._preview_values(subreddit_names, max_items=10),
                    self._preview_values(search_queries, max_items=10),
                )
            else:
                _log_info("step 1/5: Analyzing topic to extract intent and keywords.")
                content_analysis = self._analyze_content(content)
                self._raise_if_stopped(should_stop)

                intent = str(content_analysis.get("intent", "unknown"))
                topics = self._preview_values(content_analysis.get("topics"), max_items=5)
                keywords = self._preview_values(content_analysis.get("semantic_keywords"), max_items=5)
                _log_info(
                    "step 1/5 output: topic=%r, intent=%s, topics=%s, keywords=%s.",
                    content,
                    intent,
                    topics,
                    keywords,
                )

                _log_info("step 2/5: Discovering relevant subreddits (limit=%s).", subreddit_limit)
                discovered_subreddits = self._find_relevant_subreddits(
                    content_analysis,
                    limit=subreddit_limit,
                    stop_requested=should_stop,
                )
                self._raise_if_stopped(should_stop)
                if not discovered_subreddits:
                    _log_info("No subreddits discovered from analysis. Using default subreddit set.")
                    discovered_subreddits = [{"name": name} for name in self._DEFAULT_SUBREDDITS[:subreddit_limit]]

                subreddit_names = [sub["name"] for sub in discovered_subreddits]
                _log_info("step 2/5 output: discovered_subreddits=%s.", self._preview_values(subreddit_names, max_items=10))

                _log_info("step 3/5: Generating Reddit search queries.")
                search_queries = self._generate_search_queries(content_analysis)
                self._raise_if_stopped(should_stop)
                if not search_queries:
                    search_queries = content_analysis.get("topics", [])[:3] or [content]
                    _log_info("No LLM search queries generated. Falling back to topic-based queries.")
                _log_info("step 3/5 output: search_queries=%s.", self._preview_values(search_queries, max_items=10))

            _log_info(
                "step 4/5: Searching %s subreddit(s) (time_filter=%s, threads_limit=%s).",
                len(subreddit_names),
                time_filter,
                threads_limit,
            )
            candidate_threads = self._search_threads_directly(
                subreddits_to_search=subreddit_names,
                search_queries=search_queries,
                content_analysis=content_analysis,
                time_filter=time_filter,
                threads_limit=threads_limit,
                job_id=job_id,
                stop_requested=should_stop,
            )
            self._raise_if_stopped(should_stop)
            candidate_ids = [thread.get("id") for thread in candidate_threads if isinstance(thread, dict)]
            _log_info(
                "step 4/5 output: candidate_threads=%s, thread_ids=%s.",
                len(candidate_threads),
                self._preview_values(candidate_ids, max_items=10),
            )
            _log_info("step 5/5: Ranking %s candidate thread(s).", len(candidate_threads))
            self._raise_if_stopped(should_stop)
            if not candidate_threads:
                _log_info("No candidate threads found for this search.")
                return []

            return self._llm_filter_and_rank_threads(
                content=content,
                content_analysis=content_analysis,
                threads=candidate_threads,
                stop_requested=should_stop,
            )
        except Exception as exc:
            raise RuntimeError(f"Reddit discovery failed: {exc}") from exc

    def generate_search_plan(self, *, topic: str, subreddit_limit: int = 5) -> dict[str, list[str]]:
        normalized_topic = str(topic or "").strip()
        if not normalized_topic:
            raise ValueError("topic is required")

        normalized_limit = max(1, int(subreddit_limit or 1))

        try:
            content_analysis = self._analyze_content(normalized_topic)
        except Exception:
            content_analysis = {
                "topics": [normalized_topic],
                "suggested_subreddits": list(self._DEFAULT_SUBREDDITS[:normalized_limit]),
                "semantic_keywords": [normalized_topic],
            }

        topics = self._normalize_plan_items(content_analysis.get("topics"), fallback=[normalized_topic])
        try:
            generated_queries = self._generate_search_queries(content_analysis)
        except Exception:
            generated_queries = []
        queries = self._normalize_plan_items(generated_queries, fallback=topics[:3] or [normalized_topic])

        subreddits: list[str] = []
        if self.reddit is not None:
            try:
                discovered = self._find_relevant_subreddits(content_analysis, limit=normalized_limit)
                subreddits = self._normalize_subreddit_names([item.get("name") for item in discovered])
            except Exception:
                subreddits = []
        if not subreddits:
            suggested = content_analysis.get("suggested_subreddits")
            subreddits = self._normalize_subreddit_names(suggested)
        if not subreddits:
            subreddits = list(self._DEFAULT_SUBREDDITS[:normalized_limit])

        return {
            "topics": topics,
            "subreddits": subreddits[:normalized_limit],
            "queries": queries,
        }

    def _analyze_content(self, content: str) -> dict[str, Any]:
        try:
            system_message = (
                "You are an expert content analyzer that extracts structured information from text. "
                "Return valid JSON only."
            )
            human_message = f"""
            Analyze this content and extract:
            1. Main topics (3-5 keywords)
            2. Intent (question, discussion, advice, sharing)
            3. Domain/field
            4. Relevant subreddit suggestions
            5. Content type

            Content: "{content}"

            Return JSON:
            {{
              "topics": ["topic1", "topic2"],
              "intent": "discussion",
              "domain": "business",
              "suggested_subreddits": ["startups", "entrepreneur"],
              "content_type": "professional",
              "semantic_keywords": ["keyword1", "keyword2"]
            }}
            """
            response = llm_service.generate_response_sync(system_message, human_message)
        except LLMServiceError:
            raise

        try:
            return json.loads(response)
        except Exception:
            tokens = [token.lower() for token in re.findall(r"[a-zA-Z0-9]+", content) if len(token) > 2]
            dedup_tokens: list[str] = []
            seen = set()
            for token in tokens:
                if token in seen:
                    continue
                seen.add(token)
                dedup_tokens.append(token)
            return {
                "topics": dedup_tokens[:5] or [content.strip() or "topic"],
                "intent": "discussion",
                "domain": "general",
                "suggested_subreddits": ["startups", "entrepreneur", "smallbusiness", "business", "marketing"],
                "content_type": "casual",
                "semantic_keywords": dedup_tokens[:8],
            }

    def _find_relevant_subreddits(
        self,
        content_analysis: dict[str, Any],
        limit: int | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.reddit:
            return []
        should_stop = stop_requested or _never_stop

        all_subreddits: list[dict[str, Any]] = []
        for sub_name in content_analysis.get("suggested_subreddits", []):
            self._raise_if_stopped(should_stop)
            try:
                subreddit = self.reddit.subreddit(sub_name.removeprefix("r/"))
                if subreddit.subscribers > 1000:
                    all_subreddits.append(
                        {
                            "name": subreddit.display_name,
                            "title": subreddit.title,
                            "subscribers": subreddit.subscribers,
                            "description": getattr(subreddit, "public_description", "")[:200],
                            "source": "llm_suggested",
                        }
                    )
            except Exception:
                continue

        for topic in content_analysis.get("topics", []):
            self._raise_if_stopped(should_stop)
            try:
                for subreddit in list(self.reddit.subreddits.search(topic, limit=10)):
                    self._raise_if_stopped(should_stop)
                    if subreddit.subscribers > 5000:
                        all_subreddits.append(
                            {
                                "name": subreddit.display_name,
                                "title": subreddit.title,
                                "subscribers": subreddit.subscribers,
                                "description": getattr(subreddit, "public_description", "")[:200],
                                "source": "search",
                            }
                        )
            except Exception:
                continue

        seen = set()
        unique = []
        for sub in all_subreddits:
            if sub["name"] not in seen:
                seen.add(sub["name"])
                unique.append(sub)
        return unique[:limit]

    def _generate_search_queries(self, content_analysis: dict[str, Any]) -> list[str]:
        try:
            system_message = "You are a Reddit search expert."
            human_message = f"""
            Generate advanced Reddit search queries.
            Topics: {content_analysis.get('topics', [])}
            Domain: {content_analysis.get('domain', 'general')}
            Return JSON array only, example: ["query1", "query2"].
            """
            response = llm_service.generate_response_sync(system_message, human_message)
        except LLMServiceError:
            raise
        except Exception:
            topics = [item for item in content_analysis.get("topics", []) if isinstance(item, str) and item.strip()]
            return topics[:5]

        try:
            matches = re.findall(r"```json\\n(.*?)```", response, re.DOTALL)
            source = matches[0] if matches else response
            parsed = json.loads(source)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            topics = [item for item in content_analysis.get("topics", []) if isinstance(item, str) and item.strip()]
            return topics[:5]

    def _search_threads_directly(
        self,
        subreddits_to_search: list[str],
        search_queries: list[str],
        content_analysis: dict[str, Any],
        time_filter: str = "week",
        threads_limit: int = 20,
        job_id: str | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.reddit:
            return []
        should_stop = stop_requested or _never_stop

        _ = content_analysis
        all_threads: list[dict[str, Any]] = []
        for sub_name in list(set(subreddits_to_search)):
            self._raise_if_stopped(should_stop)
            try:
                prefix = f"{job_id} - " if job_id else ""
                logger.info("%sScanning subreddit r/%s with %s query term(s).", prefix, sub_name, len(search_queries))
                subreddit = self.reddit.subreddit(sub_name)
                for query in search_queries:
                    self._raise_if_stopped(should_stop)
                    try:
                        for submission in subreddit.search(query, time_filter=time_filter, limit=threads_limit):
                            self._raise_if_stopped(should_stop)
                            if submission.stickied or submission.distinguished:
                                continue
                            all_threads.append(
                                self._create_thread_data_from_submission(
                                    submission,
                                    search_query=query,
                                    stop_requested=should_stop,
                                )
                            )
                    except Exception:
                        continue
            except Exception:
                continue

        seen_ids = set()
        unique_threads = []
        for thread in all_threads:
            if thread["id"] not in seen_ids:
                seen_ids.add(thread["id"])
                unique_threads.append(thread)
        return unique_threads

    def _create_thread_data_from_submission(
        self,
        submission,
        search_query: str = "",
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        should_stop = stop_requested or _never_stop
        user_has_commented = False
        try:
            self._raise_if_stopped(should_stop)
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                self._raise_if_stopped(should_stop)
                if comment.author and comment.author == self.reddit.user.me():
                    user_has_commented = True
                    break
        except RuntimeError:
            raise
        except Exception:
            pass

        created_utc = self._normalize_submission_created_utc(getattr(submission, "created_utc", None))
        return {
            "id": submission.id,
            "title": submission.title,
            "subreddit": submission.subreddit.display_name,
            "url": f"https://reddit.com{submission.permalink}",
            "score": submission.score,
            "num_comments": submission.num_comments,
            "created_utc": created_utc,
            "selftext": submission.selftext,
            "author": str(submission.author) if submission.author else "[deleted]",
            "is_self": submission.is_self,
            "full_text": f"{submission.title} {submission.selftext}",
            "search_query": search_query,
            "user_has_commented": user_has_commented,
        }

    @staticmethod
    def _normalize_submission_created_utc(raw: object) -> float | None:
        try:
            value = float(raw)  # type: ignore[arg-type]
        except Exception:
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _preview_values(values: Any, max_items: int = 5) -> str:
        if not isinstance(values, list):
            return "[]"
        result: list[str] = []
        for value in values[:max_items]:
            if isinstance(value, str):
                result.append(value)
                continue
            if isinstance(value, (int, float, bool)):
                result.append(str(value))
                continue
            if isinstance(value, dict):
                name = value.get("name")
                if isinstance(name, str):
                    result.append(name)
                    continue
            result.append(str(value))
        if len(values) > max_items:
            result.append("...")
        return "[" + ", ".join(result) + "]"

    def _llm_filter_and_rank_threads(
        self,
        content: str,
        content_analysis: dict[str, Any],
        threads: list[dict[str, Any]],
        stop_requested: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        _ = content_analysis
        if not threads:
            return []
        should_stop = stop_requested or _never_stop
        self._raise_if_stopped(should_stop)
        try:
            from sentence_transformers import SentenceTransformer, util

            model = SentenceTransformer("all-MiniLM-L6-v2")
            self._raise_if_stopped(should_stop)
            original_embedding = model.encode(content, convert_to_tensor=True)
            thread_texts = [f"{thread['title']} {thread['selftext']}" for thread in threads]
            self._raise_if_stopped(should_stop)
            thread_embeddings = model.encode(thread_texts, convert_to_tensor=True)
            similarity_scores = util.pytorch_cos_sim(original_embedding, thread_embeddings).squeeze().tolist()

            for i, thread in enumerate(threads):
                self._raise_if_stopped(should_stop)
                thread["semantic_similarity"] = similarity_scores[i]
            threads.sort(key=lambda t: (t.get("semantic_similarity", 0.0),), reverse=True)
            return threads
        except Exception:
            return threads

    def post_comment(self, thread_id: str, comment_text: str) -> bool:
        if not self.reddit:
            return False
        try:
            submission = self.reddit.submission(id=thread_id)
            submission.reply(comment_text)
            return True
        except Exception:
            return False

    @staticmethod
    def _raise_if_stopped(stop_requested: Callable[[], bool]) -> None:
        if stop_requested():
            raise RuntimeError("Run stopped by user request.")

    @staticmethod
    def _normalize_plan_items(values: Any, *, fallback: list[str] | None = None) -> list[str]:
        source = values if isinstance(values, list) else []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in source:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if not token:
                continue
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(token)
        if normalized:
            return normalized
        return list(fallback or [])

    @staticmethod
    def _normalize_subreddit_names(values: Any) -> list[str]:
        source = values if isinstance(values, list) else []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in source:
            if not isinstance(item, str):
                continue
            token = item.strip().lower()
            if token.startswith("r/"):
                token = token[2:]
            token = token.strip()
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            normalized.append(token)
        return normalized

    def _normalize_locked_search_plan(self, *, search_plan: dict[str, Any] | None, topic: str) -> dict[str, list[str]]:
        topic_value = str(topic or "").strip()
        if not isinstance(search_plan, dict):
            return {
                "topics": [topic_value] if topic_value else [],
                "subreddits": list(self._DEFAULT_SUBREDDITS),
                "queries": [topic_value] if topic_value else [],
            }

        topics = self._normalize_plan_items(search_plan.get("topics"), fallback=[topic_value] if topic_value else [])
        queries = self._normalize_plan_items(search_plan.get("queries"), fallback=[topic_value] if topic_value else [])
        subreddits = self._normalize_subreddit_names(search_plan.get("subreddits"))
        if not subreddits:
            subreddits = list(self._DEFAULT_SUBREDDITS)
        return {
            "topics": topics,
            "subreddits": subreddits,
            "queries": queries,
        }
