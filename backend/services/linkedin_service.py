from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlencode, urlsplit, urlunsplit

from ..config.log import setup_logging
from .llm_service import LLMServiceError, llm_service
from .settings_store import SettingsStore

logger = setup_logging(__name__)
_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _never_stop() -> bool:
    return False


def _url_requires_login(url: str) -> bool:
    value = (url or "").strip().lower()
    if not value:
        return False
    return (
        "linkedin.com/login" in value
        or "/checkpoint/" in value
        or "/challenge/" in value
        or "authwall" in value
    )


def _url_is_authenticated(url: str) -> bool:
    value = (url or "").strip().lower()
    if "linkedin.com" not in value:
        return False
    return not _url_requires_login(value)


def _url_points_to_feed(url: str) -> bool:
    value = (url or "").strip().lower()
    return "linkedin.com/feed" in value


def _probe_context_authentication(context) -> bool:
    try:
        response = context.request.get(
            "https://www.linkedin.com/feed/",
            timeout=1500,
            fail_on_status_code=False,
        )
    except Exception:
        return False
    final_url = (getattr(response, "url", "") or "").strip().lower()
    return _url_is_authenticated(final_url)


def _has_linkedin_session_cookie(context) -> bool:
    cookies = context.cookies(["https://www.linkedin.com"])
    return any(
        cookie.get("name") == "li_at" and str(cookie.get("value", "")).strip()
        for cookie in cookies
    )


def _parse_headless_flag(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    return True


def _resolve_profile_dir(raw_dir: str) -> Path:
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
    return resolved


def _default_login_launcher(
    user_data_dir: str, login_url: str, wait_seconds: int
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Playwright is not installed. Install with: pip install playwright && playwright install chromium"
        ) from exc

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
        )
        try:
            pages = [page for page in context.pages if not page.is_closed()]
            page = pages[0] if pages else context.new_page()
            for extra_page in pages[1:]:
                extra_page.close()
            page.goto(login_url, wait_until="domcontentloaded")

            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                open_pages = [
                    candidate
                    for candidate in context.pages
                    if not candidate.is_closed()
                ]
                if not open_pages:
                    raise RuntimeError(
                        "LinkedIn login window was closed before login completed. Auth object was not created."
                    )
                page = open_pages[0]
                if page.is_closed():
                    raise RuntimeError(
                        "LinkedIn login window was closed before login completed. Auth object was not created."
                    )

                if any(_url_points_to_feed(candidate.url) for candidate in open_pages):
                    return
                has_session_cookie = _has_linkedin_session_cookie(context)
                if has_session_cookie and any(
                    _url_is_authenticated(candidate.url) for candidate in open_pages
                ):
                    return
                if has_session_cookie and _probe_context_authentication(context):
                    return
                time.sleep(0.5)

            raise RuntimeError(
                "LinkedIn login not completed. Auth object was not created."
            )
        finally:
            context.close()


class LinkedinService:
    """Headless LinkedIn discovery/posting integration."""

    _POST_CARD_SELECTORS: tuple[str, ...] = (
        "li.reusable-search__result-container",
        "div.reusable-search__result-container",
        "li.search-results__result-item",
        "li[data-chameleon-result-urn]",
        "div[data-chameleon-result-urn]",
        "div.feed-shared-update-v2",
        "article.feed-shared-update-v2",
        "div[data-urn*='urn:li:activity']",
        "article[data-urn*='urn:li:activity']",
        "div[data-id*='urn:li:activity']",
        "article[data-id*='urn:li:activity']",
        "div.occludable-update",
        "li.occludable-update",
    )
    _SEARCH_WAIT_SELECTORS: tuple[str, ...] = _POST_CARD_SELECTORS + (
        "[data-testid='expandable-text-box']",
        "[data-view-name='feed-commentary']",
    )
    _MAX_NO_NEW_PHASE_ROUNDS: int = 6
    _DEFAULT_DETAIL_ENRICH_TOP_K: int = 12
    _SEMANTIC_MODEL: Any | None = None
    _PROFILE_LOCKS: dict[str, threading.Lock] = {}
    _PROFILE_LOCKS_GUARD = threading.Lock()

    def __init__(self, auth: Any | None = None) -> None:
        SettingsStore().load()
        if auth is None:
            self.headless = _parse_headless_flag(os.getenv("LINKEDIN_HEADLESS", "true"))
            configured_dir = (
                os.getenv("LINKEDIN_USER_DATA_DIR") or ".reddrop/auth/linkedin"
            ).strip()
        else:
            headless_raw = str(getattr(auth, "linkedin_headless", "true"))
            self.headless = _parse_headless_flag(headless_raw)
            configured_dir = str(
                getattr(auth, "linkedin_user_data_dir", "") or ".reddrop/auth/linkedin"
            ).strip()
        self.user_data_dir = str(_resolve_profile_dir(configured_dir))

    @staticmethod
    def login_and_save_session(
        *,
        user_data_dir: str,
        wait_seconds: int = 120,
        login_url: str = "https://www.linkedin.com/login",
        launcher: Callable[[str, str, int], None] | None = None,
    ) -> dict[str, Any]:
        if wait_seconds <= 0:
            raise ValueError("--wait-seconds must be greater than 0.")
        profile_dir = _resolve_profile_dir(user_data_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        effective_launcher = launcher or _default_login_launcher
        effective_launcher(str(profile_dir), login_url, wait_seconds)
        return {
            "status": "session_saved",
            "user_data_dir": str(profile_dir),
            "login_url": login_url,
        }

    @staticmethod
    def validate_saved_session(*, user_data_dir: str) -> bool:
        profile_dir = _resolve_profile_dir(user_data_dir)
        if not profile_dir.exists() or not profile_dir.is_dir():
            return False
        for cookie_db in LinkedinService._cookie_db_candidates(profile_dir):
            if LinkedinService._has_linkedin_cookie(cookie_db):
                return True
        return False

    @staticmethod
    def resolve_saved_session_username(
        *, user_data_dir: str, timeout_ms: int = 6000
    ) -> str:
        profile_dir = _resolve_profile_dir(user_data_dir)
        if not profile_dir.exists() or not profile_dir.is_dir():
            return ""

        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            return ""

        try:
            with sync_playwright() as playwright:
                context = playwright.chromium.launch_persistent_context(
                    user_data_dir=str(profile_dir),
                    headless=True,
                )
                try:
                    pages = [page for page in context.pages if not page.is_closed()]
                    page = pages[0] if pages else context.new_page()

                    username = LinkedinService._username_from_profile_url(page.url)
                    if username:
                        return username

                    try:
                        page.goto(
                            "https://www.linkedin.com/in/me/",
                            wait_until="domcontentloaded",
                            timeout=timeout_ms,
                        )
                    except Exception:
                        return ""

                    username = LinkedinService._username_from_profile_url(page.url)
                    if username:
                        return username

                    # Fallback: inspect top-nav profile link when redirect URL did not resolve.
                    profile_href = page.evaluate(
                        """
() => {
  const links = Array.from(document.querySelectorAll('a[href*="/in/"]'));
  for (const link of links) {
    const href = String(link.getAttribute('href') || link.href || '').trim();
    if (!href) continue;
    if (href.includes('/in/me')) continue;
    if (href.includes('/in/')) return href;
  }
  return '';
}
                        """
                    )
                    username = LinkedinService._username_from_profile_url(
                        str(profile_href)
                    )
                    if username:
                        return username

                    return LinkedinService._username_from_voyager_api(context)
                finally:
                    context.close()
        except Exception:
            return ""

    def discover_relevant_threads(
        self,
        content: str,
        time_filter: str = "week",
        subreddit_limit: int = 30,
        threads_limit: int = 20,
        scroll_rounds: int = 15,
        sort_by: str = "relevance",
        selector_wait_timeout_seconds: int = 20,
        detail_wait_timeout_seconds: int = 8,
        max_comment_expand_clicks: int = 30,
        detail_enrich_top_k: int | None = None,
        job_id: str | None = None,
        stop_requested: Callable[[], bool] | None = None,
        search_plan: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        _ = subreddit_limit
        should_stop = stop_requested or _never_stop
        self._raise_if_stopped(should_stop)

        fallback_query = content.strip()
        if not fallback_query:
            return []
        query_passes = self._query_passes_from_search_plan(
            search_plan=search_plan,
            fallback=fallback_query,
        )
        if not query_passes:
            return []
        query = query_passes[0]

        rounds = max(1, int(scroll_rounds or 1))
        max_posts = max(1, int(threads_limit or 1))
        selector_wait_timeout = max(1, int(selector_wait_timeout_seconds or 1))
        detail_wait_timeout = max(1, int(detail_wait_timeout_seconds or 1))
        comment_expand_clicks = max(1, int(max_comment_expand_clicks or 1))
        detail_enrich_limit = self._detail_enrich_top_k(detail_enrich_top_k)
        normalized_time_filter = self._normalize_time_filter(time_filter)
        normalized_sort_by = self._normalize_sort_by(sort_by)
        now_utc = time.time()

        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise RuntimeError(
                "LinkedIn headless search requires playwright. "
                "Install it with: pip install playwright && playwright install chromium"
            ) from exc

        try:
            with self._profile_runtime_lock():
                with sync_playwright() as playwright:
                    context = self._launch_context(playwright)
                    try:
                        prefix = f"{job_id} - " if job_id else ""
                        all_threads: list[dict[str, Any]] = []
                        seen_thread_ids: set[str] = set()
                        seen_urls: set[str] = set()
                        aggregate = self._empty_discovery_counters()

                        for variant_index, query_variant in enumerate(
                            query_passes, start=1
                        ):
                            self._raise_if_stopped(should_stop)
                            variant_threads, counters = (
                                self._discover_threads_for_query(
                                    context=context,
                                    query=query_variant,
                                    rounds=rounds,
                                    max_posts=max_posts,
                                    selector_wait_timeout=selector_wait_timeout,
                                    detail_wait_timeout=detail_wait_timeout,
                                    comment_expand_clicks=comment_expand_clicks,
                                    detail_enrich_top_k=detail_enrich_limit,
                                    normalized_time_filter=normalized_time_filter,
                                    normalized_sort_by=normalized_sort_by,
                                    now_utc=now_utc,
                                    stop_requested=should_stop,
                                )
                            )
                            self._merge_discovery_counters(aggregate, counters)

                            added = 0
                            for row in variant_threads:
                                thread_id = str(row.get("id", "")).strip()
                                canonical_url = str(row.get("url", "")).strip()
                                if (
                                    thread_id in seen_thread_ids
                                    or canonical_url in seen_urls
                                ):
                                    aggregate["cross_query_duplicates"] += 1
                                    continue
                                seen_thread_ids.add(thread_id)
                                seen_urls.add(canonical_url)
                                all_threads.append(row)
                                added += 1
                                if len(all_threads) >= max_posts:
                                    break

                            logger.info(
                                "%sLinkedIn query pass %s/%s: query=%r added=%s total=%s phase1=%s phase2=%s "
                                "scanned=%s kept=%s invalid_url=%s unknown_time=%s dropped_time_filter=%s "
                                "duplicates=%s.",
                                prefix,
                                variant_index,
                                len(query_passes),
                                query_variant,
                                added,
                                len(all_threads),
                                counters["phase1_rows"],
                                counters["phase2_links"],
                                counters["scanned_rows"],
                                counters["kept_rows"],
                                counters["invalid_url"],
                                counters["unknown_time_rows"],
                                counters["dropped_time_filter"],
                                counters["duplicate_rows"],
                            )
                            if len(all_threads) >= max_posts:
                                break

                        threads = self._sort_threads(
                            all_threads, sort_by=normalized_sort_by
                        )[:max_posts]
                        similarity_topic = self._similarity_topic(
                            search_plan=search_plan, fallback=content
                        )
                        threads = self._apply_semantic_similarity_scores(
                            topic=similarity_topic,
                            rows=threads,
                            stop_requested=should_stop,
                        )
                        logger.info(
                            "%sLinkedIn discovery completed: query=%r variants=%s results=%s filter=%s sort=%s "
                            "phase1=%s phase2=%s scanned=%s kept=%s invalid_url=%s unknown_time=%s "
                            "dropped_time_filter=%s duplicates=%s cross_query_duplicates=%s.",
                            prefix,
                            query,
                            len(query_passes),
                            len(threads),
                            normalized_time_filter,
                            normalized_sort_by,
                            aggregate["phase1_rows"],
                            aggregate["phase2_links"],
                            aggregate["scanned_rows"],
                            aggregate["kept_rows"],
                            aggregate["invalid_url"],
                            aggregate["unknown_time_rows"],
                            aggregate["dropped_time_filter"],
                            aggregate["duplicate_rows"],
                            aggregate["cross_query_duplicates"],
                        )
                        return threads
                    finally:
                        context.close()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"LinkedIn discovery failed: {exc}") from exc

    @contextmanager
    def _profile_runtime_lock(self):
        key = str(Path(self.user_data_dir).expanduser())
        with self._PROFILE_LOCKS_GUARD:
            lock = self._PROFILE_LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                self._PROFILE_LOCKS[key] = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    @staticmethod
    def _similarity_topic(*, search_plan: dict[str, Any] | None, fallback: str) -> str:
        fallback_value = " ".join(str(fallback or "").split()).strip()
        if not isinstance(search_plan, dict):
            return fallback_value

        topics = search_plan.get("topics")
        if isinstance(topics, list):
            for item in topics:
                if not isinstance(item, str):
                    continue
                value = " ".join(item.split()).strip()
                if value:
                    return value

        topic = search_plan.get("topic")
        if isinstance(topic, str):
            value = " ".join(topic.split()).strip()
            if value:
                return value

        return fallback_value

    @classmethod
    def _embedding_model(cls) -> Any:
        if cls._SEMANTIC_MODEL is None:
            from sentence_transformers import SentenceTransformer

            cls._SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._SEMANTIC_MODEL

    def _apply_semantic_similarity_scores(
        self,
        *,
        topic: str,
        rows: list[dict[str, Any]],
        stop_requested: Callable[[], bool],
    ) -> list[dict[str, Any]]:
        if not rows:
            return rows

        for row in rows:
            row["semantic_similarity"] = float(
                row.get("semantic_similarity", 0.0) or 0.0
            )

        normalized_topic = " ".join(str(topic or "").split()).strip()
        if not normalized_topic:
            return rows

        try:
            from sentence_transformers import util

            self._raise_if_stopped(stop_requested)
            model = self._embedding_model()
            topic_embedding = model.encode(normalized_topic, convert_to_tensor=True)
            thread_texts = [
                f"{row.get('title', '')} {row.get('selftext', '')}".strip()
                for row in rows
            ]
            self._raise_if_stopped(stop_requested)
            thread_embeddings = model.encode(thread_texts, convert_to_tensor=True)
            similarity_scores = (
                util.pytorch_cos_sim(topic_embedding, thread_embeddings)
                .squeeze()
                .tolist()
            )

            if isinstance(similarity_scores, (int, float)):
                scores_list = [float(similarity_scores)]
            elif isinstance(similarity_scores, list):
                scores_list = [float(score or 0.0) for score in similarity_scores]
            else:
                scores_list = []

            for index, row in enumerate(rows):
                self._raise_if_stopped(stop_requested)
                if index < len(scores_list):
                    row["semantic_similarity"] = scores_list[index]
        except Exception:
            return rows
        return rows

    @staticmethod
    def _empty_discovery_counters() -> dict[str, int]:
        return {
            "phase1_rows": 0,
            "phase2_links": 0,
            "scanned_rows": 0,
            "kept_rows": 0,
            "invalid_url": 0,
            "duplicate_rows": 0,
            "unknown_time_rows": 0,
            "dropped_time_filter": 0,
            "cross_query_duplicates": 0,
        }

    @staticmethod
    def _merge_discovery_counters(
        target: dict[str, int], source: dict[str, int]
    ) -> None:
        for key, value in source.items():
            target[key] = int(target.get(key, 0) or 0) + int(value or 0)

    def _discover_threads_for_query(
        self,
        *,
        context: Any,
        query: str,
        rounds: int,
        max_posts: int,
        selector_wait_timeout: int,
        detail_wait_timeout: int,
        comment_expand_clicks: int,
        detail_enrich_top_k: int,
        normalized_time_filter: str,
        normalized_sort_by: str,
        now_utc: float,
        stop_requested: Callable[[], bool],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        counters = self._empty_discovery_counters()
        page = context.new_page()
        try:
            search_url = self._linkedin_search_url(
                query=query,
                time_filter=normalized_time_filter,
                sort_by=normalized_sort_by,
            )
            page.goto(search_url, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            self._ensure_authenticated(page)
            self._raise_if_authwall_page(
                page,
                context_message=(
                    "LinkedIn returned a login/auth wall in search mode. "
                    "Run LinkedIn auth login again with the same profile dir."
                ),
            )
            self._wait_for_any_selector(
                page=page,
                selectors=list(self._SEARCH_WAIT_SELECTORS),
                timeout_seconds=selector_wait_timeout,
            )
            phase_scan_limit = self._phase_scan_limit(max_posts)
            phase_1_rows = self._collect_phase_rows_with_scroll(
                page=page,
                rounds=rounds,
                max_posts=phase_scan_limit,
                wait_timeout_seconds=min(8, selector_wait_timeout),
                stop_requested=stop_requested,
            )
            phase_1_links = self._select_detail_enrichment_links(
                rows=phase_1_rows,
                max_posts=max_posts,
                detail_enrich_top_k=detail_enrich_top_k,
            )
            detail_rows_by_url = self._load_post_detail_rows(
                context=context,
                post_urls=phase_1_links,
                scroll_rounds=max(2, rounds // 2),
                selector_wait_timeout_seconds=selector_wait_timeout,
                detail_wait_timeout_seconds=detail_wait_timeout,
                max_comment_expand_clicks=comment_expand_clicks,
                stop_requested=stop_requested,
            )
            merged_rows = self._enrich_phase_rows(phase_1_rows, detail_rows_by_url)
            counters["phase1_rows"] = len(phase_1_rows)
            counters["phase2_links"] = len(phase_1_links)

            rows: list[dict[str, Any]] = []
            seen_thread_ids: set[str] = set()
            seen_urls: set[str] = set()
            for index, item in enumerate(merged_rows):
                self._raise_if_stopped(stop_requested)
                if not isinstance(item, dict):
                    continue
                counters["scanned_rows"] += 1
                canonical_url = self._canonicalize_post_url(
                    str(item.get("url", "")).strip()
                )
                if not canonical_url:
                    counters["invalid_url"] += 1
                    continue
                selftext = str(item.get("selftext", "")).strip()
                title_source = selftext or str(item.get("title", "LinkedIn post"))
                title = self._truncate_title(title_source)
                thread_id = self._thread_id_from_url(
                    canonical_url, fallback=f"linkedin-{index}"
                )
                if thread_id in seen_thread_ids or canonical_url in seen_urls:
                    counters["duplicate_rows"] += 1
                    continue

                text_score, text_comments = self._extract_engagement_metrics(selftext)
                score = self._positive_count(item.get("reactions_count")) or text_score
                num_comments = (
                    self._positive_count(item.get("comments_count")) or text_comments
                )
                created_utc, published_at = self._resolve_created_timestamp(
                    row=item,
                    selftext=selftext,
                    now_utc=now_utc,
                )
                if created_utc is None:
                    counters["unknown_time_rows"] += 1
                if not self._matches_time_filter(
                    created_utc=created_utc,
                    time_filter=normalized_time_filter,
                    now_utc=now_utc,
                ):
                    counters["dropped_time_filter"] += 1
                    continue
                seen_thread_ids.add(thread_id)
                seen_urls.add(canonical_url)
                rows.append(
                    {
                        "id": thread_id,
                        "title": title,
                        "subreddit": "linkedin",
                        "url": canonical_url,
                        "score": score,
                        "num_comments": num_comments,
                        "created_utc": created_utc,
                        "published_at": published_at,
                        "selftext": selftext,
                        "semantic_similarity": 0.0,
                        "user_has_commented": False,
                    }
                )
                counters["kept_rows"] += 1
            return rows, counters
        finally:
            close_fn = getattr(page, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    @staticmethod
    def generate_search_plan(*, topic: str) -> dict[str, object]:
        query = " ".join(str(topic or "").split()).strip()
        if not query:
            raise ValueError("topic is required")
        queries = [query]
        try:
            system_message = (
                "You are an expert LinkedIn search query planner. "
                "Return valid JSON only."
            )
            human_message = f"""
            Build a concise list of LinkedIn content-search queries for this topic:
            "{query}"

            Requirements:
            - Return exactly 5 possible queries.
            - Keep each query concise and high-signal.
            - Prefer boolean operators when useful (AND/OR/quotes).
            - Include one stricter query and one broader query when useful.
            - Return LinkedIn content-search keyword queries only.
            - Do not return URLs, site: operators, profile/company paths, or Reddit syntax.
            - Return JSON only.

            Return:
            {{
              "queries": ["...", "..."]
            }}
            """
            response = llm_service.generate_response_sync(system_message, human_message)
            llm_queries = LinkedinService._extract_search_queries_from_llm(response)
            if llm_queries:
                queries = llm_queries
        except LLMServiceError:
            pass
        except Exception:
            pass

        queries = LinkedinService._normalize_linkedin_search_queries(
            queries,
            topic=query,
            min_count=5,
            max_count=5,
        )
        return {"query": queries[0], "queries": queries}

    def _phase_rows_from_page(
        self, *, page: Any, max_posts: int
    ) -> list[dict[str, Any]]:
        limit = max(1, int(max_posts or 1))
        scan_budget = self._raw_scan_budget(limit)
        raw_rows = page.evaluate(
            f"""
() => {{
  const maxPosts = {limit};
  const scanBudget = {scan_budget};
  const normalize = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
  const firstText = (root, selectors) => {{
    for (const selector of selectors) {{
      const node = root.querySelector(selector);
      if (!node) continue;
      const text = normalize(node.textContent || node.innerText || '');
      if (text) return text;
    }}
    return '';
  }};
  const extractInt = (text) => {{
    const match = String(text || '').replace(/,/g, '').match(/\\d+(?:\\.\\d+)?\\s*[kKmM]?/);
    if (!match) return 0;
    const token = String(match[0] || '').trim().toLowerCase();
    let multiplier = 1;
    let numeric = token.replace(/\\s+/g, '');
    if (numeric.endsWith('k')) {{
      multiplier = 1000;
      numeric = numeric.slice(0, -1);
    }} else if (numeric.endsWith('m')) {{
      multiplier = 1000000;
      numeric = numeric.slice(0, -1);
    }}
    const value = parseFloat(numeric);
    if (!Number.isFinite(value) || value < 0) return 0;
    return Math.floor(value * multiplier);
  }};
  const extractPublishedIso = (node) => {{
    if (!node) return '';
    const direct = node.querySelector('time[datetime]');
    if (direct) {{
      const value = normalize(direct.getAttribute('datetime') || '');
      if (value) return value;
    }}
    for (const candidate of Array.from(node.querySelectorAll('[datetime]'))) {{
      const value = normalize(candidate.getAttribute('datetime') || '');
      if (!value) continue;
      if (String(candidate.tagName || '').toLowerCase() === 'time') return value;
    }}
    return '';
  }};
  const normalizeHref = (href) => {{
    const value = String(href || '').trim().replace(/\\\\+$/, '');
    if (!value) return '';
    if (value.startsWith('//')) return `https:${{value}}`;
    if (value.startsWith('/')) return `https://www.linkedin.com${{value}}`;
    return value;
  }};
  const decodeLinkedInCandidate = (value) => {{
    let text = String(value || '').trim();
    if (!text) return '';
    for (let i = 0; i < 2; i += 1) {{
      try {{
        const decoded = decodeURIComponent(text);
        if (decoded === text) break;
        text = decoded;
      }} catch (_err) {{
        break;
      }}
    }}
    return text
      .replace(/\\\\u002[fF]/g, '/')
      .replace(/\\\\u003[aA]/g, ':')
      .replace(/\\\\u0026/g, '&')
      .replace(/&#0*47;|&sol;/gi, '/')
      .replace(/&#0*58;|&colon;/gi, ':')
      .replace(/&amp;/gi, '&');
  }};
  const extractUrn = (value) => {{
    const match = decodeLinkedInCandidate(value).match(/urn:li:(activity|ugcPost|share):\\d+/i);
    return match ? match[0] : '';
  }};
  const canonicalFromUrn = (urn) => urn ? `https://www.linkedin.com/feed/update/${{urn}}/` : '';
  const isValidPostUrl = (url) => {{
    const value = String(url || '').trim();
    if (!value) return false;
    try {{
      const parsed = new URL(value);
      const host = parsed.hostname.toLowerCase();
      if (host !== 'www.linkedin.com' && host !== 'linkedin.com') return false;
      const path = parsed.pathname.replace(/\\/+/g, '/');
      if (/^\\/feed\\/update\\/urn:li:(activity|ugcPost|share):\\d+\\/?$/i.test(path)) return true;
      if (/^\\/posts\\/[^/?#]+\\/?$/i.test(path)) return true;
      return false;
    }} catch (_err) {{
      return false;
    }}
  }};
  const canonicalizePostUrl = (url) => {{
    const absolute = normalizeHref(url);
    if (!absolute) return '';
    const urn = extractUrn(absolute);
    if (urn) return canonicalFromUrn(urn);
    let parsed;
    try {{
      parsed = new URL(absolute);
    }} catch (_err) {{
      return '';
    }}
    const host = parsed.hostname.toLowerCase();
    if (host !== 'www.linkedin.com' && host !== 'linkedin.com') return '';
    let path = parsed.pathname.replace(/\\/+/g, '/');
    if (/^\\/feed\\/update\\/urn:li:(activity|ugcPost|share):\\d+$/i.test(path)) {{
      path = `${{path}}/`;
    }}
    if (/^\\/posts\\/[^/?#]+$/i.test(path)) {{
      path = `${{path}}/`;
    }}
    const canonical = `https://${{host}}${{path}}`;
    return isValidPostUrl(canonical) ? canonical : '';
  }};
  const canonicalizeContextUrl = (url) => {{
    const absolute = normalizeHref(url);
    if (!absolute) return '';
    let parsed;
    try {{
      parsed = new URL(absolute);
    }} catch (_err) {{
      return '';
    }}
    const host = parsed.hostname.toLowerCase();
    if (host !== 'www.linkedin.com' && host !== 'linkedin.com') return '';
    let path = parsed.pathname.replace(/\\/+/g, '/');
    const allowed = [
      /^\\/posts\\/[^/]+\\/?$/i,
      /^\\/company\\/[^/]+\\/posts\\/?$/i,
      /^\\/in\\/[^/]+\\/?$/i,
    ];
    if (!allowed.some((pattern) => pattern.test(path))) return '';
    if (!path.endsWith('/')) path = `${{path}}/`;
    return `https://${{host}}${{path}}`;
  }};
  const extractPostUrl = (node) => {{
    const anchors = Array.from(node.querySelectorAll('a[href]'));
    for (const anchor of anchors) {{
      const href = anchor.getAttribute('href') || anchor.href || '';
      const normalized = canonicalizePostUrl(href);
      if (normalized) return normalized;
    }}
    const attrs = ['data-urn', 'data-id', 'id', 'data-activity-urn'];
    for (const attr of attrs) {{
      const value = String(node.getAttribute(attr) || '');
      const urn = extractUrn(value);
      if (urn) return canonicalFromUrn(urn);
    }}
    for (const attr of attrs) {{
      for (const child of Array.from(node.querySelectorAll(`[${{attr}}]`))) {{
        const value = String(child.getAttribute(attr) || '');
        const urn = extractUrn(value);
        if (urn) return canonicalFromUrn(urn);
      }}
    }}
    return '';
  }};
  const extractContextFallbackUrl = (node) => {{
    const candidates = [];
    for (const anchor of Array.from(node.querySelectorAll('a[href]'))) {{
      const fallback = canonicalizeContextUrl(anchor.getAttribute('href') || anchor.href || '');
      if (fallback) candidates.push(fallback);
    }}
    for (const category of ['/posts/', '/company/', '/in/']) {{
      for (const candidate of candidates) {{
        if (candidate.includes(category)) return candidate;
      }}
    }}
    return '';
  }};
  const rows = [];
  const seen = new Set();
  const cardSelectors = {json.dumps(list(self._POST_CARD_SELECTORS))};
  const cards = Array.from(document.querySelectorAll(cardSelectors.join(', ')));
  for (const card of cards) {{
    const author = firstText(card, [
      '.entity-result__title-text a span[aria-hidden="true"]',
      '.entity-result__title-text',
      '.reusable-search__result-container [data-anonymize="person-name"]',
      '.update-components-actor__name',
      '.feed-shared-actor__name',
      '.update-components-actor__title span[aria-hidden="true"]',
      '.update-components-actor__title',
      '.feed-shared-actor__title span[aria-hidden="true"]',
      'a.update-components-actor__meta-link span[aria-hidden="true"]',
    ]);
    const text = firstText(card, [
      '.entity-result__summary',
      '.entity-result__primary-subtitle',
      '.reusable-search__result-container .break-words',
      '.update-components-text',
      '.update-components-update-v2__commentary',
      '.feed-shared-update-v2__commentary',
      '.feed-shared-text',
      '[data-test-id="main-feed-activity-card__commentary"]',
      '.break-words span[dir="ltr"]',
    ]);
    const published = firstText(card, [
      'time',
      '.entity-result__secondary-subtitle',
      '.reusable-search__result-container time',
      '.update-components-actor__sub-description',
    ]);
    const publishedIso = extractPublishedIso(card);
    const commentsCount = extractInt(firstText(card, [
      '.social-details-social-counts__comments',
      '.social-details-social-counts__item.social-details-social-counts__comments',
      '[aria-label*="comments" i]',
    ]));
    const reactionsCount = extractInt(firstText(card, [
      '.social-details-social-counts__reactions-count',
      '.social-details-social-counts__reactions',
      '.social-details-social-counts__social-proof-text',
      '[aria-label*="reactions" i]',
      '[aria-label*="likes" i]',
    ]));
    const url = extractPostUrl(card);
    const fallbackUrl = extractContextFallbackUrl(card);
    const title = normalize(author || text.split('\\n')[0] || 'LinkedIn post').slice(0, 240);
    const selftext = normalize(text || card.innerText || '').slice(0, 4000);
    if (!author && !selftext && !url && !fallbackUrl) continue;
    const dedupeKey = url
      ? `u:${{url}}`
      : (fallbackUrl ? `f:${{fallbackUrl}}|${{title}}|${{published}}` : `t:${{title}}|${{published}}|${{selftext.slice(0, 160)}}|${{rows.length}}`);
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);
    rows.push({{
      url,
      fallback_url: fallbackUrl,
      title,
      selftext,
      author,
      published,
      published_iso: publishedIso,
      comments_count: commentsCount,
      reactions_count: reactionsCount,
    }});
    if (rows.length >= scanBudget) break;
  }}
  if (rows.length < scanBudget) {{
    const boxes = Array.from(document.querySelectorAll('[data-testid="expandable-text-box"]'));
    for (const box of boxes) {{
      const context = box.closest('article, li, div') || box.parentElement || box;
      const selftext = normalize(box.textContent || box.innerText || '');
      if (!selftext) continue;
      const url = extractPostUrl(context);
      const fallbackUrl = extractContextFallbackUrl(context);
      const title = normalize(selftext.split('\\n')[0] || 'LinkedIn post').slice(0, 240);
      if (!title && !selftext && !url && !fallbackUrl) continue;
      rows.push({{
        url,
        fallback_url: fallbackUrl,
        title,
        selftext: selftext.slice(0, 4000),
        author: '',
        published: '',
        published_iso: '',
        comments_count: 0,
      }});
      if (rows.length >= scanBudget) break;
    }}
  }}
  if (rows.length < scanBudget) {{
    const anchors = Array.from(document.querySelectorAll('a[href]'));
    for (const anchor of anchors) {{
      const normalizedUrl = canonicalizePostUrl(anchor.getAttribute('href') || anchor.href || '');
      if (!normalizedUrl) continue;
      if (seen.has(normalizedUrl)) continue;
      seen.add(normalizedUrl);
      const container = anchor.closest('article, li, div') || anchor;
      const rawText = normalize((container && (container.innerText || container.textContent)) || anchor.innerText || '');
      const lines = rawText.split('\\n').map((line) => normalize(line)).filter(Boolean);
      const title = normalize(lines[0] || anchor.innerText || 'LinkedIn post').slice(0, 240);
      rows.push({{
        url: normalizedUrl,
        fallback_url: '',
        title,
        selftext: rawText.slice(0, 4000),
        author: '',
        published: '',
        published_iso: '',
      comments_count: 0,
      }});
      if (rows.length >= scanBudget) break;
    }}
  }}
  const rawHtml = document.documentElement ? document.documentElement.innerHTML : '';
  const html = decodeLinkedInCandidate(rawHtml);
  const urnMatches = Array.from(
    html.matchAll(/urn:li:(activity|ugcPost|share):\\d+/gi)
  ).map((match) => match[0]);
  const uniqueUrns = [];
  const seenUrns = new Set();
  for (const urn of urnMatches) {{
    if (seenUrns.has(urn)) continue;
    seenUrns.add(urn);
    uniqueUrns.push(urn);
  }}
  const usedUrns = new Set();
  for (const row of rows) {{
    const normalized = canonicalizePostUrl(row.url || '');
    row.url = normalized;
    if (normalized) {{
      const urn = extractUrn(normalized);
      if (urn) usedUrns.add(urn);
    }}
  }}
  const remainingUrns = uniqueUrns.filter((urn) => !usedUrns.has(urn));
  let urnIdx = 0;
  for (const row of rows) {{
    if (row.url) continue;
    if (urnIdx >= remainingUrns.length) break;
    row.url = canonicalFromUrn(remainingUrns[urnIdx]);
    urnIdx += 1;
  }}
  for (const row of rows) {{
    if (row.url) continue;
    row.url = canonicalizeContextUrl(row.fallback_url || '');
  }}
  return rows.slice(0, scanBudget);
}}
            """
        )
        rows = raw_rows if isinstance(raw_rows, list) else []
        normalized_rows: list[dict[str, Any]] = []
        for item in rows:
            normalized = self._normalize_phase_row(item)
            if normalized is None:
                continue
            normalized_rows.append(normalized)
            if len(normalized_rows) >= limit:
                break
        return normalized_rows

    @staticmethod
    def _truncate_title(value: str, *, max_chars: int = 90) -> str:
        normalized = " ".join(str(value or "").split()).strip()
        if not normalized:
            return "LinkedIn post"
        if max_chars <= 0:
            return normalized
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip()

    @staticmethod
    def _raw_scan_budget(max_posts: int) -> int:
        base = max(1, int(max_posts or 1))
        return min(500, max(base + 60, base * 6))

    def _load_post_detail_rows(
        self,
        *,
        context: Any,
        post_urls: list[str],
        scroll_rounds: int,
        selector_wait_timeout_seconds: int,
        detail_wait_timeout_seconds: int,
        max_comment_expand_clicks: int,
        stop_requested: Callable[[], bool],
    ) -> dict[str, dict[str, Any]]:
        detail_rows_by_url: dict[str, dict[str, Any]] = {}
        if not post_urls:
            return detail_rows_by_url

        page = context.new_page()
        for post_url in post_urls:
            self._raise_if_stopped(stop_requested)
            try:
                page.goto(post_url, wait_until="domcontentloaded")
                page.wait_for_timeout(1500)
                self._ensure_authenticated(page)
                self._raise_if_authwall_page(
                    page,
                    context_message=(
                        "LinkedIn returned a login/auth wall while opening post detail pages. "
                        "Run LinkedIn auth login again with the same profile dir."
                    ),
                )
                self._wait_for_any_selector(
                    page=page,
                    selectors=list(self._SEARCH_WAIT_SELECTORS),
                    timeout_seconds=max(1, int(selector_wait_timeout_seconds or 1)),
                )
                self._scroll_for_discovery(
                    page=page,
                    rounds=max(1, int(scroll_rounds or 1)),
                    stop_requested=stop_requested,
                    delta_y=2400,
                    pause_ms=600,
                )
                self._wait_for_any_selector(
                    page=page,
                    selectors=list(self._SEARCH_WAIT_SELECTORS),
                    timeout_seconds=max(1, int(detail_wait_timeout_seconds or 1)),
                )
                self._expand_comment_threads(
                    page=page,
                    max_clicks=max(1, int(max_comment_expand_clicks or 1)),
                )
                detail_candidates = self._phase_rows_from_page(page=page, max_posts=3)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "auth wall" in message or "session not authenticated" in message:
                    raise
                logger.info(
                    "LinkedIn detail enrichment failed for %s: %s", post_url, exc
                )
                detail_rows_by_url[post_url] = {}
                continue
            except Exception:
                logger.info("LinkedIn detail enrichment failed for %s.", post_url)
                detail_rows_by_url[post_url] = {}
                continue
            selected_detail: dict[str, Any] = {}
            for candidate in detail_candidates:
                candidate_url = self._canonicalize_post_url(
                    str(candidate.get("url", "")).strip()
                )
                if candidate_url == post_url:
                    selected_detail = candidate
                    break
            if not selected_detail and detail_candidates:
                selected_detail = detail_candidates[0]
            detail_rows_by_url[post_url] = selected_detail
        return detail_rows_by_url

    def _enrich_phase_rows(
        self,
        seed_rows: list[dict[str, Any]],
        detail_rows_by_url: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for seed in seed_rows:
            post_url = self._canonicalize_post_url(str(seed.get("url", "")).strip())
            if not post_url:
                enriched.append(dict(seed))
                continue
            detail = detail_rows_by_url.get(post_url)
            if not isinstance(detail, dict) or not detail:
                enriched.append(dict(seed))
                continue
            merged = self._merge_phase_row(seed=seed, detail=detail)
            merged["url"] = post_url
            enriched.append(merged)
        return enriched

    def _collect_phase_rows_with_scroll(
        self,
        *,
        page: Any,
        rounds: int,
        max_posts: int,
        wait_timeout_seconds: int,
        stop_requested: Callable[[], bool],
    ) -> list[dict[str, Any]]:
        limit = max(1, int(max_posts or 1))
        total_rounds = max(1, int(rounds or 1))
        by_url: dict[str, dict[str, Any]] = {}
        ordered_urls: list[str] = []
        no_new_rounds = 0

        for iteration in range(total_rounds + 1):
            self._raise_if_stopped(stop_requested)
            current_rows = self._phase_rows_from_page(page=page, max_posts=limit)
            new_rows_in_iteration = 0
            for row in current_rows:
                canonical_url = self._canonicalize_post_url(
                    str(row.get("url", "")).strip()
                )
                if not canonical_url:
                    continue
                normalized = dict(row)
                normalized["url"] = canonical_url
                existing = by_url.get(canonical_url)
                if existing is None:
                    by_url[canonical_url] = normalized
                    ordered_urls.append(canonical_url)
                    new_rows_in_iteration += 1
                    continue
                merged = self._merge_phase_row(seed=existing, detail=normalized)
                merged["url"] = canonical_url
                by_url[canonical_url] = merged

            if new_rows_in_iteration <= 0:
                no_new_rounds += 1
            else:
                no_new_rounds = 0
            if len(ordered_urls) >= limit:
                break
            if iteration >= total_rounds:
                break
            if no_new_rounds >= self._MAX_NO_NEW_PHASE_ROUNDS:
                break
            self._scroll_for_discovery(
                page=page,
                rounds=1,
                stop_requested=stop_requested,
                delta_y=3200,
                pause_ms=700,
            )
            self._wait_for_any_selector(
                page=page,
                selectors=list(self._SEARCH_WAIT_SELECTORS),
                timeout_seconds=max(1, int(wait_timeout_seconds or 1)),
            )

        return [by_url[url] for url in ordered_urls[:limit]]

    @staticmethod
    def _merge_phase_row(
        *, seed: dict[str, Any], detail: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(seed)
        detail_title = str(detail.get("title", "")).strip()
        if detail_title:
            merged["title"] = detail_title
        detail_text = str(detail.get("selftext", "")).strip()
        seed_text = str(merged.get("selftext", "")).strip()
        if detail_text and len(detail_text) >= len(seed_text):
            merged["selftext"] = detail_text
        detail_published = str(detail.get("published", "")).strip()
        seed_published = str(merged.get("published", "")).strip()
        if detail_published and (
            not seed_published or len(detail_published) >= len(seed_published)
        ):
            merged["published"] = detail_published
        detail_published_iso = str(detail.get("published_iso", "")).strip()
        if detail_published_iso:
            merged["published_iso"] = detail_published_iso
        detail_comments = LinkedinService._positive_count(detail.get("comments_count"))
        seed_comments = LinkedinService._positive_count(merged.get("comments_count"))
        if detail_comments > seed_comments:
            merged["comments_count"] = detail_comments
        detail_reactions = LinkedinService._positive_count(
            detail.get("reactions_count")
        )
        seed_reactions = LinkedinService._positive_count(merged.get("reactions_count"))
        if detail_reactions > seed_reactions:
            merged["reactions_count"] = detail_reactions
        return merged

    @staticmethod
    def _is_authwall_html(html: str) -> bool:
        lowered = str(html or "").lower()
        if "authwall" in lowered:
            return True
        if "form.login__form" in lowered or "input#username" in lowered:
            return True
        if 'href="/login"' in lowered or "href='/login'" in lowered:
            if "sign in" in lowered or "join now" in lowered:
                return True
        return False

    def _raise_if_authwall_page(self, page: Any, *, context_message: str) -> None:
        content = ""
        try:
            content_reader = getattr(page, "content", None)
            if callable(content_reader):
                content = str(content_reader() or "")
        except Exception:
            content = ""
        if not content:
            return
        if self._is_authwall_html(content):
            raise RuntimeError(context_message)

    def _wait_for_any_selector(
        self,
        *,
        page: Any,
        selectors: list[str],
        timeout_seconds: int = 25,
    ) -> bool:
        if not selectors:
            return False
        locator_fn = getattr(page, "locator", None)
        if not callable(locator_fn):
            return False
        deadline = time.monotonic() + float(max(1, int(timeout_seconds or 1)))
        while time.monotonic() < deadline:
            for selector in selectors:
                try:
                    if page.locator(selector).count() > 0:
                        return True
                except Exception:
                    continue
            try:
                mouse = getattr(page, "mouse", None)
                wheel = getattr(mouse, "wheel", None) if mouse is not None else None
                if callable(wheel):
                    wheel(0, 2000)
                else:
                    self._scroll_active_container(page=page, delta_y=2000)
            except Exception:
                try:
                    self._scroll_active_container(page=page, delta_y=2000)
                except Exception:
                    pass
            try:
                page.wait_for_timeout(500)
            except Exception:
                break
        return False

    @staticmethod
    def _expand_comment_threads(*, page: Any, max_clicks: int) -> None:
        locator_fn = getattr(page, "locator", None)
        if not callable(locator_fn):
            return
        selectors = [
            "button[aria-label*='comment' i]",
            "button.comments-comments-list__load-more-comments-button",
            "button.comments-comment-social-bar__reply-action-button",
        ]
        clicked = 0
        budget = max(0, int(max_clicks or 0))
        if budget <= 0:
            return
        for selector in selectors:
            try:
                locator = page.locator(selector)
                total = int(locator.count() or 0)
            except Exception:
                continue
            for idx in range(total):
                if clicked >= budget:
                    return
                try:
                    locator.nth(idx).click(timeout=1200)
                    page.wait_for_timeout(250)
                    clicked += 1
                except Exception:
                    continue

    def _scroll_for_discovery(
        self,
        *,
        page: Any,
        rounds: int,
        stop_requested: Callable[[], bool],
        delta_y: int,
        pause_ms: int,
    ) -> None:
        for _ in range(max(1, int(rounds or 1))):
            self._raise_if_stopped(stop_requested)
            used_mouse_wheel = False
            wheel_moved = True
            try:
                mouse = getattr(page, "mouse", None)
                wheel = getattr(mouse, "wheel", None) if mouse is not None else None
                if callable(wheel):
                    before_scroll = self._read_scroll_position(page)
                    wheel(0, int(delta_y))
                    used_mouse_wheel = True
                    page.wait_for_timeout(120)
                    after_scroll = self._read_scroll_position(page)
                    if (
                        before_scroll is not None
                        and after_scroll is not None
                        and after_scroll <= (before_scroll + 1.0)
                    ):
                        wheel_moved = False
            except Exception:
                used_mouse_wheel = False
                wheel_moved = False
            if not used_mouse_wheel or not wheel_moved:
                self._scroll_active_container(page=page, delta_y=delta_y)
            page.wait_for_timeout(int(max(100, pause_ms)))

    @staticmethod
    def _scroll_active_container(*, page: Any, delta_y: int) -> None:
        page.evaluate(
            """
(delta) => {
  const amount = Math.max(1200, Number(delta || 0) || window.innerHeight || 1200);
  const isScrollable = (node) => {
    if (!node) return false;
    const style = window.getComputedStyle(node);
    const overflowY = String(style.overflowY || '').toLowerCase();
    if (overflowY !== 'auto' && overflowY !== 'scroll') return false;
    const client = Number(node.clientHeight || 0);
    const scrollHeight = Number(node.scrollHeight || 0);
    return client > 0 && scrollHeight > client + 20;
  };
  const visibleArea = (node) => {
    const rect = node.getBoundingClientRect();
    const width = Math.max(0, Math.min(rect.right, window.innerWidth) - Math.max(rect.left, 0));
    const height = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0));
    return width * height;
  };
  const candidates = [];
  const workspace = document.querySelector('main#workspace');
  if (workspace && isScrollable(workspace)) candidates.push(workspace);
  for (const node of Array.from(document.querySelectorAll('main, section, div'))) {
    if (!isScrollable(node)) continue;
    candidates.push(node);
  }
  candidates.sort((a, b) => {
    const aSlack = (Number(a.scrollHeight || 0) - Number(a.clientHeight || 0));
    const bSlack = (Number(b.scrollHeight || 0) - Number(b.clientHeight || 0));
    if (aSlack !== bSlack) return bSlack - aSlack;
    return visibleArea(b) - visibleArea(a);
  });

  const seen = new Set();
  for (const node of candidates) {
    if (!node || seen.has(node)) continue;
    seen.add(node);
    const before = Number(node.scrollTop || 0);
    try {
      node.scrollBy(0, amount);
    } catch (_err) {
      try {
        node.scrollTop = before + amount;
      } catch (_innerErr) {
        continue;
      }
    }
    const after = Number(node.scrollTop || 0);
    if (after > before + 1) return true;
  }
  window.scrollBy(0, amount);
  return false;
}
            """,
            int(delta_y),
        )

    @staticmethod
    def _read_scroll_position(page: Any) -> float | None:
        try:
            raw = page.evaluate(
                """
() => {
  const toNumber = (value) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  };
  const values = [];
  const root = document.scrollingElement || document.documentElement || document.body;
  if (root) {
    const rootValue = toNumber(root.scrollTop || window.scrollY || 0);
    if (rootValue !== null) values.push(rootValue);
  }
  const workspace = document.querySelector('main#workspace');
  if (workspace) {
    const workspaceValue = toNumber(workspace.scrollTop || 0);
    if (workspaceValue !== null) values.push(workspaceValue);
  }
  for (const node of Array.from(document.querySelectorAll('main, section, div'))) {
    const client = Number(node.clientHeight || 0);
    const scrollHeight = Number(node.scrollHeight || 0);
    if (!(client > 0 && scrollHeight > client + 20)) continue;
    const style = window.getComputedStyle(node);
    const overflowY = String(style.overflowY || '').toLowerCase();
    if (overflowY !== 'auto' && overflowY !== 'scroll') continue;
    const value = toNumber(node.scrollTop || 0);
    if (value !== null) values.push(value);
  }
  if (!values.length) return null;
  return Math.max(...values);
}
                """
            )
        except Exception:
            return None
        try:
            value = float(raw)
        except Exception:
            return None
        if value < 0:
            return 0.0
        return value

    @staticmethod
    def _phase_scan_limit(max_posts: int) -> int:
        base = max(1, int(max_posts or 1))
        return min(120, max(20, base * 4))

    @staticmethod
    def _query_from_search_plan(
        *, search_plan: dict[str, Any] | None, fallback: str
    ) -> str:
        if not isinstance(search_plan, dict):
            return fallback
        query = str(search_plan.get("query", "")).strip()
        if query:
            return query
        queries = search_plan.get("queries")
        if isinstance(queries, list):
            for item in queries:
                if not isinstance(item, str):
                    continue
                value = item.strip()
                if value:
                    return value
        return fallback

    def _query_passes_from_search_plan(
        self,
        *,
        search_plan: dict[str, Any] | None,
        fallback: str,
    ) -> list[str]:
        base_queries: list[str] = []
        seen_base_queries: set[str] = set()

        def _add_base(candidate: object) -> None:
            normalized = " ".join(str(candidate or "").split()).strip()
            if not normalized:
                return
            if normalized in seen_base_queries:
                return
            seen_base_queries.add(normalized)
            base_queries.append(normalized)

        if isinstance(search_plan, dict):
            _add_base(search_plan.get("query"))
            queries = search_plan.get("queries")
            if isinstance(queries, list):
                for item in queries:
                    _add_base(item)
        if not base_queries:
            _add_base(fallback)

        passes: list[str] = []
        seen_passes: set[str] = set()
        for base_query in base_queries:
            normalized = " ".join(str(base_query or "").split()).strip()
            if not normalized or normalized in seen_passes:
                continue
            seen_passes.add(normalized)
            passes.append(normalized)
        return passes

    @staticmethod
    def _query_variants(query: str) -> list[str]:
        base = " ".join(str(query or "").split()).strip()
        if not base:
            return []
        variants: list[str] = []

        def _add(candidate: str) -> None:
            normalized = " ".join(str(candidate or "").split()).strip()
            if not normalized:
                return
            if normalized in variants:
                return
            variants.append(normalized)

        _add(base)
        dequoted = re.sub(r'"([^"]+)"', r"\1", base)
        relaxed = re.sub(r"[()]", " ", dequoted)
        relaxed = re.sub(r"\b(?:and|or|not)\b", " ", relaxed, flags=re.IGNORECASE)
        _add(relaxed)

        token_source = relaxed if relaxed else base
        tokens: list[str] = []
        seen_tokens: set[str] = set()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", token_source):
            normalized = token.lower()
            if normalized in {"and", "or", "not"}:
                continue
            if len(normalized) <= 2:
                continue
            if normalized in seen_tokens:
                continue
            seen_tokens.add(normalized)
            tokens.append(token)
        if tokens:
            window_size = 8
            _add(" ".join(tokens[:window_size]))
            if len(tokens) > window_size:
                _add(" ".join(tokens[-window_size:]))
            if len(tokens) > window_size * 2:
                start = max(0, (len(tokens) - window_size) // 2)
                _add(" ".join(tokens[start : start + window_size]))

        quoted_phrases = [
            part.strip() for part in re.findall(r'"([^"]+)"', base) if part.strip()
        ]
        if quoted_phrases:
            _add(" OR ".join(f'"{part}"' for part in quoted_phrases[:6]))

        return variants[:6]

    @staticmethod
    def _extract_search_query_from_llm(raw: str) -> str:
        queries = LinkedinService._extract_search_queries_from_llm(raw)
        return queries[0] if queries else ""

    @staticmethod
    def _extract_search_queries_from_llm(raw: str) -> list[str]:
        source = str(raw or "").strip()
        if not source:
            return []

        blocks = re.findall(
            r"```json\s*(.*?)```", source, flags=re.IGNORECASE | re.DOTALL
        )
        candidate = blocks[0].strip() if blocks else source

        payload: Any = None
        try:
            payload = json.loads(candidate)
        except Exception:
            payload = None

        normalized: list[str] = []
        seen: set[str] = set()

        def _add(candidate: object) -> None:
            if not isinstance(candidate, str):
                return
            value = " ".join(candidate.split()).strip()
            if not value or value in seen:
                return
            seen.add(value)
            normalized.append(value)

        if isinstance(payload, dict):
            queries = payload.get("queries")
            if isinstance(queries, list):
                for item in queries:
                    _add(item)
            _add(payload.get("query"))
            if normalized:
                return normalized[:5]

        if isinstance(payload, list):
            for item in payload:
                _add(item)
            if normalized:
                return normalized[:5]

        direct = source.strip().strip('"').strip("'")
        if not direct:
            return []
        _add(direct)
        return normalized[:5]

    @staticmethod
    def _normalize_linkedin_search_queries(
        candidates: list[str], *, topic: str, min_count: int, max_count: int
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        def _add(candidate: object) -> None:
            if not isinstance(candidate, str):
                return
            value = " ".join(candidate.split()).strip()
            if not LinkedinService._is_valid_linkedin_search_query(value):
                return
            key = value.lower()
            if key in seen:
                return
            seen.add(key)
            normalized.append(value)

        for candidate in candidates:
            _add(candidate)
            if len(normalized) >= max_count:
                return normalized

        for candidate in LinkedinService._fallback_linkedin_queries(topic):
            _add(candidate)
            if len(normalized) >= max_count:
                break

        return normalized[: max(min_count, min(len(normalized), max_count))]

    @staticmethod
    def _is_valid_linkedin_search_query(value: str) -> bool:
        query = " ".join(str(value or "").split()).strip()
        if not query or len(query) > 220:
            return False

        lower = query.lower()
        invalid_fragments = (
            "://",
            "linkedin.com",
            "/search/",
            "/company/",
            "/in/",
            "urn:li:",
            "reddit.com",
            "subreddit:",
            "r/",
        )
        if any(fragment in lower for fragment in invalid_fragments):
            return False
        if re.search(r"\b(?:site|inurl|url|author|flair|self|subreddit):", lower):
            return False

        terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query)
        content_terms = [
            term for term in terms if term.lower() not in {"and", "or", "not"}
        ]
        return bool(content_terms)

    @staticmethod
    def _fallback_linkedin_queries(topic: str) -> list[str]:
        normalized_topic = " ".join(str(topic or "").split()).strip()
        if not normalized_topic:
            return []

        plain_topic = normalized_topic.replace('"', "").strip()
        tokens: list[str] = []
        seen_tokens: set[str] = set()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", plain_topic):
            key = token.lower()
            if key in {"and", "or", "not"} or key in seen_tokens:
                continue
            seen_tokens.add(key)
            tokens.append(token)

        compact_topic = " ".join(tokens) if tokens else plain_topic
        candidates = [plain_topic]
        if len(tokens) > 1:
            candidates.extend(
                [
                    f'"{plain_topic}"',
                    " AND ".join(tokens[:5]),
                    " ".join(tokens[: min(len(tokens), 4)]),
                    f"{compact_topic} challenges",
                    f"{compact_topic} best practices",
                    f"{compact_topic} workflow",
                    f"{compact_topic} adoption",
                    f"{compact_topic} strategy",
                ]
            )
            if len(tokens) > 4:
                candidates.append(" ".join(tokens[-4:]))
        else:
            candidates.extend(
                [
                    f'"{plain_topic}"',
                    f"{plain_topic} updates",
                    f"{plain_topic} discussion",
                ]
            )
        return candidates

    def _collect_post_links(
        self, rows: list[dict[str, Any]], max_posts: int
    ) -> list[str]:
        links: list[str] = []
        seen: set[str] = set()
        for row in rows:
            canonical_url = self._canonicalize_post_url(str(row.get("url", "")).strip())
            if not canonical_url or canonical_url in seen:
                continue
            seen.add(canonical_url)
            links.append(canonical_url)
            if len(links) >= max(1, int(max_posts or 1)):
                break
        return links

    def _detail_enrich_top_k(self, value: int | None) -> int:
        if value is not None:
            try:
                return max(0, int(value))
            except Exception:
                return self._DEFAULT_DETAIL_ENRICH_TOP_K
        configured = os.getenv("LINKEDIN_DETAIL_ENRICH_TOP_K", "").strip()
        try:
            return (
                max(0, int(configured))
                if configured
                else self._DEFAULT_DETAIL_ENRICH_TOP_K
            )
        except Exception:
            return self._DEFAULT_DETAIL_ENRICH_TOP_K

    def _select_detail_enrichment_links(
        self,
        *,
        rows: list[dict[str, Any]],
        max_posts: int,
        detail_enrich_top_k: int | None = None,
    ) -> list[str]:
        configured_cap = self._detail_enrich_top_k(detail_enrich_top_k)
        cap = min(configured_cap, max(1, int(max_posts or 1)))
        if cap <= 0:
            return []

        prioritized: list[str] = []
        seen: set[str] = set()
        for row in rows:
            canonical_url = self._canonicalize_post_url(str(row.get("url", "")).strip())
            if not canonical_url or canonical_url in seen:
                continue
            if not self._needs_detail_enrichment(row):
                continue
            prioritized.append(canonical_url)
            seen.add(canonical_url)
            if len(prioritized) >= cap:
                break

        if len(prioritized) < cap:
            for canonical_url in self._collect_post_links(rows, max_posts=max_posts):
                if canonical_url in seen:
                    continue
                prioritized.append(canonical_url)
                seen.add(canonical_url)
                if len(prioritized) >= cap:
                    break
        return prioritized

    @staticmethod
    def _needs_detail_enrichment(row: dict[str, Any]) -> bool:
        published_iso = str(row.get("published_iso", "")).strip()
        if not published_iso:
            return True
        selftext = str(row.get("selftext", "")).strip()
        if len(selftext) < 120:
            return True
        score, comments = LinkedinService._extract_engagement_metrics(selftext)
        return score <= 0 and comments <= 0

    @staticmethod
    def _positive_count(value: object) -> int:
        try:
            parsed = int(str(value))
        except Exception:
            return 0
        return max(0, parsed)

    def _normalize_phase_row(self, row: Any) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        canonical_url = self._canonicalize_post_url(str(row.get("url", "")).strip())
        if not canonical_url:
            fallback = str(row.get("fallback_url", "")).strip()
            canonical_url = self._canonicalize_post_url(fallback)
        if not canonical_url:
            return None
        title = str(row.get("title", "")).strip() or "LinkedIn post"
        selftext = str(row.get("selftext", "")).strip()
        published = str(row.get("published", "")).strip()
        published_iso = str(row.get("published_iso", "")).strip()
        return {
            "url": canonical_url,
            "title": title,
            "selftext": selftext,
            "published": published,
            "published_iso": published_iso,
            "comments_count": self._positive_count(row.get("comments_count")),
            "reactions_count": self._positive_count(row.get("reactions_count")),
        }

    def post_comment(self, thread_id: str, comment_text: str) -> bool:
        normalized_comment = comment_text.strip()
        if not normalized_comment:
            return False

        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            return False

        post_url = self._thread_url(thread_id)

        try:
            with self._profile_runtime_lock():
                with sync_playwright() as playwright:
                    context = self._launch_context(playwright)
                    if not self._context_appears_authenticated(context):
                        logger.warning(
                            "LinkedIn send failed: session appears unauthenticated for profile %s. "
                            "Run LinkedIn auth login again.",
                            self.user_data_dir,
                        )
                        context.close()
                        return False
                    page = context.new_page()

                    page.goto(post_url, wait_until="domcontentloaded")
                    page.wait_for_timeout(1200)
                    self._ensure_authenticated(page)
                    self._raise_if_authwall_page(
                        page,
                        context_message=(
                            "LinkedIn send failed: auth wall detected while opening post. "
                            "Run LinkedIn auth login again with the same profile dir."
                        ),
                    )

                    comment_button = self._find_first_locator(
                        page,
                        [
                            "button[aria-label*='Comment on' i]",
                            "button[aria-label*='Comment' i]",
                            "button[aria-label*='Reply' i]",
                        ],
                    )
                    if comment_button is not None:
                        try:
                            comment_button.click(timeout=3000)
                            page.wait_for_timeout(500)
                        except Exception:
                            pass

                    editor = self._find_first_locator(
                        page, self._comment_input_selectors()
                    )
                    if editor is None:
                        logger.warning(
                            "LinkedIn send failed: comment editor not found for %s",
                            post_url,
                        )
                        context.close()
                        return False

                    editor.click(timeout=3000)
                    try:
                        editor.fill(normalized_comment, timeout=3000)
                    except Exception:
                        page.keyboard.press("Control+A")
                        page.keyboard.press("Backspace")
                        page.keyboard.type(normalized_comment, delay=12)

                    submit_button, _disabled_seen = self._wait_for_submit_button(
                        page,
                        input_box=editor,
                        timeout_ms=7000,
                    )
                    if submit_button is None:
                        logger.warning(
                            "LinkedIn send failed: submit button not found for %s",
                            post_url,
                        )
                        context.close()
                        return False

                    posted = self._submit_comment_and_verify(
                        page=page,
                        input_box=editor,
                        submit_button=submit_button,
                        expected_comment_text=normalized_comment,
                    )
                    if not posted:
                        logger.warning(
                            "LinkedIn send failed: submit action not verified for %s",
                            post_url,
                        )
                    context.close()
                    return posted
        except Exception as exc:
            logger.warning("LinkedIn send failed for %s: %s", post_url, exc)
            return False

    @staticmethod
    def _comment_input_selectors() -> list[str]:
        return [
            "form.comments-comment-box__form div[role='textbox'][contenteditable='true']",
            "form.comments-comment-box__form div[data-test-ql-editor-contenteditable='true']",
            "form.comments-comment-box__form textarea[aria-label*='comment' i]",
            "div.comments-comment-box form div[role='textbox'][contenteditable='true']",
            "div.comments-comment-box form div[data-test-ql-editor-contenteditable='true']",
            "div[role='textbox'][contenteditable='true'][aria-label*='comment' i]",
            "div.comments-comment-box-comment__text-editor div[role='textbox'][contenteditable='true']",
            "div[role='textbox'][contenteditable='true']",
        ]

    @staticmethod
    def _comment_submit_selectors() -> list[str]:
        return [
            "form.comments-comment-box__form button.comments-comment-box__submit-button--cr",
            "form.comments-comment-box__form button[class*='comments-comment-box__submit-button']",
            "form.comments-comment-box__form button.artdeco-button--primary",
            "form.comments-comment-box__form button[type='submit']",
            "div.comments-comment-box form button[type='submit']",
            "div.comments-comment-box button[type='submit']",
            "form.comments-comment-box__form button:has-text('Comment')",
            "form.comments-comment-box__form button:has-text('Reply')",
            "button[aria-label*='Post comment' i]",
            "button[aria-label*='Post']",
            "button:has-text('Post')",
        ]

    @staticmethod
    def _posted_comment_selectors() -> list[str]:
        return [
            "article.comments-comment-item",
            "li.comments-comment-item",
            "div.comments-comment-item",
            "div.comments-comments-list__comment-item",
            "div.comments-comment-item-content-body",
            "span.break-words",
        ]

    @staticmethod
    def _page_has_posted_comment_text(page: Any, comment_text: str) -> bool:
        target = " ".join(str(comment_text or "").split()).strip().lower()
        if not target:
            return False
        locator_fn = getattr(page, "locator", None)
        if not callable(locator_fn):
            return False
        for selector in LinkedinService._posted_comment_selectors():
            try:
                locator = page.locator(selector)
                count = int(locator.count() or 0)
            except Exception:
                continue
            for idx in range(min(count, 30)):
                try:
                    text = locator.nth(idx).inner_text(timeout=400)
                except Exception:
                    continue
                normalized = " ".join(str(text or "").split()).strip().lower()
                if target in normalized:
                    return True
        return False

    @staticmethod
    def _find_first_locator(
        page_or_scope: Any, selectors: list[str], require_visible: bool = True
    ) -> Any | None:
        for selector in selectors:
            locator = page_or_scope.locator(selector)
            try:
                count = locator.count()
            except Exception:
                continue
            if count <= 0:
                continue
            for idx in range(count):
                candidate = locator.nth(idx)
                if not require_visible:
                    return candidate
                try:
                    if candidate.is_visible():
                        return candidate
                except Exception:
                    continue
        return None

    @staticmethod
    def _read_editor_text(input_box: Any) -> str:
        try:
            text = input_box.inner_text(timeout=1200)
            return " ".join(text.split()).strip()
        except Exception:
            return ""

    @staticmethod
    def _is_disabled_button(button: Any) -> bool:
        try:
            disabled_attr = button.get_attribute("disabled")
            aria_disabled = button.get_attribute("aria-disabled")
            class_name = str(button.get_attribute("class") or "").lower()
            if disabled_attr is not None:
                return True
            if isinstance(aria_disabled, str) and aria_disabled.lower() == "true":
                return True
            if "disabled" in class_name:
                return True
            return False
        except Exception:
            return True

    @staticmethod
    def _editor_form_scope(input_box: Any) -> Any | None:
        if input_box is None:
            return None
        try:
            form_locator = input_box.locator("xpath=ancestor::form[1]")
            if form_locator.count() <= 0:
                return None
            return form_locator.nth(0)
        except Exception:
            return None

    @staticmethod
    def _wait_for_submit_button(
        page: Any, input_box: Any | None = None, timeout_ms: int = 7000
    ) -> tuple[Any | None, bool]:
        deadline = time.monotonic() + (timeout_ms / 1000.0)
        last_disabled_seen = False
        form_scope = LinkedinService._editor_form_scope(input_box)
        while time.monotonic() < deadline:
            candidate = None
            if form_scope is not None:
                candidate = LinkedinService._find_first_locator(
                    form_scope, LinkedinService._comment_submit_selectors()
                )
            if candidate is None:
                candidate = LinkedinService._find_first_locator(
                    page, LinkedinService._comment_submit_selectors()
                )
            if candidate is None:
                page.wait_for_timeout(120)
                continue
            if LinkedinService._is_disabled_button(candidate):
                last_disabled_seen = True
                page.wait_for_timeout(180)
                continue
            return candidate, last_disabled_seen
        return None, last_disabled_seen

    @staticmethod
    def _context_appears_authenticated(context: Any) -> bool:
        try:
            if _has_linkedin_session_cookie(context):
                return True
        except Exception:
            pass
        try:
            return bool(_probe_context_authentication(context))
        except Exception:
            return False

    @staticmethod
    def _request_post_data(request: Any) -> str:
        if request is None:
            return ""
        for attr in ("post_data", "postData", "post_data_buffer", "postDataBuffer"):
            value = getattr(request, attr, None)
            if value is None:
                continue
            try:
                value = value() if callable(value) else value
            except Exception:
                continue
            if value is None:
                continue
            if isinstance(value, (bytes, bytearray)):
                try:
                    text = value.decode("utf-8", errors="ignore")
                except Exception:
                    continue
            else:
                text = str(value)
            normalized = text.strip()
            if normalized:
                return normalized.lower()
        return ""

    @staticmethod
    def _is_comment_submission_ack(response: Any) -> bool:
        request = getattr(response, "request", None)
        method = str(getattr(request, "method", "")).strip().upper()
        if method != "POST":
            return False

        url = str(getattr(response, "url", "")).strip().lower()
        if "/voyager/api/" not in url:
            return False

        url_tokens = ("/comments", "socialactions", "createcomment", "updatecomment")
        if any(token in url for token in url_tokens):
            return True

        payload = LinkedinService._request_post_data(request)
        if not payload:
            return False
        payload_tokens = ("comment", "socialaction", "createcomment", "updatecomment")
        return any(token in payload for token in payload_tokens)

    @staticmethod
    def _submit_comment_and_verify(
        *,
        page: Any,
        input_box: Any,
        submit_button: Any,
        expected_comment_text: str = "",
    ) -> bool:
        response_status: int | None = None
        response_url = ""
        got_response = False
        response_is_comment_ack = False
        clicked = False
        expected_comment = " ".join(str(expected_comment_text or "").split()).strip()
        try:
            with page.expect_response(
                LinkedinService._is_comment_submission_ack,
                timeout=7000,
            ) as response_info:
                submit_button.click(timeout=3000)
                clicked = True
            response_status = int(response_info.value.status)
            response_url = str(getattr(response_info.value, "url", "") or "")
            got_response = True
            response_is_comment_ack = LinkedinService._is_comment_submission_ack(
                response_info.value
            )
        except Exception:
            if not clicked:
                try:
                    submit_button.click(timeout=3000)
                    clicked = True
                except Exception:
                    return False

        if not clicked:
            return False

        if (
            got_response
            and response_status is not None
            and 200 <= response_status < 300
            and response_is_comment_ack
        ):
            return True

        def _poll_verification(timeout_seconds: float) -> tuple[bool, str]:
            verification_deadline = time.monotonic() + max(0.5, float(timeout_seconds))
            last_text = ""
            while True:
                last_text = LinkedinService._read_editor_text(input_box)
                if last_text == "":
                    return True, last_text
                if expected_comment and LinkedinService._page_has_posted_comment_text(
                    page, expected_comment
                ):
                    return True, last_text
                if time.monotonic() >= verification_deadline:
                    return False, last_text
                page.wait_for_timeout(220)

        verified, last_editor_text = _poll_verification(4.5)
        if verified:
            return True

        # One retry click helps when LinkedIn ignores the first click while focus settles.
        try:
            submit_button.click(timeout=3000)
            page.wait_for_timeout(500)
        except Exception:
            pass

        verified, last_editor_text = _poll_verification(3.0)
        if verified:
            return True

        if got_response or response_url:
            logger.info(
                "LinkedIn submit remained unverified (status=%s, ack=%s, url=%s, editor_len=%s).",
                response_status,
                response_is_comment_ack,
                response_url,
                len(last_editor_text),
            )
        return False

    def _launch_context(self, playwright):
        profile_dir = Path(self.user_data_dir).expanduser()
        profile_dir.mkdir(parents=True, exist_ok=True)
        return playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=self.headless,
        )

    @staticmethod
    def _ensure_authenticated(page) -> None:
        url = str(getattr(page, "url", "") or "").lower()
        if _url_requires_login(url):
            raise RuntimeError(
                "LinkedIn session not authenticated. Run LinkedIn auth login first."
            )

    @staticmethod
    def _normalize_time_filter(raw: str) -> str:
        value = str(raw or "").strip().lower()
        if value not in {"hour", "day", "week", "month", "year", "all"}:
            return "week"
        return value

    @staticmethod
    def _normalize_sort_by(raw: str) -> str:
        value = str(raw or "").strip().lower()
        alias_map = {
            "recent": "recency",
            "latest": "recency",
            "new": "recency",
            "top": "engagement",
            "popular": "engagement",
        }
        normalized = alias_map.get(value, value)
        if normalized not in {"relevance", "recency", "engagement"}:
            return "relevance"
        return normalized

    @staticmethod
    def _linkedin_date_posted_facet(time_filter: str) -> str | None:
        mapping = {
            "hour": "past-24h",
            "day": "past-24h",
            "week": "past-week",
            "month": "past-month",
            "year": "past-year",
        }
        return mapping.get(time_filter)

    @staticmethod
    def _linkedin_search_url(*, query: str, time_filter: str, sort_by: str) -> str:
        normalized_sort_by = LinkedinService._normalize_sort_by(sort_by)
        normalized_time_filter = LinkedinService._normalize_time_filter(time_filter)
        if normalized_sort_by == "recency":
            origin = "GLOBAL_SEARCH_HEADER"
            sort_token = "date_posted"
        else:
            origin = "FACETED_SEARCH"
            sort_token = "relevance"

        params: dict[str, str] = {
            "keywords": str(query or ""),
            "origin": origin,
            "sortBy": json.dumps([sort_token], separators=(",", ":")),
        }
        date_posted = LinkedinService._linkedin_date_posted_facet(
            normalized_time_filter
        )
        if date_posted is not None:
            params["datePosted"] = json.dumps([date_posted], separators=(",", ":"))
        return "https://www.linkedin.com/search/results/content/?" + urlencode(
            params, quote_via=quote
        )

    @staticmethod
    def _decode_linkedin_url_candidate(raw: object) -> str:
        value = str(raw or "").strip().rstrip("\\")
        if not value:
            return ""
        for _ in range(2):
            decoded = unquote(value)
            if decoded == value:
                break
            value = decoded
        replacements = {
            r"\u002F": "/",
            r"\u002f": "/",
            r"\u003A": ":",
            r"\u003a": ":",
            r"\u0026": "&",
            "&#47;": "/",
            "&#58;": ":",
            "&sol;": "/",
            "&colon;": ":",
            "&amp;": "&",
        }
        for source, target in replacements.items():
            value = value.replace(source, target)
        return value

    @staticmethod
    def _extract_post_urn(raw: object) -> str:
        value = LinkedinService._decode_linkedin_url_candidate(raw)
        match = re.search(
            r"urn:li:(activity|ugcPost|share):\d+", value, flags=re.IGNORECASE
        )
        return "" if match is None else match.group(0)

    @staticmethod
    def _canonicalize_post_url(url: str) -> str:
        value = LinkedinService._decode_linkedin_url_candidate(url)
        if not value:
            return ""
        urn = LinkedinService._extract_post_urn(value)
        if urn:
            return f"https://www.linkedin.com/feed/update/{urn}/"
        if value.startswith("urn:li:"):
            value = f"https://www.linkedin.com/feed/update/{value}/"
        elif value.startswith("//"):
            value = f"https:{value}"
        elif value.startswith("/"):
            value = f"https://www.linkedin.com{value}"

        try:
            parts = urlsplit(value)
        except Exception:
            return ""
        host = parts.netloc.lower()
        if host not in {"www.linkedin.com", "linkedin.com"}:
            return ""

        path = re.sub(r"/{2,}", "/", parts.path)
        if re.match(
            r"^/feed/update/urn:li:(activity|ugcPost|share):\d+$",
            path,
            flags=re.IGNORECASE,
        ):
            path = f"{path}/"
        if re.match(r"^/posts/[^/?#]+$", path, flags=re.IGNORECASE):
            if not path.endswith("/"):
                path = f"{path}/"

        canonical = urlunsplit(("https", host, path, "", ""))
        if not LinkedinService._is_valid_linkedin_post_url(canonical):
            return ""
        return canonical

    @staticmethod
    def _is_valid_linkedin_post_url(url: str) -> bool:
        value = str(url or "").strip()
        if not value:
            return False
        try:
            parts = urlsplit(value)
        except Exception:
            return False
        host = parts.netloc.lower()
        if host not in {"www.linkedin.com", "linkedin.com"}:
            return False
        path = re.sub(r"/{2,}", "/", parts.path)
        if re.match(
            r"^/feed/update/urn:li:(activity|ugcPost|share):\d+/?$",
            path,
            flags=re.IGNORECASE,
        ):
            return True
        if re.match(r"^/posts/[^/?#]+/?$", path, flags=re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _parse_count_token(token: str) -> int:
        normalized = str(token or "").strip().lower().replace(",", "")
        if not normalized:
            return 0
        multiplier = 1.0
        if normalized.endswith("k"):
            normalized = normalized[:-1]
            multiplier = 1000.0
        elif normalized.endswith("m"):
            normalized = normalized[:-1]
            multiplier = 1000000.0
        try:
            value = float(normalized)
        except Exception:
            return 0
        if value < 0:
            return 0
        return int(value * multiplier)

    @staticmethod
    def _extract_engagement_metrics(text: str) -> tuple[int, int]:
        normalized = " ".join(str(text or "").split()).lower()
        score = 0
        num_comments = 0

        reaction_match = re.search(
            r"\b(\d[\d.,]*[km]?)\s+(?:reactions?|likes?)\b", normalized
        )
        if reaction_match:
            score = LinkedinService._parse_count_token(reaction_match.group(1))

        comments_match = re.search(r"\b(\d[\d.,]*[km]?)\s+comments?\b", normalized)
        if comments_match:
            num_comments = LinkedinService._parse_count_token(comments_match.group(1))

        return score, num_comments

    @staticmethod
    def _extract_created_utc(*, selftext: str, now_utc: float) -> float | None:
        normalized = " ".join(str(selftext or "").split()).lower()
        if not normalized:
            return None
        if "just now" in normalized or re.search(r"\bnow\b", normalized):
            return float(now_utc)

        match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks|mo|mos|month|months|y|yr|yrs|year|years)\b",
            normalized,
        )
        if not match:
            return None
        quantity = float(match.group(1))
        unit = match.group(2)
        multiplier = {
            "s": 1,
            "sec": 1,
            "secs": 1,
            "second": 1,
            "seconds": 1,
            "m": 60,
            "min": 60,
            "mins": 60,
            "minute": 60,
            "minutes": 60,
            "h": 3600,
            "hr": 3600,
            "hrs": 3600,
            "hour": 3600,
            "hours": 3600,
            "d": 86400,
            "day": 86400,
            "days": 86400,
            "w": 604800,
            "wk": 604800,
            "wks": 604800,
            "week": 604800,
            "weeks": 604800,
            "mo": 2592000,
            "mos": 2592000,
            "month": 2592000,
            "months": 2592000,
            "y": 31536000,
            "yr": 31536000,
            "yrs": 31536000,
            "year": 31536000,
            "years": 31536000,
        }.get(unit)
        if multiplier is None:
            return None
        candidate = float(now_utc) - (quantity * float(multiplier))
        return max(0.0, candidate)

    @staticmethod
    def _parse_published_iso(raw: object) -> tuple[float | None, str | None]:
        value = str(raw or "").strip()
        if not value:
            return None, None
        if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            try:
                parsed = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
            except Exception:
                return None, None
            return float(parsed.timestamp()), value
        normalized = value
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except Exception:
            return None, None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed_utc = parsed.astimezone(timezone.utc)
        return float(parsed_utc.timestamp()), parsed_utc.isoformat().replace(
            "+00:00", "Z"
        )

    @staticmethod
    def _is_imprecise_relative_published_label(value: str) -> bool:
        normalized = " ".join(str(value or "").split()).lower()
        if not normalized:
            return False
        if normalized in {"today", "yesterday"}:
            return True
        return bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*(d|day|days|w|wk|wks|week|weeks|mo|mos|month|months|y|yr|yrs|year|years)\b",
                normalized,
            )
        )

    @staticmethod
    def _resolve_created_timestamp(
        *,
        row: dict[str, Any],
        selftext: str,
        now_utc: float,
    ) -> tuple[float | None, str | None]:
        created_utc, published_at = LinkedinService._parse_published_iso(
            row.get("published_iso")
        )
        if created_utc is not None:
            return created_utc, published_at
        published_text = str(row.get("published", "")).strip()
        if published_text:
            created_utc = LinkedinService._extract_created_utc(
                selftext=published_text, now_utc=now_utc
            )
            if created_utc is not None:
                if LinkedinService._is_imprecise_relative_published_label(
                    published_text
                ):
                    date_only = (
                        datetime.fromtimestamp(created_utc, tz=timezone.utc)
                        .date()
                        .isoformat()
                    )
                    return created_utc, date_only
                return created_utc, None
        _ = selftext
        return None, None

    @staticmethod
    def _include_unknown_timestamps_for_bounded_filters() -> bool:
        raw = os.getenv(
            "LINKEDIN_INCLUDE_UNKNOWN_TIMESTAMPS_FOR_BOUNDED_FILTERS", "true"
        )
        value = str(raw or "").strip().lower()
        return value not in {"0", "false", "no", "off"}

    @staticmethod
    def _matches_time_filter(
        *, created_utc: float | None, time_filter: str, now_utc: float
    ) -> bool:
        if time_filter == "all":
            return True
        if created_utc is None:
            return LinkedinService._include_unknown_timestamps_for_bounded_filters()

        max_age = {
            "hour": 3600.0,
            "day": 86400.0,
            "week": 604800.0,
            "month": 2592000.0,
            "year": 31536000.0,
        }.get(time_filter, 604800.0)
        age = max(0.0, float(now_utc) - float(created_utc))
        return age <= max_age

    @staticmethod
    def _sort_threads(
        rows: list[dict[str, Any]], *, sort_by: str
    ) -> list[dict[str, Any]]:
        if sort_by == "recency":
            return sorted(
                rows,
                key=lambda row: float(row.get("created_utc") or -1.0),
                reverse=True,
            )
        if sort_by == "engagement":
            return sorted(
                rows,
                key=lambda row: (
                    int(row.get("score", 0) or 0)
                    + int(row.get("num_comments", 0) or 0),
                    int(row.get("score", 0) or 0),
                    int(row.get("num_comments", 0) or 0),
                    float(row.get("created_utc") or -1.0),
                ),
                reverse=True,
            )
        return list(rows)

    @staticmethod
    def _thread_id_from_url(url: str, *, fallback: str) -> str:
        value = url.strip()
        if not value:
            return fallback

        urn_match = re.search(r"urn:li:[^/?&#]+", value)
        if urn_match:
            return urn_match.group(0)

        try:
            parts = urlsplit(value)
        except Exception:
            parts = None

        if parts is not None and parts.scheme in {"http", "https"}:
            host = parts.netloc.lower()
            path = re.sub(r"/{2,}", "/", parts.path)
            if host in {"linkedin.com", "www.linkedin.com"}:
                if path.startswith("/posts/") and not path.endswith("/"):
                    path = f"{path}/"
                if re.match(r"^/feed/update/urn:li:[^/]+$", path):
                    path = f"{path}/"
                return urlunsplit(("https", host, path, "", ""))

        cleaned = re.sub(r"^https?://", "", value)
        cleaned = cleaned.split("?", 1)[0].split("#", 1)[0].strip("/")
        return cleaned or fallback

    @staticmethod
    def _thread_url(thread_id: str) -> str:
        value = thread_id.strip()
        if not value:
            return "https://www.linkedin.com/"
        if value.startswith("urn:li:"):
            return f"https://www.linkedin.com/feed/update/{value.rstrip('/')}/"
        if value.startswith("www.linkedin.com/") or value.startswith("linkedin.com/"):
            return f"https://{value}"
        if value.startswith("feed/update/urn:li:"):
            return f"https://www.linkedin.com/{value.rstrip('/')}/"
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return f"https://www.linkedin.com/{value.lstrip('/')}"

    @staticmethod
    def _username_from_profile_url(url: str) -> str:
        value = (url or "").strip()
        if not value:
            return ""
        match = re.search(r"linkedin\\.com/in/([^/?#]+)", value, flags=re.IGNORECASE)
        if not match:
            return ""
        username = unquote(match.group(1).strip())
        if not username or username.lower() == "me":
            return ""
        return username

    @staticmethod
    def _username_from_voyager_api(context) -> str:
        try:
            cookies = context.cookies(["https://www.linkedin.com"])
        except Exception:
            return ""
        jsession = ""
        for cookie in cookies:
            if str(cookie.get("name", "")).strip() == "JSESSIONID":
                jsession = str(cookie.get("value", "")).strip()
                break
        csrf = jsession.strip('"')
        if not csrf:
            return ""
        try:
            response = context.request.get(
                "https://www.linkedin.com/voyager/api/me",
                timeout=5000,
                fail_on_status_code=False,
                headers={
                    "x-restli-protocol-version": "2.0.0",
                    "csrf-token": csrf,
                },
            )
        except Exception:
            return ""
        if int(getattr(response, "status", 0)) != 200:
            return ""
        try:
            payload = response.json()
        except Exception:
            try:
                payload = json.loads(response.text())
            except Exception:
                return ""
        if not isinstance(payload, dict):
            return ""
        mini_profile = payload.get("miniProfile")
        if not isinstance(mini_profile, dict):
            return ""
        value = str(mini_profile.get("publicIdentifier", "")).strip()
        return value

    @staticmethod
    def _raise_if_stopped(stop_requested: Callable[[], bool]) -> None:
        if stop_requested():
            raise RuntimeError("Run stopped by user request.")

    @staticmethod
    def _cookie_db_candidates(profile_dir: Path) -> list[Path]:
        candidates = [
            profile_dir / "Default" / "Cookies",
            profile_dir / "Default" / "Network" / "Cookies",
            profile_dir / "Network" / "Cookies",
            profile_dir / "Cookies",
        ]
        return [path for path in candidates if path.exists() and path.is_file()]

    @staticmethod
    def _has_linkedin_cookie(cookie_db: Path) -> bool:
        tmp_path: Path | None = None
        conn: sqlite3.Connection | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="reddrop-linkedin-cookie-", suffix=".sqlite", delete=False
            ) as handle:
                tmp_path = Path(handle.name)
            shutil.copy2(cookie_db, tmp_path)
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(1) FROM cookies WHERE name = ? AND host_key LIKE ?",
                ("li_at", "%linkedin.com%"),
            )
            row = cursor.fetchone()
            count = int(row[0]) if row else 0
            return count > 0
        except Exception:
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
