from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus

from ..config.log import setup_logging
from .settings_store import SettingsStore

logger = setup_logging(__name__)


def _never_stop() -> bool:
    return False


def _parse_headless_flag(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    return True


class LinkedinService:
    """Headless LinkedIn discovery/posting integration."""

    def __init__(self) -> None:
        SettingsStore().load()
        self.headless = _parse_headless_flag(os.getenv("LINKEDIN_HEADLESS", "true"))
        configured_dir = (os.getenv("LINKEDIN_USER_DATA_DIR") or "~/.linkedin-playwright").strip()
        self.user_data_dir = str(Path(configured_dir).expanduser())

    def discover_relevant_threads(
        self,
        content: str,
        time_filter: str = "week",
        subreddit_limit: int = 30,
        threads_limit: int = 20,
        job_id: str | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        _ = time_filter
        _ = subreddit_limit
        should_stop = stop_requested or _never_stop
        self._raise_if_stopped(should_stop)

        query = content.strip()
        if not query:
            return []

        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise RuntimeError(
                "LinkedIn headless search requires playwright. "
                "Install it with: pip install playwright && playwright install chromium"
            ) from exc

        try:
            with sync_playwright() as playwright:
                context = self._launch_context(playwright)
                page = context.new_page()

                search_url = (
                    "https://www.linkedin.com/search/results/content/"
                    f"?keywords={quote_plus(query)}&origin=GLOBAL_SEARCH_HEADER"
                )
                page.goto(search_url, wait_until="domcontentloaded")
                page.wait_for_timeout(1500)
                self._ensure_authenticated(page)
                self._raise_if_stopped(should_stop)

                raw_posts = page.evaluate(
                    """
() => {
  const anchors = Array.from(document.querySelectorAll('a[href*="/feed/update/"], a[href*="/posts/"]'));
  const rows = [];
  const seen = new Set();
  for (const anchor of anchors) {
    const href = String(anchor.href || '').trim();
    if (!href || seen.has(href)) continue;
    seen.add(href);
    const container = anchor.closest('article, li, div');
    const rawText = String((container && container.innerText) || anchor.innerText || '').trim();
    const lines = rawText.split('\\n').map((line) => line.trim()).filter(Boolean);
    const title = (lines[0] || anchor.innerText || 'LinkedIn post').slice(0, 240);
    rows.push({ url: href, title, selftext: rawText.slice(0, 4000) });
    if (rows.length >= 100) break;
  }
  return rows;
}
                    """
                )

                rows = raw_posts if isinstance(raw_posts, list) else []
                threads: list[dict[str, Any]] = []
                for index, item in enumerate(rows):
                    self._raise_if_stopped(should_stop)
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get("url", "")).strip()
                    if not url:
                        continue
                    title = str(item.get("title", "LinkedIn post")).strip() or "LinkedIn post"
                    selftext = str(item.get("selftext", "")).strip()
                    thread_id = self._thread_id_from_url(url, fallback=f"linkedin-{index}")
                    threads.append(
                        {
                            "id": thread_id,
                            "title": title,
                            "subreddit": "linkedin",
                            "url": url,
                            "score": 0,
                            "num_comments": 0,
                            "created_utc": None,
                            "selftext": selftext,
                            "semantic_similarity": 0.0,
                            "user_has_commented": False,
                        }
                    )
                    if len(threads) >= max(1, threads_limit):
                        break

                context.close()
                logger.info(
                    "%sLinkedIn discovery completed: query=%r results=%s.",
                    f"{job_id} - " if job_id else "",
                    query,
                    len(threads),
                )
                return threads
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"LinkedIn discovery failed: {exc}") from exc

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
            with sync_playwright() as playwright:
                context = self._launch_context(playwright)
                page = context.new_page()

                page.goto(post_url, wait_until="domcontentloaded")
                page.wait_for_timeout(1200)
                self._ensure_authenticated(page)

                comment_button = page.locator(
                    "button[aria-label*='Comment'], button:has-text('Comment')"
                ).first
                if comment_button.count() == 0:
                    context.close()
                    return False
                comment_button.click()
                page.wait_for_timeout(400)

                editor = page.locator("div[contenteditable='true']").first
                if editor.count() == 0:
                    context.close()
                    return False

                editor.click()
                try:
                    editor.fill(normalized_comment)
                except Exception:
                    page.keyboard.type(normalized_comment)

                submit = page.locator(
                    "button[aria-label*='Post'], button:has-text('Post')"
                ).first
                if submit.count() > 0:
                    submit.click()
                else:
                    page.keyboard.press("Meta+Enter")

                page.wait_for_timeout(1000)
                context.close()
                return True
        except Exception:
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
        url = page.url.lower()
        if "/login" in url or "/checkpoint/" in url:
            raise RuntimeError(
                "LinkedIn session not authenticated. Run login bootstrap first: "
                "python -m app.linkedin_scraper login --user-data-dir <dir> --wait-seconds <seconds>"
            )

    @staticmethod
    def _thread_id_from_url(url: str, *, fallback: str) -> str:
        value = url.strip()
        if not value:
            return fallback

        urn_match = re.search(r"urn:li:[^/?&#]+", value)
        if urn_match:
            return urn_match.group(0)

        cleaned = re.sub(r"^https?://", "", value)
        cleaned = cleaned.split("?", 1)[0].split("#", 1)[0].strip("/")
        return cleaned or fallback

    @staticmethod
    def _thread_url(thread_id: str) -> str:
        value = thread_id.strip()
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return f"https://www.linkedin.com/{value.lstrip('/')}"

    @staticmethod
    def _raise_if_stopped(stop_requested: Callable[[], bool]) -> None:
        if stop_requested():
            raise RuntimeError("Run stopped by user request.")
