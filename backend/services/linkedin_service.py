from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus

from ..config.log import setup_logging
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
    return any(cookie.get("name") == "li_at" and str(cookie.get("value", "")).strip() for cookie in cookies)


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


def _default_login_launcher(user_data_dir: str, login_url: str, wait_seconds: int) -> None:
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
            args=["--incognito"],
        )
        try:
            pages = [page for page in context.pages if not page.is_closed()]
            page = pages[0] if pages else context.new_page()
            for extra_page in pages[1:]:
                extra_page.close()
            page.goto(login_url, wait_until="domcontentloaded")

            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                open_pages = [candidate for candidate in context.pages if not candidate.is_closed()]
                if not open_pages:
                    raise RuntimeError("LinkedIn login window was closed before login completed. Auth object was not created.")
                page = open_pages[0]
                if page.is_closed():
                    raise RuntimeError("LinkedIn login window was closed before login completed. Auth object was not created.")

                if any(_url_points_to_feed(candidate.url) for candidate in open_pages):
                    return
                has_session_cookie = _has_linkedin_session_cookie(context)
                if has_session_cookie and any(_url_is_authenticated(candidate.url) for candidate in open_pages):
                    return
                if has_session_cookie and _probe_context_authentication(context):
                    return
                time.sleep(0.5)

            raise RuntimeError("LinkedIn login not completed. Auth object was not created.")
        finally:
            context.close()


class LinkedinService:
    """Headless LinkedIn discovery/posting integration."""

    def __init__(self) -> None:
        SettingsStore().load()
        self.headless = _parse_headless_flag(os.getenv("LINKEDIN_HEADLESS", "true"))
        configured_dir = (os.getenv("LINKEDIN_USER_DATA_DIR") or ".reddrop/auth/linkedin").strip()
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
            raise RuntimeError("LinkedIn session not authenticated. Run LinkedIn auth login first.")

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
            with tempfile.NamedTemporaryFile(prefix="reddrop-linkedin-cookie-", suffix=".sqlite", delete=False) as handle:
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
