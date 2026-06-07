from __future__ import annotations

import builtins
from pathlib import Path
import subprocess
import sys
import types

import pytest

from backend.models import AuthProfile
from backend.models import normalize_search_plan_for_social
from backend.services.linkedin_service import LinkedinService


class _FakePage:
    def __init__(self, posts=None, posts_by_url=None):
        self._posts = list(posts or [])
        self._posts_by_url = dict(posts_by_url or {})
        self.current_url: str = ""
        self.visited_urls: list[str] = []
        self.scroll_eval_calls: int = 0
        self.evaluated_scripts: list[str] = []

    def goto(self, url, *, wait_until):
        self.current_url = str(url)
        self.visited_urls.append(self.current_url)
        _ = wait_until

    def wait_for_timeout(self, millis):
        _ = millis

    def evaluate(self, script, *args):
        _ = args
        self.evaluated_scripts.append(str(script))
        if "window.scrollBy" in script:
            self.scroll_eval_calls += 1
            return None
        scoped = self._posts_by_url.get(self.current_url)
        if scoped is not None:
            return list(scoped)
        return list(self._posts)


class _FakeContext:
    def __init__(self, posts=None, posts_by_url=None):
        self._page = _FakePage(posts=posts, posts_by_url=posts_by_url)

    def new_page(self):
        return self._page

    def close(self):
        return None


class _FakePlaywrightManager:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        _ = exc_type
        _ = exc
        _ = tb
        return False


def _install_fake_playwright(monkeypatch) -> None:
    def fake_sync_playwright():
        return _FakePlaywrightManager()

    fake_module = types.SimpleNamespace(sync_playwright=fake_sync_playwright)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)


def test_discover_relevant_threads_uses_native_playwright_only(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    def fail_if_bridge_run(*args, **kwargs):
        _ = args
        _ = kwargs
        raise AssertionError("external linkedin scraper bridge must not be called")

    monkeypatch.setattr(subprocess, "run", fail_if_bridge_run)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:555/",
                    "title": "Post title",
                    "selftext": "Post body",
                }
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=20,
        scroll_rounds=2,
        time_filter="all",
    )

    assert len(rows) == 1
    assert rows[0]["id"] == "urn:li:activity:555"
    assert rows[0]["url"] == "https://www.linkedin.com/feed/update/urn:li:activity:555/"
    assert rows[0]["title"] == "Post body"
    assert rows[0]["selftext"] == "Post body"


def test_discover_relevant_threads_truncates_title_to_90_chars(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    long_post = (
        "This is a long LinkedIn post body that should drive the conversation title and be "
        "truncated to ninety characters exactly for list readability."
    )
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:999/",
                    "title": "Author Name",
                    "selftext": long_post,
                }
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=20,
        scroll_rounds=2,
        time_filter="all",
    )

    assert len(rows) == 1
    assert rows[0]["title"] == long_post[:90]
    assert len(rows[0]["title"]) == 90


def test_discover_relevant_threads_raises_when_playwright_missing(
    monkeypatch, tmp_path
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "playwright.sync_api":
            raise ImportError("playwright not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)

    with pytest.raises(RuntimeError, match="requires playwright"):
        service.discover_relevant_threads(
            content="crm sales",
            threads_limit=20,
            scroll_rounds=2,
        )


def test_discover_relevant_threads_applies_time_filter_and_recency_sort(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:111/",
                    "title": "Older post",
                    "published": "2d",
                    "selftext": "Pipeline review 2d 5 comments 3 reactions",
                },
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:222/",
                    "title": "Newer post",
                    "published": "2h",
                    "selftext": "Forecast update 2h 1 comment 10 reactions",
                },
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:333/",
                    "title": "Newest post",
                    "published": "30m",
                    "selftext": "Deal desk notes 30m 2 comments 1 reaction",
                },
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=20,
        scroll_rounds=2,
        time_filter="day",
        sort_by="recency",
    )

    assert [row["id"] for row in rows] == [
        "urn:li:activity:333",
        "urn:li:activity:222",
    ]
    assert all(row["created_utc"] is not None for row in rows)


def test_discover_relevant_threads_orders_by_engagement(monkeypatch, tmp_path) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:444/",
                    "title": "Lower engagement",
                    "selftext": "Notes 1h 1 comment 2 reactions",
                },
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:555/",
                    "title": "Higher engagement",
                    "selftext": "Notes 1h 25 comments 40 reactions",
                },
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=20,
        scroll_rounds=2,
        time_filter="all",
        sort_by="engagement",
    )

    assert [row["id"] for row in rows] == [
        "urn:li:activity:555",
        "urn:li:activity:444",
    ]
    assert rows[0]["num_comments"] == 25
    assert rows[0]["score"] == 40


def test_discover_relevant_threads_uses_linkedin_recency_query_params(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    fake_context = _FakeContext(
        posts=[
            {
                "url": "https://www.linkedin.com/feed/update/urn:li:activity:901/",
                "title": "Post title",
                "selftext": "Post body 1h",
            }
        ]
    )
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="pluto AND pizza",
        threads_limit=20,
        scroll_rounds=2,
        time_filter="all",
        sort_by="recency",
    )

    assert len(rows) == 1
    assert len(fake_context._page.visited_urls) >= 1
    url = fake_context._page.visited_urls[0]
    assert "search/results/content/?" in url
    assert "origin=GLOBAL_SEARCH_HEADER" in url
    assert "sortBy=%5B%22date_posted%22%5D" in url
    assert "datePosted=" not in url


def test_discover_relevant_threads_uses_linkedin_relevance_and_past_month_params(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    fake_context = _FakeContext(
        posts=[
            {
                "url": "https://www.linkedin.com/feed/update/urn:li:activity:902/",
                "title": "Post title",
                "selftext": "Post body 2d",
            }
        ]
    )
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="pluto AND pizza",
        threads_limit=20,
        scroll_rounds=2,
        time_filter="month",
        sort_by="relevance",
    )

    assert len(rows) == 1
    assert len(fake_context._page.visited_urls) >= 1
    url = fake_context._page.visited_urls[0]
    assert "search/results/content/?" in url
    assert "origin=FACETED_SEARCH" in url
    assert "sortBy=%5B%22relevance%22%5D" in url
    assert "datePosted=%5B%22past-month%22%5D" in url


def test_discover_relevant_threads_excludes_company_posts_pages(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/company/franlift/posts/",
                    "title": "FranLift",
                    "selftext": "FranLift 3m Follow",
                },
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:123456789/",
                    "title": "Valid post",
                    "selftext": "Post body 20m 7 comments 11 reactions",
                },
                {
                    "url": "https://www.linkedin.com/posts/jane-doe_activity-987654321-abcD/",
                    "title": "Valid posts permalink",
                    "selftext": "Post body 2h 3 comments 4 reactions",
                },
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="sales crm",
        threads_limit=20,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
    )

    urls = [row["url"] for row in rows]
    assert "https://www.linkedin.com/company/franlift/posts/" not in urls
    assert "https://www.linkedin.com/feed/update/urn:li:activity:123456789/" in urls
    assert "https://www.linkedin.com/posts/jane-doe_activity-987654321-abcD/" in urls


def test_discover_relevant_threads_enriches_seed_rows_from_detail_pages(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    fake_context = _FakeContext(
        posts=[
            {
                "url": "https://www.linkedin.com/feed/update/urn:li:activity:5001/",
                "title": "Seed title",
                "selftext": "Short seed",
            }
        ],
        posts_by_url={
            "https://www.linkedin.com/feed/update/urn:li:activity:5001/": [
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:5001/",
                    "title": "Detail title",
                    "selftext": "Long detail body 2h 12 comments 25 reactions",
                }
            ]
        },
    )
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="sales crm",
        threads_limit=20,
        scroll_rounds=2,
        time_filter="all",
        sort_by="engagement",
    )

    assert len(rows) == 1
    assert rows[0]["title"] == "Long detail body 2h 12 comments 25 reactions"
    assert rows[0]["selftext"] == "Long detail body 2h 12 comments 25 reactions"
    assert rows[0]["num_comments"] == 12
    assert rows[0]["score"] == 25


def test_discover_relevant_threads_visits_only_threads_limit_detail_pages(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    seed_posts = [
        {
            "url": f"https://www.linkedin.com/feed/update/urn:li:activity:{idx}/",
            "title": f"Row {idx}",
            "selftext": "Seed",
        }
        for idx in [7001, 7002, 7003, 7004, 7005]
    ]
    fake_context = _FakeContext(posts=seed_posts)
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="sales crm",
        threads_limit=3,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
    )

    assert len(rows) == 3
    detail_visits = [
        url
        for url in fake_context._page.visited_urls
        if "/feed/update/urn:li:activity:" in url
    ]
    assert len(detail_visits) == 3


def test_discover_relevant_threads_uses_mouse_wheel_for_scroll_rounds_when_available(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    fake_context = _FakeContext(posts=[])
    wheel_calls: list[tuple[int, int]] = []
    fake_context._page.mouse = types.SimpleNamespace(
        wheel=lambda x, y: wheel_calls.append((int(x), int(y)))
    )
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="sales crm",
        threads_limit=10,
        scroll_rounds=4,
        time_filter="all",
        sort_by="recency",
    )

    assert rows == []
    assert wheel_calls == [(0, 3200), (0, 3200), (0, 3200), (0, 3200)]
    assert fake_context._page.scroll_eval_calls == 0


def test_discover_relevant_threads_scans_phase1_beyond_threads_limit(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    fake_context = _FakeContext(posts=[])
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    seen_phase_limits: list[int] = []

    def fake_phase_rows(*, page, max_posts):
        _ = page
        seen_phase_limits.append(int(max_posts))
        return []

    monkeypatch.setattr(service, "_phase_rows_from_page", fake_phase_rows)
    monkeypatch.setattr(service, "_collect_post_links", lambda rows, max_posts: [])

    rows = service.discover_relevant_threads(
        content="sales crm",
        threads_limit=10,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
    )

    assert rows == []
    assert seen_phase_limits
    assert seen_phase_limits[0] > 10


def test_discover_relevant_threads_falls_back_to_js_scroll_when_wheel_does_not_move(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    fake_context = _FakeContext(posts=[])
    wheel_calls: list[tuple[int, int]] = []
    fake_context._page.mouse = types.SimpleNamespace(
        wheel=lambda x, y: wheel_calls.append((int(x), int(y)))
    )
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)
    monkeypatch.setattr(
        service, "_read_scroll_position", lambda page: 0.0, raising=False
    )

    rows = service.discover_relevant_threads(
        content="sales crm",
        threads_limit=10,
        scroll_rounds=4,
        time_filter="all",
        sort_by="recency",
    )

    assert rows == []
    assert wheel_calls == [(0, 3200), (0, 3200), (0, 3200), (0, 3200)]
    assert fake_context._page.scroll_eval_calls == 4


def test_discover_relevant_threads_accumulates_phase_rows_across_scroll_rounds(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    fake_context = _FakeContext(posts=[])
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)
    monkeypatch.setattr(service, "_load_post_detail_rows", lambda **kwargs: {})
    monkeypatch.setattr(service, "_collect_post_links", lambda rows, max_posts: [])

    sequence = [
        [
            {
                "url": "https://www.linkedin.com/feed/update/urn:li:activity:8101/",
                "title": "Row 1",
                "selftext": "Body 1h",
            }
        ],
        [
            {
                "url": "https://www.linkedin.com/feed/update/urn:li:activity:8102/",
                "title": "Row 2",
                "selftext": "Body 2h",
            }
        ],
        [
            {
                "url": "https://www.linkedin.com/feed/update/urn:li:activity:8103/",
                "title": "Row 3",
                "selftext": "Body 3h",
            }
        ],
    ]

    def fake_phase_rows(*, page, max_posts):
        _ = page
        _ = max_posts
        if sequence:
            return sequence.pop(0)
        return []

    monkeypatch.setattr(service, "_phase_rows_from_page", fake_phase_rows)

    rows = service.discover_relevant_threads(
        content="sales crm",
        threads_limit=10,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
    )

    assert [row["id"] for row in rows] == [
        "urn:li:activity:8101",
        "urn:li:activity:8102",
        "urn:li:activity:8103",
    ]


def test_scroll_active_container_js_targets_workspace() -> None:
    class _RecorderPage:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def evaluate(self, script, delta):
            self.calls.append((str(script), int(delta)))
            return None

    page = _RecorderPage()
    LinkedinService._scroll_active_container(page=page, delta_y=3200)

    assert len(page.calls) == 1
    script, delta = page.calls[0]
    assert delta == 3200
    assert "main#workspace" in script


def test_canonicalize_post_url_accepts_generic_posts_permalink() -> None:
    canonical = LinkedinService._canonicalize_post_url(
        "https://www.linkedin.com/posts/pluto-and-pizza/"
    )
    assert canonical == "https://www.linkedin.com/posts/pluto-and-pizza/"


def test_discover_relevant_threads_waits_for_selectors_and_expands_comments(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    context = _FakeContext(
        posts=[
            {
                "url": "https://www.linkedin.com/feed/update/urn:li:activity:9901/",
                "title": "Seed",
                "selftext": "Seed body",
            }
        ],
        posts_by_url={
            "https://www.linkedin.com/feed/update/urn:li:activity:9901/": [
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:9901/",
                    "title": "Detail",
                    "selftext": "Detail body",
                }
            ]
        },
    )
    monkeypatch.setattr(service, "_launch_context", lambda playwright: context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    wait_calls: list[tuple[int, tuple[str, ...]]] = []
    expand_calls: list[int] = []

    def fake_wait(*, page, selectors, timeout_seconds=25):
        _ = page
        wait_calls.append((int(timeout_seconds), tuple(selectors)))
        return True

    def fake_expand(*, page, max_clicks):
        _ = page
        expand_calls.append(int(max_clicks))

    monkeypatch.setattr(service, "_wait_for_any_selector", fake_wait, raising=False)
    monkeypatch.setattr(service, "_expand_comment_threads", fake_expand, raising=False)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=1,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
    )

    assert len(rows) == 1
    assert wait_calls
    assert any("div.feed-shared-update-v2" in selectors for _, selectors in wait_calls)
    assert expand_calls


def test_discover_relevant_threads_fails_when_authwall_html_is_detected(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    context = _FakeContext(posts=[])
    context._page.content = lambda: "<html><body>authwall sign in</body></html>"
    monkeypatch.setattr(service, "_launch_context", lambda playwright: context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    with pytest.raises(RuntimeError, match="auth wall"):
        service.discover_relevant_threads(
            content="crm sales",
            threads_limit=5,
            scroll_rounds=2,
            time_filter="week",
            sort_by="relevance",
        )


def test_discover_relevant_threads_uses_published_iso_for_real_post_date(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:777/",
                    "title": "Post title",
                    "selftext": "Post body",
                    "published_iso": "2025-01-01T12:00:00Z",
                }
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=5,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
    )

    assert len(rows) == 1
    assert rows[0]["published_at"] == "2025-01-01T12:00:00Z"
    assert float(rows[0]["created_utc"]) == pytest.approx(1735732800.0, abs=1.0)


def test_search_result_selector_coverage_includes_reusable_search_cards() -> None:
    assert (
        "li.reusable-search__result-container" in LinkedinService._POST_CARD_SELECTORS
    )


def test_collect_phase_rows_with_scroll_stops_after_three_no_new_rounds(
    monkeypatch, tmp_path
) -> None:
    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    page = object()

    repeated_row = {
        "url": "https://www.linkedin.com/feed/update/urn:li:activity:9301/",
        "title": "Row 1",
        "selftext": "Body",
    }
    phase_call_count = {"value": 0}
    scroll_calls: list[int] = []
    wait_calls: list[int] = []

    def fake_phase_rows(*, page, max_posts):
        _ = page
        _ = max_posts
        phase_call_count["value"] += 1
        return [dict(repeated_row)]

    def fake_scroll(*, page, rounds, stop_requested, delta_y, pause_ms):
        _ = page
        _ = rounds
        _ = stop_requested
        _ = delta_y
        _ = pause_ms
        scroll_calls.append(1)

    def fake_wait(*, page, selectors, timeout_seconds=25):
        _ = page
        _ = selectors
        wait_calls.append(int(timeout_seconds))
        return True

    monkeypatch.setattr(service, "_phase_rows_from_page", fake_phase_rows)
    monkeypatch.setattr(service, "_scroll_for_discovery", fake_scroll)
    monkeypatch.setattr(service, "_wait_for_any_selector", fake_wait)

    rows = service._collect_phase_rows_with_scroll(
        page=page,
        rounds=20,
        max_posts=20,
        wait_timeout_seconds=5,
        stop_requested=lambda: False,
    )

    assert len(rows) == 1
    assert phase_call_count["value"] == 7
    assert len(scroll_calls) == 6
    assert len(wait_calls) == 6


def test_discover_relevant_threads_caps_detail_enrichment_links(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service, "_launch_context", lambda playwright: _FakeContext(posts=[])
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)
    monkeypatch.setattr(service, "_wait_for_any_selector", lambda **kwargs: True)

    seed_rows = [
        {
            "url": f"https://www.linkedin.com/feed/update/urn:li:activity:{9700 + idx}/",
            "title": f"Row {idx}",
            "selftext": "Short",
            "published_iso": "",
        }
        for idx in range(40)
    ]
    monkeypatch.setattr(
        service, "_collect_phase_rows_with_scroll", lambda **kwargs: list(seed_rows)
    )
    captured = {"count": 0}

    def fake_load_post_detail_rows(
        *,
        context,
        post_urls,
        scroll_rounds,
        selector_wait_timeout_seconds,
        detail_wait_timeout_seconds,
        max_comment_expand_clicks,
        stop_requested,
    ):
        _ = context
        _ = scroll_rounds
        _ = selector_wait_timeout_seconds
        _ = detail_wait_timeout_seconds
        _ = max_comment_expand_clicks
        _ = stop_requested
        captured["count"] = len(post_urls)
        return {}

    monkeypatch.setattr(service, "_load_post_detail_rows", fake_load_post_detail_rows)

    rows = service.discover_relevant_threads(
        content="sales crm",
        threads_limit=40,
        scroll_rounds=3,
        time_filter="all",
        sort_by="relevance",
    )

    assert len(rows) == 40
    assert captured["count"] == 12


def test_matches_time_filter_includes_unknown_created_utc_for_bounded_ranges_by_default() -> (
    None
):
    now_utc = 1_700_000_000.0
    assert (
        LinkedinService._matches_time_filter(
            created_utc=None, time_filter="all", now_utc=now_utc
        )
        is True
    )
    assert (
        LinkedinService._matches_time_filter(
            created_utc=None, time_filter="week", now_utc=now_utc
        )
        is True
    )


def test_matches_time_filter_can_exclude_unknown_created_utc_when_flag_disabled(
    monkeypatch,
) -> None:
    now_utc = 1_700_000_000.0
    monkeypatch.setenv(
        "LINKEDIN_INCLUDE_UNKNOWN_TIMESTAMPS_FOR_BOUNDED_FILTERS", "false"
    )
    assert (
        LinkedinService._matches_time_filter(
            created_utc=None, time_filter="week", now_utc=now_utc
        )
        is False
    )


def test_resolve_created_timestamp_parses_row_published_when_iso_missing() -> None:
    created_utc, published_at = LinkedinService._resolve_created_timestamp(
        row={"published": "5d"},
        selftext="body without relative timestamp",
        now_utc=1_700_000_000.0,
    )
    assert created_utc is not None
    assert float(created_utc) == pytest.approx(1_700_000_000.0 - 5 * 86400, abs=1.0)
    assert published_at == "2023-11-09"


def test_resolve_created_timestamp_does_not_infer_from_selftext_when_published_missing() -> (
    None
):
    created_utc, published_at = LinkedinService._resolve_created_timestamp(
        row={},
        selftext="this body mentions 2h but is not a timestamp field",
        now_utc=1_700_000_000.0,
    )
    assert created_utc is None
    assert published_at is None


def test_discover_relevant_threads_uses_saved_broader_query_when_first_pass_sparse(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    fake_context = _FakeContext(posts=[])
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)
    monkeypatch.setattr(service, "_wait_for_any_selector", lambda **kwargs: True)
    monkeypatch.setattr(service, "_load_post_detail_rows", lambda **kwargs: {})

    def fake_collect_phase_rows_with_scroll(**kwargs):
        page = kwargs["page"]
        current_url = str(getattr(page, "current_url", ""))
        if "keywords=crm%20sales" in current_url:
            return [
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:777001/",
                    "title": "Variant hit",
                    "selftext": "Post body 1h",
                }
            ]
        return []

    monkeypatch.setattr(
        service, "_collect_phase_rows_with_scroll", fake_collect_phase_rows_with_scroll
    )

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=10,
        scroll_rounds=2,
        time_filter="week",
        sort_by="relevance",
        search_plan={
            "query": '("CRM adoption" OR "CRM hygiene") AND ("sales activity" OR "activity logging")',
            "queries": [
                '("CRM adoption" OR "CRM hygiene") AND ("sales activity" OR "activity logging")',
                "crm sales",
            ],
        },
    )

    assert len(rows) == 1
    assert rows[0]["id"] == "urn:li:activity:777001"
    search_visits = [
        url
        for url in fake_context._page.visited_urls
        if "search/results/content/" in url
    ]
    assert len(search_visits) >= 2


def test_query_variants_include_compact_late_terms_for_boolean_queries() -> None:
    variants = LinkedinService._query_variants(
        '("sales activity" OR "deal context" OR "deal updates") '
        'CRM (logging OR "data entry" OR capture) '
        '("follow-up" OR "follow ups" OR forecasting OR "CRM adoption")'
    )

    assert any(
        len(variant.split()) <= 8 and "forecasting" in variant and "adoption" in variant
        for variant in variants
    )


def test_canonicalize_post_url_extracts_encoded_activity_urn_from_query_params() -> (
    None
):
    canonical = LinkedinService._canonicalize_post_url(
        "https://www.linkedin.com/company/acme/posts/?tracking=abc&updateEntityUrn=urn%3Ali%3Aactivity%3A1234567890"
    )
    assert (
        canonical == "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/"
    )


def test_canonicalize_post_url_decodes_escaped_hydration_urn() -> None:
    raw_url = (
        r"https://www.linkedin.com/company/acme/posts/"
        r"?updateEntityUrn=urn\u003Ali\u003Aactivity\u003A1234567890"
    )
    canonical = LinkedinService._canonicalize_post_url(raw_url)
    assert (
        canonical == "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/"
    )


def test_discover_relevant_threads_recovers_post_from_company_link_with_encoded_urn(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/company/acme/posts/?tracking=abc&updateEntityUrn=urn%3Ali%3Aactivity%3A999000111/",
                    "title": "Company context row",
                    "selftext": "crm sales 2h",
                }
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=10,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
    )

    assert len(rows) == 1
    assert rows[0]["id"] == "urn:li:activity:999000111"
    assert (
        rows[0]["url"]
        == "https://www.linkedin.com/feed/update/urn:li:activity:999000111/"
    )


def test_discover_recovers_post_from_escaped_hydration_urn(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": r"urn\u003Ali\u003Aactivity\u003A999000222",
                    "title": "Hydration row",
                    "selftext": "crm follow-up context 1h",
                }
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=10,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
    )

    assert len(rows) == 1
    assert rows[0]["id"] == "urn:li:activity:999000222"
    assert (
        rows[0]["url"]
        == "https://www.linkedin.com/feed/update/urn:li:activity:999000222/"
    )


def test_discover_relevant_threads_computes_similarity_from_topic(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:222333444/",
                    "title": "CRM adoption tips",
                    "selftext": "How to improve sales activity logging",
                }
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)
    monkeypatch.setattr(
        service,
        "_apply_semantic_similarity_scores",
        lambda *, topic, rows, stop_requested: [
            {
                **row,
                "semantic_similarity": (0.91 if topic == "crm sales topic" else 0.0),
            }
            for row in rows
        ],
        raising=False,
    )

    rows = service.discover_relevant_threads(
        content="crm sales topic",
        threads_limit=10,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
    )

    assert len(rows) == 1
    assert rows[0]["semantic_similarity"] == pytest.approx(0.91)


def test_linkedin_search_plan_preserves_multiple_queries() -> None:
    plan = normalize_search_plan_for_social(
        social="linkedin",
        topic="crm sales",
        raw={
            "query": "primary crm query",
            "queries": [
                "primary crm query",
                "pipeline hygiene",
                "forecasting adoption",
            ],
        },
    )

    assert plan.query == "primary crm query"
    assert plan.queries == [
        "primary crm query",
        "pipeline hygiene",
        "forecasting adoption",
    ]


def test_discover_relevant_threads_runs_all_linkedin_plan_queries(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    fake_context = _FakeContext(posts=[])
    monkeypatch.setattr(service, "_launch_context", lambda playwright: fake_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)
    monkeypatch.setattr(service, "_wait_for_any_selector", lambda **kwargs: True)
    monkeypatch.setattr(service, "_load_post_detail_rows", lambda **kwargs: {})
    monkeypatch.setattr(service, "_query_variants", lambda query: [query])

    def fake_collect_phase_rows_with_scroll(**kwargs):
        page = kwargs["page"]
        current_url = str(getattr(page, "current_url", ""))
        if "keywords=pipeline%20hygiene" in current_url:
            return [
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:8801/",
                    "title": "Pipeline hit",
                    "selftext": "Pipeline hygiene post 1h",
                }
            ]
        return []

    monkeypatch.setattr(
        service, "_collect_phase_rows_with_scroll", fake_collect_phase_rows_with_scroll
    )

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=10,
        scroll_rounds=2,
        time_filter="all",
        sort_by="relevance",
        search_plan={
            "query": "strict query",
            "queries": ["strict query", "pipeline hygiene"],
        },
    )

    assert [row["id"] for row in rows] == ["urn:li:activity:8801"]


def test_linkedin_query_passes_use_only_locked_plan_queries(tmp_path) -> None:
    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)

    passes = service._query_passes_from_search_plan(
        search_plan={
            "query": '"strict crm" AND "sales ops"',
            "queries": [
                '"strict crm" AND "sales ops"',
                "pipeline hygiene",
                "forecasting adoption",
            ],
        },
        fallback="raw user topic",
    )

    assert passes == [
        '"strict crm" AND "sales ops"',
        "pipeline hygiene",
        "forecasting adoption",
    ]


def test_detail_authwall_errors_are_not_swallowed(monkeypatch, tmp_path) -> None:
    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    context = _FakeContext(posts=[])
    context._page.content = lambda: "<html><body>authwall sign in</body></html>"
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    with pytest.raises(RuntimeError, match="auth wall"):
        service._load_post_detail_rows(
            context=context,
            post_urls=["https://www.linkedin.com/feed/update/urn:li:activity:8802/"],
            scroll_rounds=1,
            selector_wait_timeout_seconds=1,
            detail_wait_timeout_seconds=1,
            max_comment_expand_clicks=1,
            stop_requested=lambda: False,
        )


def test_discover_relevant_threads_uses_structured_engagement_counts(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service,
        "_launch_context",
        lambda playwright: _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:8803/",
                    "title": "Structured counts",
                    "selftext": "No count tokens in this body",
                    "reactions_count": 17,
                    "comments_count": 9,
                }
            ]
        ),
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=10,
        scroll_rounds=2,
        time_filter="all",
        sort_by="engagement",
    )

    assert len(rows) == 1
    assert rows[0]["score"] == 17
    assert rows[0]["num_comments"] == 9


def test_discover_relevant_threads_uses_profile_runtime_lock(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    events: list[str] = []

    class _RecorderLock:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type
            _ = exc
            _ = tb
            events.append("exit")
            return False

    def fake_launch_context(playwright):
        _ = playwright
        assert events == ["enter"]
        return _FakeContext(
            posts=[
                {
                    "url": "https://www.linkedin.com/feed/update/urn:li:activity:8804/",
                    "title": "Locked",
                    "selftext": "Locked profile scrape",
                }
            ]
        )

    monkeypatch.setattr(service, "_profile_runtime_lock", lambda: _RecorderLock())
    monkeypatch.setattr(service, "_launch_context", fake_launch_context)
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)

    rows = service.discover_relevant_threads(
        content="crm sales",
        threads_limit=10,
        scroll_rounds=2,
        time_filter="all",
    )

    assert len(rows) == 1
    assert events == ["enter", "exit"]


def test_discover_relevant_threads_accepts_detail_enrich_top_k(
    monkeypatch, tmp_path
) -> None:
    _install_fake_playwright(monkeypatch)

    auth = AuthProfile(
        id="linkedin-auth",
        social="linkedin",
        linkedin_headless="true",
        linkedin_user_data_dir=str(tmp_path / "profile"),
    )
    service = LinkedinService(auth=auth)
    monkeypatch.setattr(
        service, "_launch_context", lambda playwright: _FakeContext(posts=[])
    )
    monkeypatch.setattr(service, "_ensure_authenticated", lambda page: None)
    monkeypatch.setattr(service, "_wait_for_any_selector", lambda **kwargs: True)
    seed_rows = [
        {
            "url": f"https://www.linkedin.com/feed/update/urn:li:activity:{8900 + idx}/",
            "title": f"Row {idx}",
            "selftext": "Short",
            "published_iso": "",
        }
        for idx in range(40)
    ]
    monkeypatch.setattr(
        service, "_collect_phase_rows_with_scroll", lambda **kwargs: list(seed_rows)
    )
    captured = {"count": 0}

    def fake_load_post_detail_rows(**kwargs):
        captured["count"] = len(kwargs["post_urls"])
        return {}

    monkeypatch.setattr(service, "_load_post_detail_rows", fake_load_post_detail_rows)

    rows = service.discover_relevant_threads(
        content="sales crm",
        threads_limit=40,
        scroll_rounds=3,
        time_filter="all",
        sort_by="relevance",
        detail_enrich_top_k=25,
    )

    assert len(rows) == 40
    assert captured["count"] == 25


def test_sentence_transformers_is_declared_for_semantic_scoring() -> None:
    requirements = Path("backend/requirements.txt").read_text(encoding="utf-8")

    assert "sentence-transformers" in requirements
