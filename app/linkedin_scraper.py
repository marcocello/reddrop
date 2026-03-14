from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _resolve_user_data_dir(raw_dir: str) -> Path:
    raw = (raw_dir or "").strip() or ".reddrop/auth/linkedin"
    candidate = Path(raw).expanduser()
    resolved = candidate if candidate.is_absolute() else (_REPO_ROOT / candidate)
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
        resolved = (_REPO_ROOT / ".reddrop" / relative_tail).resolve()
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.linkedin_scraper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login = subparsers.add_parser("login", help="Open LinkedIn login in a persistent browser profile.")
    login.add_argument(
        "--user-data-dir",
        default=str(_REPO_ROOT / ".reddrop/auth/linkedin"),
        help="Persistent profile directory used by Playwright.",
    )
    login.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
        help="How long to keep the browser open for manual login.",
    )
    return parser


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
            print(
                f"LinkedIn login opened. Complete login in the browser (max {wait_seconds}s); "
                "the browser closes automatically after success."
            )

            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                open_pages = [candidate for candidate in context.pages if not candidate.is_closed()]
                if not open_pages:
                    raise RuntimeError("LinkedIn login window was closed before login completed. Auth object was not created.")
                page = open_pages[0]
                if page.is_closed():
                    raise RuntimeError("LinkedIn login window was closed before login completed. Auth object was not created.")

                if any(_url_points_to_feed(candidate.url) for candidate in open_pages):
                    print("LinkedIn login completed. Closing browser.")
                    return
                has_session_cookie = _has_linkedin_session_cookie(context)
                if has_session_cookie and any(_url_is_authenticated(candidate.url) for candidate in open_pages):
                    print("LinkedIn login completed. Closing browser.")
                    return
                if has_session_cookie and _probe_context_authentication(context):
                    print("LinkedIn login completed. Closing browser.")
                    return
                time.sleep(0.5)

            raise RuntimeError("LinkedIn login not completed. Auth object was not created.")
        finally:
            context.close()


def login_and_save_session(
    user_data_dir: str,
    login_url: str = "https://www.linkedin.com/login",
    wait_seconds: int = 120,
    launcher: Callable[[str, str, int], None] | None = None,
) -> dict[str, Any]:
    if wait_seconds <= 0:
        raise ValueError("--wait-seconds must be greater than 0.")

    profile_dir = _resolve_user_data_dir(user_data_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    effective_launcher = launcher or _default_login_launcher
    effective_launcher(str(profile_dir), login_url, wait_seconds)

    return {
        "status": "session_saved",
        "user_data_dir": str(profile_dir),
        "login_url": login_url,
    }


def _run_login(*, user_data_dir: str, wait_seconds: int) -> int:
    try:
        payload = login_and_save_session(user_data_dir=user_data_dir, wait_seconds=wait_seconds)
    except Exception as exc:
        print(str(exc))
        return 1

    print("LinkedIn login flow finished.")
    print(payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "login":
        return _run_login(user_data_dir=args.user_data_dir, wait_seconds=args.wait_seconds)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
