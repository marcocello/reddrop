from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from ..config.log import setup_logging
from ..models import Conversation, PeriodicSearchJob, SearchArtifact
from ..services.auth_store import AuthStore
from ..services.job_config_store import JobConfigStore
from ..services.persona_store import PersonaStore
from ..services.search_query_store import SearchQueryStore
from ..services.reply_generation_service import ReplyGenerationService
from ..services.linkedin_service import LinkedinService
from ..services.reddit_service import RedditService
from ..services.search_storage import SearchStorage
from .commands import add as add_command
from .commands import reply as reply_command
from .commands import search as search_command
from .commands import send as send_command

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOME = _BACKEND_ROOT / ".reddrop"
DEFAULT_JOBS_CONFIG = DEFAULT_HOME / "jobs_config.json"
DEFAULT_RUNS_DIR = DEFAULT_HOME / "runs"
DEFAULT_TIME_FILTER = "week"
DEFAULT_SUBREDDIT_LIMIT = 5
DEFAULT_THREADS_LIMIT = 10
DEFAULT_REPLIES_PER_ITERATION = 3
DEFAULT_MIN_SIMILARITY_SCORE = 0.35
DEFAULT_MAX_RUNTIME_MINUTES = 1440
JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
logger = setup_logging(__name__)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, default=str))


def _error(message: str) -> int:
    logger.error(message)
    _emit({"error": message})
    return 1


def _job_store() -> JobConfigStore:
    return JobConfigStore(DEFAULT_JOBS_CONFIG)


def _search_query_store() -> SearchQueryStore:
    return SearchQueryStore(DEFAULT_HOME / "search_queries.json")


def _search_storage() -> SearchStorage:
    return SearchStorage(DEFAULT_RUNS_DIR)


def _auth_store() -> AuthStore:
    return AuthStore(
        DEFAULT_HOME / "auth", legacy_candidates=[DEFAULT_HOME / "auths.json"]
    )


def _persona_store() -> PersonaStore:
    try:
        return PersonaStore([DEFAULT_HOME / "personas.json"])
    except TypeError:
        return PersonaStore()


def _resolve_social(raw: object) -> str:
    value = str(raw or "").strip().lower()
    if value == "linkedin":
        return "linkedin"
    return "reddit"


def _resolve_linkedin_user_data_dir(raw_dir: str) -> str:
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
    return str(resolved)


def _apply_auth_to_env(job: PeriodicSearchJob) -> None:
    auth_id = (job.auth_id or "").strip()
    if not auth_id:
        raise RuntimeError(f"auth_id is required for job {job.name}")

    auth = _auth_store().get_auth(auth_id)
    if auth is None:
        raise RuntimeError(f"Auth not found for job {job.name}: {auth_id}")

    auth_social = _resolve_social(auth.social)
    job_social = _resolve_social(job.social)
    if auth_social != job_social:
        raise RuntimeError(
            f"Auth social mismatch for job {job.name}: {auth_social} != {job_social}"
        )

    if job_social == "linkedin":
        os.environ["LINKEDIN_EMAIL"] = auth.linkedin_email.strip()
        os.environ["LINKEDIN_PASSWORD"] = auth.linkedin_password.strip()
        os.environ["LINKEDIN_HEADLESS"] = auth.linkedin_headless.strip() or "true"
        os.environ["LINKEDIN_USER_DATA_DIR"] = _resolve_linkedin_user_data_dir(
            auth.linkedin_user_data_dir.strip()
        )
        os.environ["LINKEDIN_WAIT_SECONDS"] = (
            auth.linkedin_wait_seconds.strip() or "120"
        )
        return

    client_id = auth.reddit_client_id.strip()
    client_secret = auth.reddit_client_secret.strip()
    user_agent = auth.reddit_user_agent.strip()
    if client_id:
        os.environ["REDDIT_CLIENT_ID"] = client_id
    if client_secret:
        os.environ["REDDIT_CLIENT_SECRET"] = client_secret
    if user_agent:
        os.environ["REDDIT_USER_AGENT"] = user_agent
    elif not (os.getenv("REDDIT_USER_AGENT") or "").strip():
        os.environ["REDDIT_USER_AGENT"] = "Reddrop:v1.0"
    os.environ["REDDIT_USERNAME"] = auth.reddit_username.strip()
    os.environ["REDDIT_PASSWORD"] = auth.reddit_password.strip()


def _saved_search_query_for_job(job: PeriodicSearchJob):
    if job.job_type != "search":
        raise RuntimeError(
            f"Locked search query is only valid for search jobs: {job.name}"
        )
    path_id = str(job.search_query_id or "").strip()
    if not path_id:
        raise RuntimeError(f"Search job {job.name} is missing search_query_id.")
    path = _search_query_store().get_query(path_id)
    if path is None:
        raise RuntimeError(f"Search query not found for job {job.name}: {path_id}")
    if path.status != "saved":
        raise RuntimeError(
            f"Search query must be saved for job {job.name}: {path.name}"
        )
    expected_version = max(1, int(path.version or 1))
    configured_version = getattr(job, "search_query_version", None)
    if configured_version is not None and int(configured_version) != expected_version:
        raise RuntimeError(
            f"Search query version mismatch for job {job.name}: "
            f"job={int(configured_version)} path={expected_version}."
        )
    return path


def _prepare_search_job_for_runtime(job: PeriodicSearchJob) -> None:
    if job.job_type == "search":
        path = _saved_search_query_for_job(job)
        job.social = _resolve_social(path.social)
        job.topic = path.topic
        job.search_plan = path.plan
    _apply_auth_to_env(job)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be an integer.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return value


def _similarity_score(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Value must be a number between 0 and 1."
        ) from exc
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError("Value must be between 0 and 1.")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reddrop", description="Reddrop CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="Create or update the job configuration.")
    add.add_argument("--name", required=True)
    add.add_argument("--job-type", default="search", choices=["search", "reply"])
    add.add_argument("--auth-id")
    add.add_argument("--source-job-id")
    add.add_argument("--social", default="reddit", choices=["reddit", "linkedin"])
    add.add_argument(
        "--time-filter",
        default=DEFAULT_TIME_FILTER,
        choices=["hour", "day", "week", "month", "year", "all"],
    )
    add.add_argument("--subreddit-limit", type=int, default=DEFAULT_SUBREDDIT_LIMIT)
    add.add_argument("--threads-limit", type=int, default=DEFAULT_THREADS_LIMIT)
    add.add_argument("--scroll-rounds", type=_positive_int, default=15)
    add.add_argument(
        "--sort-by", default="relevance", choices=["relevance", "recency", "engagement"]
    )
    add.add_argument("--selector-wait-timeout-seconds", type=_positive_int, default=20)
    add.add_argument("--detail-wait-timeout-seconds", type=_positive_int, default=8)
    add.add_argument("--max-comment-expand-clicks", type=_positive_int, default=30)
    add.add_argument("--detail-enrich-top-k", type=_positive_int, default=12)
    add.add_argument("--search-query-id")
    add.add_argument("--search-query-version", type=_positive_int)
    add.add_argument(
        "--replies-per-iteration",
        type=_positive_int,
        default=DEFAULT_REPLIES_PER_ITERATION,
    )
    add.add_argument("--min-similarity-score", type=_similarity_score, default=0.35)
    add.add_argument(
        "--max-runtime-minutes", type=_positive_int, default=DEFAULT_MAX_RUNTIME_MINUTES
    )
    add.add_argument("--personas", nargs="*", default=[])

    search = subparsers.add_parser(
        "search",
        help="Search configured social platforms with saved job configuration.",
    )
    search.add_argument("name", nargs="?")

    reply = subparsers.add_parser(
        "reply", help="Generate replies from a search artifact."
    )
    reply.add_argument("search_file")
    reply.add_argument("--replies", type=_positive_int, default=3)
    reply.add_argument("--min-similarity", type=_similarity_score, default=0.35)
    reply.add_argument("--personas", required=True)

    send = subparsers.add_parser(
        "send", help="Send unsent replies from a search artifact."
    )
    send.add_argument("search_file")

    completion = subparsers.add_parser(
        "completion", help="Print shell completion script."
    )
    completion.add_argument("shell", choices=["bash", "zsh"])

    complete = subparsers.add_parser("__complete_search_names", help=argparse.SUPPRESS)
    complete.add_argument("prefix", nargs="?", default="")
    return parser


def _complete_search_names(store: JobConfigStore, prefix: str) -> int:
    normalized_prefix = prefix.strip().lower()
    names = sorted({job.name for job in store.list_jobs()})
    for name in names:
        if not normalized_prefix or name.lower().startswith(normalized_prefix):
            print(name)
    return 0


def _resolve_prog_name() -> str:
    candidate = Path(sys.argv[0]).name
    if not candidate or candidate in {"-m", "__main__.py"} or candidate.endswith(".py"):
        return "reddrop"
    return candidate


def _emit_bash_completion(prog_name: str) -> int:
    script = f"""_reddrop_complete() {{
  local cur
  local cmd
  COMPREPLY=()
  cmd="${{COMP_WORDS[0]}}"
  cur="${{COMP_WORDS[COMP_CWORD]}}"

  if [[ $COMP_CWORD -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "add search reply send completion" -- "$cur") )
    return
  fi

  if [[ "${{COMP_WORDS[1]}}" == "search" && $COMP_CWORD -eq 2 ]]; then
    local names
    names="$("$cmd" __complete_search_names "$cur" 2>/dev/null)"
    COMPREPLY=( $(compgen -W "$names" -- "$cur") )
  fi
}}
complete -F _reddrop_complete {prog_name} ./{prog_name}
"""
    print(script.rstrip("\n"))
    return 0


def _emit_zsh_completion(prog_name: str) -> int:
    script = f"""#compdef {prog_name}

_reddrop_complete() {{
  local cmd
  local curcontext="$curcontext" state line
  local -a commands search_names
  cmd="${{words[1]}}"

  commands=('add:Add or update a job' 'search:Search one or all jobs' 'reply:Generate replies from search artifact' 'send:Send unsent replies from search artifact' 'completion:Print completion script')

  if (( CURRENT == 2 )); then
    _describe 'reddrop command' commands
    return
  fi

  if [[ "${{words[2]}}" == "search" && CURRENT == 3 ]]; then
    search_names=("${{(@f)$("$cmd" __complete_search_names "${{words[CURRENT]}}" 2>/dev/null)}}")
    _describe 'job name' search_names
  fi
}}

compdef _reddrop_complete {prog_name} ./{prog_name}
"""
    print(script.rstrip("\n"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    store = _job_store()

    if args.command == "add":
        if args.job_type == "search":
            path_id = str(args.search_query_id or "").strip()
            if not path_id:
                return _error("--search-query-id is required for search jobs.")
            path = _search_query_store().get_query(path_id)
            if path is None:
                return _error(f"Search query not found: {path_id}")
            if path.status != "saved":
                return _error(
                    f"Search query must be saved before creating search jobs: {path.name}"
                )
            path_version = max(1, int(path.version or 1))
            if (
                args.search_query_version is not None
                and int(args.search_query_version) != path_version
            ):
                return _error(
                    f"Search query version mismatch: requested {int(args.search_query_version)} but saved version is {path_version}"
                )
            if not str(args.auth_id or "").strip():
                return _error("--auth-id is required for search jobs.")
            args.social = _resolve_social(path.social)
            args.search_query_id = path.id
            args.search_query_version = path_version
            args.topic = None
            args.search_plan = None
            args.source_job_id = None
            args.personas = None
            args.min_similarity_score = None
            args.replies_per_iteration = None
            if args.social != "linkedin":
                args.sort_by = None
                args.selector_wait_timeout_seconds = None
                args.detail_wait_timeout_seconds = None
                args.max_comment_expand_clicks = None
                args.detail_enrich_top_k = None
        if args.job_type == "reply":
            if not str(args.source_job_id or "").strip():
                return _error("--source-job-id is required for reply jobs.")
            if not args.personas:
                return _error("--personas is required for reply jobs.")
            source = store.get_job_by_id(str(args.source_job_id))
            if source is None:
                return _error(f"Source search job not found: {args.source_job_id}")
            if source.job_type != "search":
                return _error(f"Source job must be search type: {source.name}")
            if not str(source.auth_id or "").strip():
                return _error(
                    f"Source search job has no auth_id configured: {source.name}"
                )
            args.social = source.social
            args.auth_id = source.auth_id
            args.topic = None
            args.time_filter = None
            args.subreddit_limit = None
            args.threads_limit = None
            args.scroll_rounds = None
            args.selector_wait_timeout_seconds = None
            args.detail_wait_timeout_seconds = None
            args.max_comment_expand_clicks = None
            args.detail_enrich_top_k = None
            args.min_similarity_score = (
                float(args.min_similarity_score)
                if getattr(args, "min_similarity_score", None) is not None
                else DEFAULT_MIN_SIMILARITY_SCORE
            )
            args.replies_per_iteration = (
                int(args.replies_per_iteration)
                if getattr(args, "replies_per_iteration", None) is not None
                else DEFAULT_REPLIES_PER_ITERATION
            )
        return add_command.handle(
            args,
            store,
            _error,
            _emit,
            JOB_NAME_PATTERN,
        )
    if args.command == "search":
        deps = search_command.SearchDependencies(
            logger=logger,
            search_storage_factory=_search_storage,
            reddit_service_cls=RedditService,
            linkedin_service_cls=LinkedinService,
            conversation_model=Conversation,
            search_artifact_model=SearchArtifact,
            now_fn=search_command.utc_now,
            prepare_job=_prepare_search_job_for_runtime,
        )
        return search_command.handle(args, store, _emit, _error, deps)
    if args.command == "reply":
        deps = reply_command.ReplyDependencies(
            logger=logger,
            runs_dir=DEFAULT_RUNS_DIR,
            persona_store_cls=_persona_store,
            reply_generation_service_cls=ReplyGenerationService,
            search_storage_cls=SearchStorage,
            search_artifact_model=SearchArtifact,
            conversation_model=Conversation,
            now_fn=reply_command.utc_now,
        )
        return reply_command.handle(args, _emit, _error, deps)
    if args.command == "send":
        deps = send_command.SendDependencies(
            logger=logger,
            runs_dir=DEFAULT_RUNS_DIR,
            search_storage_cls=SearchStorage,
            reddit_service_cls=RedditService,
            linkedin_service_cls=LinkedinService,
            job_store_factory=_job_store,
            conversation_model=Conversation,
            search_artifact_model=SearchArtifact,
            now_fn=send_command.utc_now,
            prepare_job=_apply_auth_to_env,
        )
        return send_command.handle(args, _emit, _error, deps)
    if args.command == "completion":
        if args.shell == "bash":
            return _emit_bash_completion(_resolve_prog_name())
        if args.shell == "zsh":
            return _emit_zsh_completion(_resolve_prog_name())
        return _error(f"Unsupported shell: {args.shell}")
    if args.command == "__complete_search_names":
        return _complete_search_names(store, args.prefix)
    return _error("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
