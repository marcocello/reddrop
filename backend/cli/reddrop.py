from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..config.log import setup_logging
from ..models import Conversation, SearchArtifact
from ..services.job_config_store import JobConfigStore
from ..services.persona_store import PersonaStore
from ..services.reply_generation_service import ReplyGenerationService
from ..services.reddit_service import RedditService
from ..services.search_storage import SearchStorage
from .commands import add as add_command
from .commands import reply as reply_command
from .commands import search as search_command
from .commands import send as send_command

DEFAULT_HOME = Path(".reddrop")
DEFAULT_JOBS_CONFIG = DEFAULT_HOME / "jobs_config.json"
DEFAULT_RUNS_DIR = DEFAULT_HOME / "runs"
DEFAULT_TIME_FILTER = "week"
DEFAULT_SUBREDDIT_LIMIT = 5
DEFAULT_THREADS_LIMIT = 10
DEFAULT_REPLIES_PER_ITERATION = 3
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


def _search_storage() -> SearchStorage:
    return SearchStorage(DEFAULT_RUNS_DIR)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be an integer.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reddrop", description="Reddrop CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="Create or update the job configuration.")
    add.add_argument("--name", required=True)
    add.add_argument("--topic", required=True)
    add.add_argument("--time-filter", default=DEFAULT_TIME_FILTER, choices=["hour", "day", "week", "month", "year", "all"])
    add.add_argument("--subreddit-limit", type=int, default=DEFAULT_SUBREDDIT_LIMIT)
    add.add_argument("--threads-limit", type=int, default=DEFAULT_THREADS_LIMIT)
    add.add_argument("--replies-per-iteration", type=_positive_int, default=DEFAULT_REPLIES_PER_ITERATION)
    add.add_argument("--max-runtime-minutes", type=_positive_int, default=DEFAULT_MAX_RUNTIME_MINUTES)
    add.add_argument("--personas", nargs="*", default=[])

    search = subparsers.add_parser("search", help="Search Reddit with the saved job configuration.")
    search.add_argument("name", nargs="?")

    reply = subparsers.add_parser("reply", help="Generate replies from a search artifact.")
    reply.add_argument("search_file")
    reply.add_argument("--replies", type=_positive_int, default=3)
    reply.add_argument("--personas", required=True)

    send = subparsers.add_parser("send", help="Send unsent replies from a search artifact.")
    send.add_argument("search_file")

    completion = subparsers.add_parser("completion", help="Print shell completion script.")
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
            conversation_model=Conversation,
            search_artifact_model=SearchArtifact,
            now_fn=search_command.utc_now,
        )
        return search_command.handle(args, store, _emit, _error, deps)
    if args.command == "reply":
        deps = reply_command.ReplyDependencies(
            logger=logger,
            runs_dir=DEFAULT_RUNS_DIR,
            persona_store_cls=PersonaStore,
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
            conversation_model=Conversation,
            search_artifact_model=SearchArtifact,
            now_fn=send_command.utc_now,
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
