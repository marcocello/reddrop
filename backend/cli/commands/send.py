from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _never_stop() -> bool:
    return False


@dataclass(frozen=True)
class SendDependencies:
    logger: Any
    runs_dir: Path
    search_storage_cls: Any
    reddit_service_cls: Any
    conversation_model: Any
    search_artifact_model: Any
    now_fn: Any
    should_stop: Callable[[], bool] = _never_stop


def _resolve_search_file(search_file: str, runs_dir: Path) -> Path | None:
    provided = Path(search_file)
    if provided.exists() and provided.is_file():
        return provided

    candidate = runs_dir / search_file
    if candidate.exists() and candidate.is_file():
        return candidate

    # Compatibility input: reply_<prefix>.json now maps to search_<prefix>.json.
    if search_file.startswith("reply_"):
        mapped = search_file.replace("reply_", "search_", 1)
        mapped_candidate = runs_dir / mapped
        if mapped_candidate.exists() and mapped_candidate.is_file():
            return mapped_candidate
    return None


def handle(args, emit, error, deps: SendDependencies) -> int:
    if deps.should_stop():
        return error("Run stopped by user request.")

    search_path = _resolve_search_file(args.search_file, deps.runs_dir)
    if search_path is None:
        return error(f"Search file not found: {args.search_file}")

    storage = deps.search_storage_cls(deps.runs_dir)
    artifact = storage.get_search_by_path(search_path)
    if artifact is None:
        return error(f"Invalid search artifact: {search_path.name}")

    deps.logger.info("Sending replies from file %s.", search_path.name)
    reddit_service = deps.reddit_service_cls()

    updated_conversations: list[Any] = []
    sent_count = 0
    skipped_count = 0
    total_reply_candidates = 0

    for item in artifact.conversations:
        if deps.should_stop():
            return error("Run stopped by user request.")
        updated = deps.conversation_model.model_validate(item.model_dump(mode="python"))
        if not updated.reply.strip():
            updated_conversations.append(updated)
            continue

        total_reply_candidates += 1
        if updated.user_has_commented:
            skipped_count += 1
            updated_conversations.append(updated)
            continue

        if reddit_service.post_comment(thread_id=updated.thread_id, comment_text=updated.reply):
            updated.user_has_commented = True
            sent_count += 1

        updated_conversations.append(updated)

    updated_artifact = deps.search_artifact_model(
        id=artifact.id,
        name=artifact.name,
        topic=artifact.topic,
        created_at=artifact.created_at,
        updated_at=deps.now_fn(),
        conversations=updated_conversations,
    )
    storage.save_search_at_path(search_path, updated_artifact)

    deps.logger.info(
        "Send completed for %s: sent=%s, already_commented=%s, candidates=%s, total=%s.",
        search_path.name,
        sent_count,
        skipped_count,
        total_reply_candidates,
        len(updated_conversations),
    )
    emit(
        {
            "search_file": search_path.name,
            "reply_file": search_path.name,
            "sent": sent_count,
            "already_commented": skipped_count,
            "candidates": total_reply_candidates,
            "total": len(updated_conversations),
        }
    )
    return 0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
