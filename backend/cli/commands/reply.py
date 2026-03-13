from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_MIN_REPLY_SIMILARITY = 0.35


def _never_stop() -> bool:
    return False


@dataclass(frozen=True)
class ReplyDependencies:
    logger: Any
    runs_dir: Path
    persona_store_cls: Any
    reply_generation_service_cls: Any
    search_storage_cls: Any
    search_artifact_model: Any
    conversation_model: Any
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


def _similarity_value(conversation: Any) -> float:
    try:
        return float(getattr(conversation, "semantic_similarity", 0.0) or 0.0)
    except Exception:
        return 0.0


def _resolve_min_similarity(raw: object) -> float:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except Exception:
        return DEFAULT_MIN_REPLY_SIMILARITY
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def handle(args, emit, error, deps: ReplyDependencies) -> int:
    if deps.should_stop():
        return error("Run stopped by user request.")

    search_path = _resolve_search_file(args.search_file, deps.runs_dir)
    if search_path is None:
        return error(f"Search file not found: {args.search_file}")

    storage = deps.search_storage_cls(deps.runs_dir)
    search_artifact = storage.get_search_by_path(search_path)
    if search_artifact is None:
        return error(f"Invalid search artifact: {search_path.name}")

    replied_keys = {
        (item.thread_id, item.subreddit)
        for item in search_artifact.conversations
        if getattr(item, "reply", "").strip()
    }

    min_similarity = _resolve_min_similarity(getattr(args, "min_similarity", DEFAULT_MIN_REPLY_SIMILARITY))

    eligible = [
        item
        for item in search_artifact.conversations
        if (item.thread_id, item.subreddit) not in replied_keys and _similarity_value(item) >= min_similarity
    ]
    ranked = sorted(eligible, key=_similarity_value, reverse=True)
    selected = ranked[: args.replies]

    persona_name = args.personas.strip()
    if not persona_name:
        return error("Persona name is required.")
    try:
        persona = deps.persona_store_cls().get_persona(persona_name)
    except FileNotFoundError as exc:
        return error(str(exc))
    except Exception as exc:
        return error(f"Failed to load personas file: {exc}")
    if persona is None:
        return error(f"Persona not found: {args.personas}")

    deps.logger.info(
        "Generating replies for search file %s (selected=%s/%s eligible, already_replied=%s, min_similarity=%.2f, total=%s).",
        search_path.name,
        len(selected),
        len(eligible),
        len(replied_keys),
        min_similarity,
        len(search_artifact.conversations),
    )

    generator = deps.reply_generation_service_cls()
    generated_map: dict[tuple[str, str], str] = {}
    for conversation in selected:
        if deps.should_stop():
            return error("Run stopped by user request.")
        generated_map[(conversation.thread_id, conversation.subreddit)] = generator.generate_reply(
            topic=search_artifact.topic,
            conversation=conversation,
            persona=persona,
        )

    updated_conversations = []
    for item in search_artifact.conversations:
        if deps.should_stop():
            return error("Run stopped by user request.")
        key = (item.thread_id, item.subreddit)
        if key in generated_map:
            payload = item.model_dump(mode="python")
            payload["reply"] = generated_map[key]
            updated_conversations.append(
                deps.conversation_model(**payload)
            )
        else:
            updated_conversations.append(item)

    updated_artifact = deps.search_artifact_model(
        id=search_artifact.id,
        name=search_artifact.name,
        topic=search_artifact.topic,
        created_at=search_artifact.created_at,
        updated_at=deps.now_fn(),
        conversations=updated_conversations,
    )
    storage.save_search_at_path(search_path, updated_artifact)

    total_with_replies = sum(1 for item in updated_conversations if item.reply.strip())
    emit(
        {
            "search_file": search_path.name,
            "reply_file": search_path.name,
            "generated": len(generated_map),
            "total_with_replies": total_with_replies,
            "min_similarity": min_similarity,
        }
    )
    return 0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
