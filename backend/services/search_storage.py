from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..models import Conversation, SearchArtifact


class SearchStorage:
    def __init__(self, data_dir: str | Path = "data/runs") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_search(self, artifact: SearchArtifact) -> Path:
        path = self._path_for_job(artifact.id)
        with self._exclusive_lock(path):
            existing = self._read_artifact(path)
            merged = self._merge_with_existing(existing, artifact)
            self._write_atomic(path, merged)
        return path

    def get_search(self, job_id: str) -> SearchArtifact | None:
        return self._read_artifact(self._path_for_job(job_id))

    def get_search_by_path(self, path: Path) -> SearchArtifact | None:
        return self._read_artifact(path)

    def save_search_at_path(self, path: Path, artifact: SearchArtifact) -> Path:
        with self._exclusive_lock(path):
            existing = self._read_artifact(path)
            merged = self._merge_with_existing(existing, artifact)
            self._write_atomic(path, merged)
        return path

    def update_search_by_path(
        self,
        path: Path,
        updater: Callable[[SearchArtifact], SearchArtifact],
    ) -> SearchArtifact | None:
        with self._exclusive_lock(path):
            current = self._read_artifact(path)
            if current is None:
                return None
            updated = updater(current)
            normalized = SearchArtifact.model_validate(updated.model_dump(mode="python"))
            self._write_atomic(path, normalized)
            return normalized

    def path_for_job(self, job_id: str) -> Path:
        return self._path_for_job(job_id)

    @staticmethod
    def _job_prefix(job_id: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "", job_id.strip().lower())
        return cleaned[:8] or "job"

    def _path_for_job(self, job_id: str) -> Path:
        return self.data_dir / f"search_{self._job_prefix(job_id)}.json"

    @staticmethod
    def _merge_conversations(existing: list[Conversation], incoming: list[Conversation]) -> list[Conversation]:
        merged: list[Conversation] = []
        index_by_key: dict[tuple[str, str], int] = {}

        for item in existing:
            key = (item.thread_id, item.subreddit)
            if key in index_by_key:
                merged[index_by_key[key]] = item
            else:
                index_by_key[key] = len(merged)
                merged.append(item)

        for item in incoming:
            key = (item.thread_id, item.subreddit)
            if key in index_by_key:
                previous = merged[index_by_key[key]]
                replacement = Conversation.model_validate(item.model_dump(mode="python"))
                if not replacement.reply.strip() and previous.reply.strip():
                    replacement.reply = previous.reply
                if previous.user_has_commented and not replacement.user_has_commented:
                    replacement.user_has_commented = True
                merged[index_by_key[key]] = replacement
            else:
                index_by_key[key] = len(merged)
                merged.append(item)

        return merged

    def _merge_with_existing(self, existing: SearchArtifact | None, artifact: SearchArtifact) -> SearchArtifact:
        if existing is None:
            return artifact

        merged_created_at = existing.created_at
        merged_conversations = self._merge_conversations(existing.conversations, artifact.conversations)
        merged_updated_at = artifact.updated_at.astimezone(timezone.utc)

        return SearchArtifact(
            id=artifact.id,
            name=artifact.name,
            topic=artifact.topic,
            created_at=merged_created_at,
            updated_at=merged_updated_at,
            conversations=merged_conversations,
        )

    @staticmethod
    def _read_artifact(path: Path) -> SearchArtifact | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload = SearchStorage._normalize_legacy_payload(payload)
            return SearchArtifact.model_validate(payload)
        except Exception:
            return None

    @staticmethod
    def _normalize_legacy_payload(payload: dict) -> dict:
        if not isinstance(payload, dict):
            return payload
        if "updated_at" in payload:
            return SearchStorage._normalize_conversation_reply_field(payload)
        created_raw = payload.get("created_at")
        if isinstance(created_raw, str):
            payload["updated_at"] = created_raw
        else:
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload.pop("run_id", None)
        if not payload.get("name"):
            source_id = str(payload.get("id", payload.get("job_id", ""))).strip()
            payload["name"] = f"job-{source_id[:8]}" if source_id else "job"
        return SearchStorage._normalize_conversation_reply_field(payload)

    @staticmethod
    def _normalize_conversation_reply_field(payload: dict) -> dict:
        conversations = payload.get("conversations")
        if not isinstance(conversations, list):
            return payload
        normalized: list[dict] = []
        for item in conversations:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            record = dict(item)
            if "reply" not in record:
                record["reply"] = ""
            normalized.append(record)
        payload["conversations"] = normalized
        return payload

    @staticmethod
    def _lock_path(path: Path) -> Path:
        return path.with_suffix(f"{path.suffix}.lock")

    @contextmanager
    def _exclusive_lock(self, path: Path):
        lock_path = self._lock_path(path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_atomic(path: Path, artifact: SearchArtifact) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
            json.dump(artifact.model_dump(mode="json"), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
