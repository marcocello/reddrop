from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_URL, UUID, uuid5

from ..models import Conversation, SearchArtifact


class SearchStorage:
    def __init__(self, data_dir: str | Path = "data/runs") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.conversation_path = self.data_dir / "conversation.json"
        self._cleanup_legacy_lock_files()

    def save_search(self, artifact: SearchArtifact) -> Path:
        path = self._path_for_job(artifact.id)
        with self._exclusive_lock(path):
            existing = self._read_artifact(path)
            merged = self._merge_with_existing(existing, artifact)
            self._write_atomic(path, merged)
            self._refresh_conversation_file_locked()
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
            self._refresh_conversation_file_locked()
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
            normalized = SearchArtifact(
                id=normalized.id,
                name=normalized.name,
                topic=normalized.topic,
                created_at=normalized.created_at,
                updated_at=normalized.updated_at.astimezone(timezone.utc),
                conversations=self._merge_conversations([], normalized.conversations),
            )
            self._write_atomic(path, normalized)
            self._refresh_conversation_file_locked()
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
        existing_by_key: dict[tuple[str, str], Conversation] = {}
        existing_by_url: dict[str, Conversation] = {}
        existing_by_linkedin_equivalence: dict[str, Conversation] = {}
        for item in existing:
            normalized_existing = Conversation.model_validate(item.model_dump(mode="python"))
            url_key = SearchStorage._conversation_url_key(normalized_existing.url)
            linkedin_equivalence_key = SearchStorage._linkedin_equivalence_key(
                subreddit=normalized_existing.subreddit,
                url=normalized_existing.url,
                selftext=normalized_existing.selftext,
                title=normalized_existing.title,
            )
            if url_key:
                normalized_existing.id = SearchStorage._conversation_id_for_url_key(url_key)
            existing_by_key[(normalized_existing.thread_id, normalized_existing.subreddit)] = normalized_existing
            if url_key:
                existing_by_url[url_key] = normalized_existing
            if linkedin_equivalence_key:
                existing_by_linkedin_equivalence[linkedin_equivalence_key] = normalized_existing

        merged: list[Conversation] = []
        index_by_key: dict[tuple[str, str], int] = {}
        index_by_url: dict[str, int] = {}
        index_by_linkedin_equivalence: dict[str, int] = {}

        for item in incoming:
            normalized = Conversation.model_validate(item.model_dump(mode="python"))
            thread_key = (normalized.thread_id, normalized.subreddit)
            url_key = SearchStorage._conversation_url_key(normalized.url)
            linkedin_equivalence_key = SearchStorage._linkedin_equivalence_key(
                subreddit=normalized.subreddit,
                url=normalized.url,
                selftext=normalized.selftext,
                title=normalized.title,
            )
            if url_key:
                normalized.id = SearchStorage._conversation_id_for_url_key(url_key)

            previous = existing_by_key.get(thread_key)
            if previous is None and url_key:
                previous = existing_by_url.get(url_key)
            if previous is None and linkedin_equivalence_key:
                previous = existing_by_linkedin_equivalence.get(linkedin_equivalence_key)
            if previous is not None:
                if not url_key:
                    normalized.id = previous.id
                if not normalized.reply.strip() and previous.reply.strip():
                    normalized.reply = previous.reply
                if previous.user_has_commented and not normalized.user_has_commented:
                    normalized.user_has_commented = True

            target_index: int | None = None
            if thread_key in index_by_key:
                target_index = index_by_key[thread_key]
            elif url_key and url_key in index_by_url:
                target_index = index_by_url[url_key]
            elif linkedin_equivalence_key and linkedin_equivalence_key in index_by_linkedin_equivalence:
                target_index = index_by_linkedin_equivalence[linkedin_equivalence_key]

            if target_index is None:
                target_index = len(merged)
                merged.append(normalized)
            else:
                prior = merged[target_index]
                if not url_key:
                    normalized.id = prior.id
                if not normalized.reply.strip() and prior.reply.strip():
                    normalized.reply = prior.reply
                if prior.user_has_commented and not normalized.user_has_commented:
                    normalized.user_has_commented = True
                merged[target_index] = normalized

            active = merged[target_index]
            index_by_key[(active.thread_id, active.subreddit)] = target_index
            active_url_key = SearchStorage._conversation_url_key(active.url)
            if active_url_key:
                index_by_url[active_url_key] = target_index
            active_linkedin_equivalence_key = SearchStorage._linkedin_equivalence_key(
                subreddit=active.subreddit,
                url=active.url,
                selftext=active.selftext,
                title=active.title,
            )
            if active_linkedin_equivalence_key:
                index_by_linkedin_equivalence[active_linkedin_equivalence_key] = target_index

        return merged

    @staticmethod
    def _conversation_id_for_url_key(url_key: str) -> UUID:
        return uuid5(NAMESPACE_URL, url_key)

    @staticmethod
    def _conversation_url_key(raw_url: object) -> str:
        value = str(raw_url or "").strip()
        if not value:
            return ""
        try:
            parts = urlsplit(value)
        except Exception:
            return value.rstrip("/")

        if parts.scheme and parts.netloc:
            path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")
            return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))
        return value.rstrip("/")

    @staticmethod
    def _linkedin_equivalence_key(*, subreddit: object, url: object, selftext: object, title: object) -> str:
        if str(subreddit or "").strip().lower() != "linkedin":
            return ""

        canonical_url = SearchStorage._conversation_url_key(url)
        if not re.search(
            r"/feed/update/urn:li:(activity|ugcPost|share):\d+/?$",
            canonical_url,
            flags=re.IGNORECASE,
        ):
            return ""

        text_source = " ".join(str(selftext or "").split()).strip().lower()
        if not text_source:
            text_source = " ".join(str(title or "").split()).strip().lower()
        if len(text_source) < 120:
            return ""

        digest = hashlib.sha1(text_source.encode("utf-8")).hexdigest()[:24]
        return f"linkedin:{digest}"

    def _merge_with_existing(self, existing: SearchArtifact | None, artifact: SearchArtifact) -> SearchArtifact:
        if existing is None:
            return SearchArtifact(
                id=artifact.id,
                name=artifact.name,
                topic=artifact.topic,
                created_at=artifact.created_at,
                updated_at=artifact.updated_at.astimezone(timezone.utc),
                conversations=self._merge_conversations([], artifact.conversations),
            )

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
        if "topic" not in payload:
            payload["topic"] = ""
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
        return path.parent / ".search_storage.lock"

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

    def refresh_conversation_file(self) -> None:
        with self._exclusive_lock(self.conversation_path):
            self._refresh_conversation_file_locked()

    def _refresh_conversation_file_locked(self) -> None:
        payload: dict[str, object] = {
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "artifacts": [],
            "conversations": [],
        }
        artifacts: list[dict[str, object]] = []
        conversations: list[dict[str, object]] = []

        for path in sorted(self.data_dir.glob("search_*.json"), key=lambda item: item.name):
            artifact = self._read_artifact(path)
            if artifact is None:
                continue
            artifact_payload = artifact.model_dump(mode="json")
            artifact_payload.pop("topic", None)
            artifact_payload["search_artifact"] = path.name
            artifacts.append(artifact_payload)

            for conversation in artifact.conversations:
                row = conversation.model_dump(mode="json")
                row["job_id"] = artifact.id
                row["job_name"] = artifact.name
                row["search_artifact"] = path.name
                row["search_updated_at"] = artifact.updated_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                conversations.append(row)

        payload["artifacts"] = artifacts
        payload["conversations"] = conversations
        self._write_json_atomic(self.conversation_path, payload)

    def _cleanup_legacy_lock_files(self) -> None:
        for legacy in self.data_dir.glob("search_*.json.lock"):
            try:
                legacy.unlink()
            except Exception:
                continue

    @staticmethod
    def _write_atomic(path: Path, artifact: SearchArtifact) -> None:
        payload = artifact.model_dump(mode="json")
        payload.pop("topic", None)
        SearchStorage._write_json_atomic(path, payload)

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
