from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from ..models import AuthProfile

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


class AuthStore:
    def __init__(
        self,
        base_dir: str | Path | None = None,
        legacy_candidates: list[str | Path] | None = None,
    ) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else (_BACKEND_ROOT / ".reddrop/auth")
        self._platform_files: dict[str, Path] = {
            "reddit": self._base_dir / "reddit" / "auths.json",
            "linkedin": self._base_dir / "linkedin" / "auths.json",
        }
        default_legacy: list[str | Path] = [_BACKEND_ROOT / ".reddrop/auths.json"]
        self._legacy_candidates = [Path(item) for item in (legacy_candidates or default_legacy)]
        self._lock = Lock()

    def resolve_file(self) -> Path | None:
        for path in self._platform_files.values():
            if path.exists() and path.is_file():
                return self._base_dir
        for candidate in self._legacy_candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _load_entries_from_text(text: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(text)
        except Exception:
            return []
        entries = payload.get("auths", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            return []
        return [item for item in entries if isinstance(item, dict)]

    @classmethod
    def _load_entries_unlocked(cls, path: Path) -> list[dict[str, Any]]:
        try:
            return cls._load_entries_from_text(path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def list_auths(self) -> list[AuthProfile]:
        normalized_by_id: dict[str, AuthProfile] = {}
        with self._lock:
            with self._exclusive_lock(self._platform_lock_path()):
                for social in ("reddit", "linkedin"):
                    path = self._platform_files[social]
                    if not path.exists() or not path.is_file():
                        continue
                    entries = self._load_entries_unlocked(path)
                    for item in entries:
                        payload = dict(item)
                        payload["social"] = social
                        try:
                            parsed = AuthProfile.model_validate(payload)
                        except Exception:
                            continue
                        key = parsed.id.strip().lower()
                        if key and key not in normalized_by_id:
                            normalized_by_id[key] = parsed

        if normalized_by_id:
            return list(normalized_by_id.values())

        normalized: list[AuthProfile] = []
        with self._lock:
            for candidate in self._legacy_candidates:
                if not candidate.exists() or not candidate.is_file():
                    continue
                with self._exclusive_lock(self._path_lock_path(candidate)):
                    entries = self._load_entries_unlocked(candidate)
                for item in entries:
                    try:
                        parsed = AuthProfile.model_validate(item)
                    except Exception:
                        continue
                    normalized.append(parsed)
        return normalized

    def get_auth(self, auth_id: str) -> AuthProfile | None:
        normalized = auth_id.strip()
        if not normalized:
            return None
        for auth in self.list_auths():
            if auth.id == normalized:
                return auth
        return None

    def save_auths(self, auths: list[AuthProfile], destination: str | Path | None = None) -> Path:
        if destination is not None:
            target = Path(destination)
            payload = {"auths": [item.model_dump(mode="json") for item in auths]}
            with self._lock:
                with self._exclusive_lock(self._path_lock_path(target)):
                    self._write_atomic(target, payload)
            return target

        grouped: dict[str, list[dict[str, Any]]] = {"reddit": [], "linkedin": []}
        for item in auths:
            social = "linkedin" if item.social == "linkedin" else "reddit"
            grouped[social].append(item.model_dump(mode="json"))

        with self._lock:
            with self._exclusive_lock(self._platform_lock_path()):
                for social, path in self._platform_files.items():
                    payload = {"auths": grouped[social]}
                    self._write_atomic(path, payload)
        return self._base_dir

    def _platform_lock_path(self) -> Path:
        return self._base_dir / ".auths.lock"

    @staticmethod
    def _path_lock_path(path: Path) -> Path:
        return path.with_suffix(f"{path.suffix}.lock")

    @contextmanager
    def _exclusive_lock(self, lock_path: Path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
