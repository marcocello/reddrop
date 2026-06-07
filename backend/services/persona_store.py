from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from ..models import PersonaProfile

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


class PersonaStore:
    def __init__(self, candidates: list[str | Path] | None = None) -> None:
        default_candidates: list[str | Path] = [_BACKEND_ROOT / ".reddrop/personas.json"]
        self._candidates = [Path(item) for item in (candidates or default_candidates)]
        self._lock = Lock()

    def resolve_file(self) -> Path | None:
        for candidate in self._candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def get_persona(self, name: str) -> PersonaProfile | None:
        target = name.strip().lower()
        if not target:
            return None

        path = self.resolve_file()
        if path is None:
            raise FileNotFoundError("Personas file not found: backend/.reddrop/personas.json")
        for persona in self._personas_from_path(path):
            if persona.name.strip().lower() == target:
                return persona
        return None

    def list_personas(self) -> list[PersonaProfile]:
        path = self.resolve_file()
        if path is None:
            return []
        return self._personas_from_path(path)

    def save_personas(self, personas: list[PersonaProfile], destination: str | Path | None = None) -> Path:
        target = Path(destination) if destination is not None else (self.resolve_file() or self._candidates[0])
        payload = {"personas": [item.model_dump(mode="json") for item in personas]}
        with self._lock:
            with self._exclusive_lock(self._lock_path(target)):
                self._write_atomic(target, payload)
        return target

    def _personas_from_path(self, path: Path) -> list[PersonaProfile]:
        with self._exclusive_lock(self._lock_path(path)):
            payload = json.loads(path.read_text(encoding="utf-8"))
        entries = self._extract_persona_entries(payload)
        return [PersonaProfile.model_validate(entry) for entry in entries]

    @staticmethod
    def _extract_persona_entries(payload: dict) -> list[dict]:
        if not isinstance(payload, dict):
            return []

        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            if key.strip().lower() == "personas" and isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _lock_path(path: Path) -> Path:
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
    def _write_atomic(path: Path, payload: dict) -> None:
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
