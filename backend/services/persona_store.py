from __future__ import annotations

import json
from pathlib import Path

from ..models import PersonaProfile


class PersonaStore:
    def __init__(self, candidates: list[str | Path] | None = None) -> None:
        default_candidates: list[str | Path] = [".reddrop/personas.json"]
        self._candidates = [Path(item) for item in (candidates or default_candidates)]

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
            raise FileNotFoundError("Personas file not found: .reddrop/personas.json")
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
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"personas": [item.model_dump(mode="json") for item in personas]}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def _personas_from_path(self, path: Path) -> list[PersonaProfile]:
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
