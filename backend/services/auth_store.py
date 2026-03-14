from __future__ import annotations

import json
from pathlib import Path

from ..models import AuthProfile


class AuthStore:
    def __init__(self, candidates: list[str | Path] | None = None) -> None:
        default_candidates: list[str | Path] = [".reddrop/auths.json"]
        self._candidates = [Path(item) for item in (candidates or default_candidates)]

    def resolve_file(self) -> Path | None:
        for candidate in self._candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def list_auths(self) -> list[AuthProfile]:
        path = self.resolve_file()
        if path is None:
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        entries = payload.get("auths", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            return []
        normalized: list[AuthProfile] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            try:
                normalized.append(AuthProfile.model_validate(item))
            except Exception:
                continue
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
        target = Path(destination) if destination is not None else (self.resolve_file() or self._candidates[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"auths": [item.model_dump(mode="json") for item in auths]}
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target
