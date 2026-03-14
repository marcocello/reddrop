from __future__ import annotations

import json
from pathlib import Path
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

    def resolve_file(self) -> Path | None:
        for path in self._platform_files.values():
            if path.exists() and path.is_file():
                return self._base_dir
        for candidate in self._legacy_candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _load_entries(path: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        entries = payload.get("auths", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            return []
        return [item for item in entries if isinstance(item, dict)]

    def list_auths(self) -> list[AuthProfile]:
        normalized_by_id: dict[str, AuthProfile] = {}

        for social in ("reddit", "linkedin"):
            path = self._platform_files[social]
            if not path.exists() or not path.is_file():
                continue
            entries = self._load_entries(path)
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
        for candidate in self._legacy_candidates:
            if not candidate.exists() or not candidate.is_file():
                continue
            for item in self._load_entries(candidate):
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
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {"auths": [item.model_dump(mode="json", exclude={"status"}) for item in auths]}
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return target

        grouped: dict[str, list[dict[str, Any]]] = {"reddit": [], "linkedin": []}
        for item in auths:
            social = "linkedin" if item.social == "linkedin" else "reddit"
            grouped[social].append(item.model_dump(mode="json", exclude={"status"}))

        for social, path in self._platform_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"auths": grouped[social]}
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self._base_dir
