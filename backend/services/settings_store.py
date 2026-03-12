from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class SettingsStore:
    REDDIT_KEYS = {
        "client_id": "REDDIT_CLIENT_ID",
        "client_secret": "REDDIT_CLIENT_SECRET",
        "user_agent": "REDDIT_USER_AGENT",
        "username": "REDDIT_USERNAME",
        "password": "REDDIT_PASSWORD",
    }
    OPENROUTER_KEYS = {
        "api_key": "OPENROUTER_API_KEY",
        "base_url": "OPENROUTER_BASE_URL",
        "model": "OPENROUTER_MODEL",
        "http_referer": "OPENROUTER_HTTP_REFERER",
        "x_title": "OPENROUTER_X_TITLE",
        "timeout_seconds": "OPENROUTER_TIMEOUT_SECONDS",
    }
    DEFAULTS = {
        "REDDIT_USER_AGENT": "Reddrop:v1.0",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "OPENROUTER_MODEL": "x-ai/grok-4.1-fast",
        "OPENROUTER_X_TITLE": "reddrop",
        "OPENROUTER_TIMEOUT_SECONDS": "20",
    }

    def __init__(self, settings_path: str | Path = ".reddrop/settings.yaml") -> None:
        self.settings_path = Path(settings_path)

    def load(self) -> dict[str, Any]:
        payload = self._load_file_payload()
        self._sync_env_from_payload(payload, overwrite=False)
        return {
            "reddit": self._load_section(self.REDDIT_KEYS, payload.get("reddit")),
            "openrouter": self._load_section(self.OPENROUTER_KEYS, payload.get("openrouter")),
        }

    def save(self, *, reddit: dict[str, str], openrouter: dict[str, str]) -> dict[str, Any]:
        payload = {
            "reddit": self._normalize_section(self.REDDIT_KEYS, reddit),
            "openrouter": self._normalize_section(self.OPENROUTER_KEYS, openrouter),
        }
        self._persist(payload)
        self._sync_env_from_payload(payload, overwrite=True)
        return self.load()

    def _load_section(self, mapping: dict[str, str], section_values: object) -> dict[str, str]:
        source = section_values if isinstance(section_values, dict) else {}
        payload: dict[str, str] = {}
        for field, env_key in mapping.items():
            default = self.DEFAULTS.get(env_key, "")
            env_value = (os.getenv(env_key) or "").strip()
            if env_value:
                payload[field] = env_value
                continue
            file_value = str(source.get(field, "")).strip()
            payload[field] = file_value if file_value else default
        return payload

    def _load_file_payload(self) -> dict[str, dict[str, str]]:
        if not self.settings_path.exists():
            return {"reddit": {}, "openrouter": {}}
        try:
            loaded = yaml.safe_load(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return {"reddit": {}, "openrouter": {}}
        if not isinstance(loaded, dict):
            return {"reddit": {}, "openrouter": {}}

        reddit = loaded.get("reddit")
        openrouter = loaded.get("openrouter")
        return {
            "reddit": reddit if isinstance(reddit, dict) else {},
            "openrouter": openrouter if isinstance(openrouter, dict) else {},
        }

    def _sync_env_from_payload(self, payload: dict[str, dict[str, str]], *, overwrite: bool) -> None:
        if not isinstance(payload, dict):
            return
        section_pairs = (
            (self.REDDIT_KEYS, payload.get("reddit", {})),
            (self.OPENROUTER_KEYS, payload.get("openrouter", {})),
        )
        for mapping, section in section_pairs:
            if not isinstance(section, dict):
                continue
            for field, env_key in mapping.items():
                if not overwrite and (os.getenv(env_key) or "").strip():
                    continue
                os.environ[env_key] = str(section.get(field, "")).strip()

    @staticmethod
    def _normalize_section(mapping: dict[str, str], values: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for field in mapping:
            normalized[field] = str(values.get(field, "")).strip()
        return normalized

    def _persist(self, payload: dict[str, dict[str, str]]) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        dumped = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False).rstrip()
        self.settings_path.write_text(dumped + "\n", encoding="utf-8")
