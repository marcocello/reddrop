from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from ..models import SearchQueriesConfig, SearchQuery

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


class SearchQueryStore:
    def __init__(self, config_path: str | Path = _BACKEND_ROOT / ".reddrop/search_queries.json") -> None:
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._config = self._load()

    def list_queries(self) -> list[SearchQuery]:
        with self._lock:
            return [SearchQuery.model_validate(item.model_dump(mode="python")) for item in self._config.search_queries]

    def get_query(self, query_ref: str) -> SearchQuery | None:
        normalized = query_ref.strip()
        if not normalized:
            return None
        with self._lock:
            for item in self._config.search_queries:
                if item.id == normalized or item.name == normalized:
                    return SearchQuery.model_validate(item.model_dump(mode="python"))
        return None

    def upsert_query(self, query: SearchQuery) -> SearchQuery:
        with self._lock:
            target_index: int | None = None
            for index, item in enumerate(self._config.search_queries):
                if item.id == query.id:
                    target_index = index
                    break
            if target_index is None:
                self._config.search_queries.append(query)
            else:
                self._config.search_queries[target_index] = query
            self._persist()
            return SearchQuery.model_validate(query.model_dump(mode="python"))

    def delete_query(self, query_id: str) -> bool:
        normalized = query_id.strip()
        if not normalized:
            return False
        with self._lock:
            original_len = len(self._config.search_queries)
            self._config.search_queries = [item for item in self._config.search_queries if item.id != normalized]
            deleted = len(self._config.search_queries) != original_len
            if deleted:
                self._persist()
            return deleted

    def _load(self) -> SearchQueriesConfig:
        if not self.config_path.exists():
            return SearchQueriesConfig()
        try:
            with self._exclusive_lock():
                payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "search_queries" not in payload and "search_paths" in payload:
                payload["search_queries"] = payload.get("search_paths")
                payload.pop("search_paths", None)
            return SearchQueriesConfig.model_validate(payload)
        except Exception:
            return SearchQueriesConfig()

    def _persist(self) -> None:
        payload = self._config.model_dump(mode="json", exclude_none=True)
        with self._exclusive_lock():
            self._write_atomic(payload)

    def _lock_path(self) -> Path:
        return self.config_path.with_suffix(f"{self.config_path.suffix}.lock")

    @contextmanager
    def _exclusive_lock(self):
        lock_path = self._lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write_atomic(self, payload: dict) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.config_path.parent),
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, self.config_path)
