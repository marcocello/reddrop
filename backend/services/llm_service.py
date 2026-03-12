from __future__ import annotations

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI

from ..config.log import setup_logging
from .settings_store import SettingsStore
logger = setup_logging(__name__)


class LLMService:
    DEFAULT_REQUEST_TIMEOUT_SECONDS = 20.0

    def __init__(self) -> None:
        self.client: AsyncOpenAI | None = None
        self.client_type = "demo"
        self.model_name: str | None = None
        self._client_loop_id: int | None = None
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self._initialize_client()

    @staticmethod
    def _current_loop_id() -> int | None:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return None

    def _initialize_client(self) -> None:
        try:
            SettingsStore().load()
            openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
            if openrouter_key:
                openrouter_base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()
                openrouter_model = (os.getenv("OPENROUTER_MODEL") or "x-ai/grok-4.1-fast").strip()
                kwargs = {
                    "base_url": openrouter_base_url,
                    "api_key": openrouter_key,
                    # Keep requests interruptible and avoid long hidden retry waits.
                    "max_retries": 0,
                    "timeout": self._request_timeout_seconds(),
                }
                headers = self._openrouter_headers()
                if headers:
                    kwargs["default_headers"] = headers
                self.client = AsyncOpenAI(**kwargs)
                self.client_type = "openrouter"
                self.model_name = openrouter_model
                self._client_loop_id = self._current_loop_id()
                return

            self.client = None
            self.client_type = "demo"
            self.model_name = None
            self._client_loop_id = None
            logger.warning("OpenRouter key not found. LLM service is running in demo fallback mode.")
        except Exception:
            self.client = None
            self.client_type = "demo"
            self.model_name = None
            self._client_loop_id = None
            logger.exception("Failed to initialize OpenRouter client. Falling back to demo mode.")

    @staticmethod
    def _openrouter_headers() -> dict[str, str]:
        headers: dict[str, str] = {}
        referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
        title = (os.getenv("OPENROUTER_X_TITLE") or "reddrop").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers

    def _request_timeout_seconds(self) -> float:
        raw = (os.getenv("OPENROUTER_TIMEOUT_SECONDS") or "").strip()
        if not raw:
            return self.DEFAULT_REQUEST_TIMEOUT_SECONDS
        try:
            value = float(raw)
            return value if value > 0 else self.DEFAULT_REQUEST_TIMEOUT_SECONDS
        except Exception:
            return self.DEFAULT_REQUEST_TIMEOUT_SECONDS

    def _ensure_client_for_current_loop(self) -> None:
        if not self.client or self.client_type != "openrouter":
            return
        loop_id = self._current_loop_id()
        if loop_id is None:
            return
        if self._client_loop_id is None:
            self._client_loop_id = loop_id
            return
        if self._client_loop_id != loop_id:
            self._initialize_client()
            self._client_loop_id = loop_id

    async def _openrouter_completion(self, system_message: str, human_message: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name or "x-ai/grok-4.1-fast",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": human_message},
            ],
            temperature=self.temperature,
        )

        choice = response.choices[0] if response.choices else None
        message = getattr(choice, "message", None)
        raw_content = getattr(message, "content", "")
        content = self._normalize_content(raw_content).strip()
        return content or self._get_demo_response(human_message)

    async def generate_response(self, system_message: str, human_message: str) -> str:
        if not self.client:
            logger.info("Generating demo fallback response because OpenRouter is not configured.")
            return self._get_demo_response(human_message)

        self._ensure_client_for_current_loop()

        try:
            return await self._openrouter_completion(system_message, human_message)
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc):
                logger.warning("OpenRouter loop closed. Reinitializing client and retrying once.")
                self._initialize_client()
                if self.client:
                    self._ensure_client_for_current_loop()
                    try:
                        return await self._openrouter_completion(system_message, human_message)
                    except Exception as retry_exc:
                        logger.error(
                            "OpenRouter request retry failed (%s). Returning demo fallback response.",
                            type(retry_exc).__name__,
                        )
                        return self._get_demo_response(human_message)
            logger.error("OpenRouter request failed (%s). Returning demo fallback response.", type(exc).__name__)
            return self._get_demo_response(human_message)
        except Exception as exc:
            logger.error("OpenRouter request failed (%s). Returning demo fallback response.", type(exc).__name__)
            return self._get_demo_response(human_message)

    @staticmethod
    def _normalize_content(raw_content: Any) -> str:
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            parts: list[str] = []
            for part in raw_content:
                if isinstance(part, str):
                    parts.append(part)
                    continue
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return ""

    def generate_response_sync(self, system_message: str, human_message: str) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.generate_response(system_message, human_message))
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.generate_response(system_message, human_message))
        finally:
            loop.close()

    def _get_demo_response(self, human_message: str) -> str:
        lowered = human_message.lower()
        if "generate" in lowered or "comment" in lowered:
            return (
                "Great discussion. The tradeoff here depends on how quickly you need feedback versus execution speed. "
                "Has anyone here tested both approaches recently?"
            )
        if "adapt" in lowered:
            return (
                "Interesting angle. I have seen similar patterns where practical examples improve response quality. "
                "What outcome are you optimizing for first?"
            )
        return (
            "Thanks for sharing this. I think the strongest next step is testing one concrete tactic and measuring replies."
        )

    def is_available(self) -> bool:
        return self.client is not None

    def get_client_info(self) -> dict[str, Any]:
        return {
            "client_type": self.client_type,
            "available": self.is_available(),
            "model": self.model_name,
        }

    def reload_from_env(self) -> None:
        try:
            self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        except Exception:
            self.temperature = 0.7
        self._initialize_client()


llm_service = LLMService()
