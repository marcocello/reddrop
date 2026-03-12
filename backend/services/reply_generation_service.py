from __future__ import annotations

import json
from typing import Optional

from ..config.log import setup_logging
from ..models import Conversation, PersonaProfile
from .llm_service import llm_service

logger = setup_logging(__name__)


class ReplyGenerationService:
    @staticmethod
    def _default_tone_profile() -> dict[str, str]:
        return {
            "tone": "helpful and conversational",
            "style": "clear, concise, practical",
            "guidance": "Answer directly, stay specific, and avoid hard promotion.",
        }

    def _extract_tone_profile(self, conversation: Conversation) -> dict[str, str]:
        system_message = (
            "Extract the tone and style from a Reddit thread. "
            "Return JSON only with keys: tone, style, guidance."
        )
        thread_body = conversation.selftext.strip() or "(not provided)"
        human_message = f"""
Thread title:
{conversation.title}

Thread body:
{thread_body}
"""
        try:
            raw = llm_service.generate_response_sync(system_message, human_message)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return self._default_tone_profile()
            tone = str(parsed.get("tone", "")).strip()
            style = str(parsed.get("style", "")).strip()
            guidance = str(parsed.get("guidance", "")).strip()
            if not tone or not style or not guidance:
                return self._default_tone_profile()
            return {"tone": tone, "style": style, "guidance": guidance}
        except Exception:
            logger.info("Tone/style extraction failed. Using default tone profile.")
            return self._default_tone_profile()

    def generate_reply(
        self,
        *,
        topic: str,
        conversation: Conversation,
        persona: Optional[PersonaProfile] = None,
    ) -> str:
        tone_profile = self._extract_tone_profile(conversation)
        system_message = (
            "You write reddit-native replies. "
            "Output only the final reply text with no markdown fences. "
            "Style rules: not verbose at all, use jargon when natural, no explanation, no mansplaining. "
            "Be short, practical and genuine."
            "Do not use the em dash character."
        )
        persona_section = ""
        if persona is not None:
            persona_section = f"""
Persona name:
{persona.name}

Persona description:
{persona.description}

Persona objective:
{persona.objective}
"""
        human_message = f"""
Generate a reply for this Reddit thread.

Topic:
{topic}

{persona_section}

Tone profile extracted from thread:
- Tone: {tone_profile["tone"]}
- Style: {tone_profile["style"]}
- Guidance: {tone_profile["guidance"]}

Thread title:
{conversation.title}

Thread body:
{conversation.selftext.strip() or "(not provided)"}

Subreddit:
r/{conversation.subreddit}

Thread URL:
{conversation.url}

Keep it practical, specific, and conversational in 2-4 sentences.
"""
        response = llm_service.generate_response_sync(system_message, human_message).strip()
        if response:
            return response

        logger.info("LLM reply generation returned empty output. Using deterministic fallback.")
        return (
            f"Interesting thread. On '{topic}', I'd focus on one concrete next step and share a short result update "
            f"so others can react with specifics."
        )
