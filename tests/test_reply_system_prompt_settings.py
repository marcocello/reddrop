from __future__ import annotations

from backend.models import Conversation, PersonaProfile
from backend.services.reply_generation_service import ReplyGenerationService
from backend.services import reply_generation_service as reply_module


def test_reply_generation_uses_default_system_prompt(monkeypatch) -> None:
    prompts: list[str] = []

    def _fake_generate_response_sync(system_message: str, human_message: str) -> str:
        _ = human_message
        prompts.append(system_message)
        if "Extract the tone and style" in system_message:
            return '{"tone":"helpful","style":"clear","guidance":"specific"}'
        return "Generated reply"

    monkeypatch.setattr(reply_module.llm_service, "generate_response_sync", _fake_generate_response_sync)

    service = ReplyGenerationService()
    conversation = Conversation(
        thread_id="t1",
        title="Need CRM advice",
        subreddit="sales",
        url="https://example.com/post/1",
        score=1,
        num_comments=1,
        selftext="How should we improve CRM adoption?",
        semantic_similarity=0.8,
    )
    persona = PersonaProfile(name="A", objective="C. direct and practical")
    reply = service.generate_reply(topic="crm", conversation=conversation, persona=persona)

    assert reply == "Generated reply"
    assert len(prompts) >= 2
    assert prompts[1] == ReplyGenerationService.DEFAULT_SYSTEM_PROMPT
