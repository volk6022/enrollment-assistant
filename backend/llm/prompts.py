"""Prompt templates for both LlamaClient call paths.

One rule governs every template here (plan.md §4.3 / streaming-research-
findings.md §4.2, verified: repeat updates cost ~60ms regardless of
transcript length once `cache_prompt` reuses the unchanged prefix):
**volatile content goes last.** Whatever is stable for the lifetime of a
turn — role, RAG context, dialogue history so far — comes first; the
transcript still growing as the caller speaks, and whatever is specific to
the exact instant a decision is requested, come last.

`stream_answer` goes through /v1/chat/completions, which llama-server
renders through the model's own Jinja chat template — so
`build_answer_messages` only needs to return role/content pairs, nothing
rendered by hand.

`decide()` goes through the raw /completion endpoint, which does NOT apply
that template. The `build_*_prompt` functions below therefore render ChatML
markup themselves (`<|im_start|>{role}\\n{content}<|im_end|>\\n`, confirmed
against this model's own chat_template via /props), ending on an open
`<|im_start|>assistant\\n` so the json_schema grammar constrains generation
from the very first token.
"""

from __future__ import annotations

from backend.llm.client import Message

_ROLE_INSTRUCTION_LISTEN = (
    "Ты — ассистент приёмной комиссии института. Собеседник говорит без остановки "
    "уже 20 секунд. Реши, нужно ли вмешаться и переспросить, чтобы убедиться в "
    "понимании, или продолжать слушать молча. В поле understood запиши то, что ты "
    "уже понял из его речи на данный момент."
)

_ROLE_INSTRUCTION_BARGE_IN = (
    "Ты — ассистент приёмной комиссии института. Ты сейчас произносишь ответ, и "
    "собеседник заговорил одновременно с тобой. Реши, требует ли он прервать твой "
    "ответ прямо сейчас. Поддакивание («ага», «угу», «понятно», «да») перебиванием "
    "не является — из-за него прерываться не нужно."
)

_ROLE_INSTRUCTION_INTENT = (
    "Ты — ассистент приёмной комиссии института. Определи намерение собеседника по "
    "истории диалога и его последней реплике. intent: question — задал вопрос по "
    "существу; clarification_needed — вопрос есть, но неточен, нужно уточнить; "
    "unclear — не понятно, чего он хочет; goodbye — прощается или благодарит и "
    "завершает разговор; smalltalk — реплика не по теме приёмной комиссии. Если "
    "нужен поиск по базе знаний, сформулируй в query самостоятельный поисковый "
    "запрос; иначе оставь query пустым."
)

_EMPTY_HISTORY_PLACEHOLDER = "(пусто, это начало разговора)"


def _chat_markup(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def _render_decide_prompt(*, system: str, user: str) -> str:
    return _chat_markup("system", system) + _chat_markup("user", user) + "<|im_start|>assistant\n"


def build_answer_messages(
    *,
    system_prompt: str,
    rag_context: str,
    dialogue_history: str,
    transcript: str,
) -> list[Message]:
    """system + user turns for stream_answer.

    `system_prompt` already carries role and scenario rules
    (dialogue/scenarios.yaml, owned by T-08) — this function only owns the
    layout, not the content of the rules. The pre-filled
    `<think>\\n\\n</think>` assistant turn that suppresses reasoning is NOT
    added here: it's protocol mechanics specific to /v1/chat/completions, so
    LlamaClient.stream_answer appends it itself.
    """
    sections: list[str] = []
    if rag_context.strip():
        sections.append(f"Контекст из базы знаний:\n{rag_context.strip()}")
    if dialogue_history.strip():
        sections.append(f"История диалога:\n{dialogue_history.strip()}")
    sections.append(f"Текущая реплика собеседника: {transcript.strip()}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def build_interject_prompt(*, dialogue_history: str, transcript_so_far: str) -> str:
    """InterjectDecision prompt (FR-07). Volatile part: the growing transcript."""
    user = (
        f"История диалога:\n{dialogue_history.strip() or _EMPTY_HISTORY_PLACEHOLDER}\n\n"
        f"Собеседник говорит без остановки, вот его речь на данный момент:\n"
        f"{transcript_so_far.strip()}"
    )
    return _render_decide_prompt(system=_ROLE_INSTRUCTION_LISTEN, user=user)


def build_barge_in_prompt(
    *,
    dialogue_history: str,
    draft_answer: str,
    voiced_seconds: float,
    total_seconds: float,
    interlocutor_transcript: str,
) -> str:
    """BargeInDecision prompt (FR-13, input set is FR-14 verbatim).

    Layout follows llm.md §5's stable-first order: instruction, history up
    to the start of this answer, and the answer itself go in the system
    turn — none of that changes while this same answer is being voiced.
    Voiced fraction and what the interlocutor is saying right now go in the
    user turn, because those change on every call.
    """
    voiced_pct = round((voiced_seconds / total_seconds) * 100) if total_seconds else 0
    system = (
        f"{_ROLE_INSTRUCTION_BARGE_IN}\n\n"
        f"История диалога до начала этого ответа:\n"
        f"{dialogue_history.strip() or _EMPTY_HISTORY_PLACEHOLDER}\n\n"
        f"Твой формируемый ответ целиком:\n{draft_answer.strip()}"
    )
    user = (
        f"Озвучено {voiced_seconds:.1f} с из {total_seconds:.1f} с ({voiced_pct}%).\n"
        f"Собеседник говорит сейчас: {interlocutor_transcript.strip()}"
    )
    return _render_decide_prompt(system=system, user=user)


def build_intent_prompt(*, dialogue_history: str, transcript: str) -> str:
    """IntentDecision prompt (FR-21/FR-22). Volatile part: the latest utterance."""
    user = (
        f"История диалога:\n{dialogue_history.strip() or _EMPTY_HISTORY_PLACEHOLDER}\n\n"
        f"Последняя реплика собеседника: {transcript.strip()}"
    )
    return _render_decide_prompt(system=_ROLE_INSTRUCTION_INTENT, user=user)
