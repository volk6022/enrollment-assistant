from __future__ import annotations

from .base import BaseExtractor, RetrievalPacket


class PolicyExtractor(BaseExtractor):
    intent = "policy"
    schema = {
        "type": "object",
        "properties": {
            "direct_answer": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "source_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["direct_answer", "key_points"],
        "additionalProperties": False,
    }

    def _build_prompt(self, packet: RetrievalPacket) -> str:
        intent = packet.context.intent
        return (
            "Ты извлекаешь факты для приемной комиссии. Верни JSON по схеме. "
            "Поле direct_answer должно содержать краткий прямой ответ на вопрос в одном предложении. "
            "Поле key_points — до 5 коротких фактов по делу. Поле conditions — важные условия и ограничения. "
            "Не цитируй документы и не пиши общие фразы вроде 'по найденным фрагментам'."
            f"\nIntent: {intent}\nВопрос: {packet.context.raw_question}\n\nФрагменты:\n{self._joined_sections(packet)}"
        )

    def _fallback(self, packet: RetrievalPacket) -> dict:
        snippets = []
        for section in packet.sections[:3]:
            for line in section.splitlines():
                line = line.strip(" -•;.")
                if 10 < len(line) < 180:
                    snippets.append(line)
                if len(snippets) >= 5:
                    break
            if len(snippets) >= 5:
                break
        direct = snippets[0] if snippets else "Не удалось извлечь краткий ответ."
        return {"direct_answer": direct, "key_points": snippets[:5], "conditions": [], "source_points": self._source_points(packet)}
