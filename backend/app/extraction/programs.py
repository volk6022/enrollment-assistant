from __future__ import annotations

import re

from .base import BaseExtractor, RetrievalPacket, flatten_text


class ProgramsByFormExtractor(BaseExtractor):
    intent = "programs_by_form"
    schema = {
        "type": "object",
        "properties": {
            "form": {"type": "string"},
            "restrictions": {"type": "array", "items": {"type": "string"}},
            "programs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "name": {"type": "string"},
                        "profile": {"type": "string"},
                        "degree": {"type": "string"},
                        "duration": {"type": "string"},
                    },
                    "required": ["code", "name"],
                    "additionalProperties": False,
                },
            },
            "source_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["programs"],
        "additionalProperties": False,
    }

    def _build_prompt(self, packet: RetrievalPacket) -> str:
        form = packet.context.study_form or "не указана"
        return (
            "Извлеки из фрагментов перечень программ набора по форме обучения. "
            "Верни только программы, относящиеся к вопросу пользователя. "
            "Если в вопросе указана очная или заочная форма, отфильтруй по ней. "
            "Для каждой программы извлеки код, название, профиль/направленность, уровень и срок обучения, если он указан."
            f"\nВопрос: {packet.context.raw_question}\nФорма из вопроса: {form}\n\nФрагменты:\n{self._joined_sections(packet)}"
        )

    def _fallback(self, packet: RetrievalPacket) -> dict:
        text = flatten_text(packet)
        programs = []
        for m in re.finditer(r"(40\.\d{2}\.\d{2})\s+([^\n]{3,120})", text):
            code = m.group(1)
            name = m.group(2).strip(" .,:;")
            if len(name) > 3:
                programs.append({"code": code, "name": name})
        restrictions = []
        lowered = text.lower()
        if "только действующие сотрудники" in lowered:
            restrictions.append("только для действующих сотрудников")
        return {"form": packet.context.study_form or "", "restrictions": restrictions, "programs": programs[:12], "source_points": self._source_points(packet)}
