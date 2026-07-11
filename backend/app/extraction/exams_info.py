from __future__ import annotations

from .base import BaseExtractor, RetrievalPacket, extract_number_pairs, flatten_text


class ExamsInfoExtractor(BaseExtractor):
    intent = "exams"
    schema = {
        "type": "object",
        "properties": {
            "required_ege": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"subject": {"type": "string"}, "min_score": {"type": "integer"}},
                    "required": ["subject"],
                    "additionalProperties": False,
                },
            },
            "extra_exams": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "min_score": {"type": "integer"},
                        "format": {"type": "string"},
                    },
                    "required": ["subject"],
                    "additionalProperties": False,
                },
            },
            "source_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["required_ege", "extra_exams"],
        "additionalProperties": False,
    }

    def _build_prompt(self, packet: RetrievalPacket) -> str:
        return (
            "Извлеки обязательные ЕГЭ и дополнительные вступительные испытания для поступления. "
            "Раздели результат на required_ege и extra_exams. Для каждого предмета укажи минимальный балл, если он есть. "
            "Если указан формат испытания, укажи его."
            f"\nВопрос: {packet.context.raw_question}\n\nФрагменты:\n{self._joined_sections(packet)}"
        )

    def _fallback(self, packet: RetrievalPacket) -> dict:
        pairs = extract_number_pairs(flatten_text(packet))
        required, extra = [], []
        for name, score in pairs[:12]:
            entry = {"subject": name, "min_score": score}
            lowered = name.lower()
            if any(t in lowered for t in ["физ", "доп", "контрольн", "тестирован"]):
                extra.append(entry)
            else:
                required.append(entry)
        return {"required_ege": required[:6], "extra_exams": extra[:6], "source_points": self._source_points(packet)}
