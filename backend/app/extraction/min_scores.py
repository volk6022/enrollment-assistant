from __future__ import annotations

from .base import BaseExtractor, RetrievalPacket, extract_number_pairs, flatten_text


class MinScoresExtractor(BaseExtractor):
    intent = "min_scores"
    schema = {
        "type": "object",
        "properties": {
            "program": {"type": "string"},
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "min_score": {"type": "integer"},
                    },
                    "required": ["subject", "min_score"],
                    "additionalProperties": False,
                },
            },
            "source_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["scores"],
        "additionalProperties": False,
    }

    def _fallback(self, packet: RetrievalPacket) -> dict:
        text = flatten_text(packet)
        scores = [{"subject": name, "min_score": score} for name, score in extract_number_pairs(text)[:12]]
        return {"program": packet.context.level or "all", "scores": scores, "source_points": self._source_points(packet)}
