"""Query rephrasing for conversational (spoken) input.

Rewrites a messy spoken question into one clean canonical question in the official
language of the NPA documents, PRESERVING exact terms/abbreviations (паспорт, ЕГЭ,
СВО, адъюнктура) and mapping colloquial wording onto NPA stock phrases (e.g.
"врачи/комиссия" -> "медицинское освидетельствование"). Used by the pipeline's
conversational multi-query mode: retrieve on BOTH the spoken query and this
canonical rewrite (RRF-union), then rerank with the canonical (clean) query.
Lifts src_recall@5 on spoken input 0.625 -> ~0.84 (RESULTS.md, "conv_multiquery_v3").

Prompt design note: the model is asked to emit a canonical question AND a keyword
line. The keyword line is DISCARDED for retrieval — keyword expansion was tested
twice and rejected (RESULTS.md) — but generating it makes the model reason about
the domain terms first, which measurably sharpens the canonical question (the
NPA-stock-phrase rule is what fixes cases like "комиссия/врачи" -> "медицинское
освидетельствование"). The rules are distilled from experiments-rag-params/runs/
keyword_guide.md (a sub-agent grounded them in the real NPA vocabulary).
"""
from __future__ import annotations

import re

from rag.config import DEFAULT, RagConfig
from rag.generate import LlamaServer

REPHRASE_SYSTEM = (
    "Ты помогаешь искать ответ в базе нормативных документов (НПА) вуза МВД (ДВЮИ МВД) "
    "по разговорному вопросу абитуриента. Верни РОВНО две строки:\n"
    "ВОПРОС: <один чёткий вопрос официальным языком; ДОСЛОВНО сохрани термины и "
    "аббревиатуры из вопроса — паспорт, СНИЛС, ЕГЭ, СВО, адъюнктура; НЕ меняй предмет "
    "вопроса и переводи бытовые слова в устойчивые обороты НПА>\n"
    "КЛЮЧЕВЫЕ: <6-10 дискриминативных терминов/оборотов через «;»>\n\n"
    "Правила:\n"
    "1. Специфичное вместо общего. Избегай общих слов, матчащих пол-корпуса: "
    "«поступление», «вступительные испытания», «приём», «документы», «упражнения», "
    "«нормативы», «баллы», «льготы».\n"
    "2. Бери устойчивые обороты из НПА: «медицинское освидетельствование», «категория "
    "годности», «расписание болезней», «дополнительное вступительное испытание», "
    "«контрольное упражнение», «минимальное количество баллов ЕГЭ», «сумма конкурсных "
    "баллов», «особое право при приёме», «первоочередной порядок зачисления».\n"
    "3. Для физподготовки — точные названия: «подтягивание на перекладине», «бег 100 м», "
    "«бег (кросс) 1000 м», «силовое комплексное упражнение».\n"
    "4. НЕ добавляй темы из соседних вопросов (в вопрос про льготы не тяни «физподготовка»).\n"
    "5. НЕ придумывай факты и определения. Незнакомый термин оставь как есть."
)
FEWSHOT = [
    {"role": "user", "content": "Слушайте, а кто там врачи будут, которые мне комиссию проводить?"},
    {"role": "assistant", "content": "ВОПРОС: Кто проводит медицинское освидетельствование кандидата на поступление?\nКЛЮЧЕВЫЕ: медицинское освидетельствование; врачи-специалисты; военно-врачебная комиссия; категория годности; состав комиссии"},
    {"role": "user", "content": "А паспорт обязательно надо подавать?"},
    {"role": "assistant", "content": "ВОПРОС: Обязательно ли предоставлять паспорт при подаче документов на поступление?\nКЛЮЧЕВЫЕ: паспорт; документ, удостоверяющий личность; перечень документов, представляемых кандидатом; личное дело кандидата"},
    {"role": "user", "content": "Вот, отец в СВО участвует, есть какая-нибудь квота для нас?"},
    {"role": "assistant", "content": "ВОПРОС: Какая квота при поступлении предусмотрена для детей участников СВО?\nКЛЮЧЕВЫЕ: СВО; специальная военная операция; особое право при приёме; первоочередной порядок зачисления; дети военнослужащих; отдельная квота приёма"},
    {"role": "user", "content": "По физре сколько баллов надо набрать чтоб сдать?"},
    {"role": "assistant", "content": "ВОПРОС: Сколько баллов нужно набрать по физической подготовке для сдачи вступительного испытания?\nКЛЮЧЕВЫЕ: физическая подготовка; контрольное упражнение; подтягивание на перекладине; бег 100 м; силовое комплексное упражнение; минимальное количество баллов"},
]


def _extract_question(text: str) -> str:
    """Pull the ВОПРОС: line; fall back to the first '?'-terminated sentence."""
    for line in text.splitlines():
        m = re.match(r"(?i)^\s*ВОПРОС\s*[:\-]\s*(.+)", line.strip())
        if m:
            return m.group(1).strip()
    i = text.find("?")
    return text[: i + 1].strip() if i != -1 else text.strip().split("\n")[0].strip()


def rephrase_canonical(server: LlamaServer, question: str, cfg: RagConfig = DEFAULT) -> str:
    """Spoken question -> clean canonical question (NPA terms, subject preserved)."""
    messages = [{"role": "system", "content": REPHRASE_SYSTEM}, *FEWSHOT,
                {"role": "user", "content": question}]
    out = server.complete(messages, cfg)
    return _extract_question(out["answer"]) or question
