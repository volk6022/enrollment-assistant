"""Pre-TTS text normalization: fix inputs Silero silently mangles or drops.

Silero just SKIPS bare digit runs ("42"/"5"/"100") instead of reading them --
the LLM answer routinely contains them (scores, deadlines, counts), so every
such answer loses information on the audio side with no error surfaced
anywhere. Converting digits to Russian words before synthesis was tried as a
prompt instruction first (see "что вообще предстоит ещё сделать.txt") but the
2B model didn't reliably follow it; a hardcoded text-level fix is cheap and
deterministic, so that's the approach here instead.

Verification loop for any rule added here: txt -> Silero -> wav -> faster-whisper
-> txt, read back the transcript and confirm the number/term survived.
"""
from __future__ import annotations

import re

_ONES = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_ONES_FEM = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
          "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят",
         "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
             "шестьсот", "семьсот", "восемьсот", "девятьсот"]

# (singular, few(2-4), many(5+/0)) -- standard Russian plural-count agreement.
_SCALES = [
    (1_000, ("тысяча", "тысячи", "тысяч"), True),           # feminine -> use _ONES_FEM
    (1_000_000, ("миллион", "миллиона", "миллионов"), False),
    (1_000_000_000, ("миллиард", "милиарда", "миллиардов"), False),
]


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    n100 = n % 100
    n10 = n % 10
    if 11 <= n100 <= 14:
        return forms[2]
    if n10 == 1:
        return forms[0]
    if 2 <= n10 <= 4:
        return forms[1]
    return forms[2]


def _three_digits(n: int, feminine: bool = False) -> list[str]:
    """0 < n < 1000 -> word list (nominative case)."""
    words = []
    h, rest = divmod(n, 100)
    if h:
        words.append(_HUNDREDS[h])
    if rest >= 10 and rest < 20:
        words.append(_TEENS[rest - 10])
    else:
        t, o = divmod(rest, 10)
        if t:
            words.append(_TENS[t])
        if o:
            words.append((_ONES_FEM if feminine else _ONES)[o])
    return words


def int_to_ru_words(n: int) -> str:
    """Cardinal number -> Russian words, nominative case. Handles 0..999_999_999_999."""
    if n == 0:
        return "ноль"
    neg = n < 0
    n = abs(n)
    chunks = []
    remaining = n
    scale_idx = 0
    parts_by_scale = []
    # split into (billions, millions, thousands, units)
    for base, forms, feminine in reversed(_SCALES):
        count, remaining = divmod(remaining, base)
        if count:
            words = _three_digits(count, feminine=feminine)
            words.append(_plural(count, forms))
            parts_by_scale.append(" ".join(words))
    if remaining or not parts_by_scale:
        words = _three_digits(remaining)
        if words:
            parts_by_scale.append(" ".join(words))
    result = " ".join(parts_by_scale) if parts_by_scale else "ноль"
    return ("минус " + result) if neg else result


_TIME_RE = re.compile(r"(?<![\wа-яА-ЯёЁ:])([01]?\d|2[0-3]):([0-5]\d)(?![\wа-яА-ЯёЁ:])")
_NUMBER_RE = re.compile(r"(?<![\wа-яА-ЯёЁ])-?\d+(?:[.,]\d+)?(?![\wа-яА-ЯёЁ])")

# Dates need the ORDINAL genitive ("двадцать пятого июля", not the cardinal
# "двадцать пять июля" _replace_number would produce) -- caught in testing:
# "20 июня" read as "двадцать июня" instead of "двадцатого июня", vs. "20
# баллов" which correctly stays cardinal "двадцать". Only Russian grammatical
# number IS distinguished by context (number+month / number+"года" = ordinal;
# everything else = cardinal), so these date patterns must be matched and
# substituted BEFORE the generic cardinal _NUMBER_RE runs.
_ONES_ORD = ["", "первого", "второго", "третьего", "четвёртого", "пятого",
             "шестого", "седьмого", "восьмого", "девятого"]
_TEENS_ORD = ["десятого", "одиннадцатого", "двенадцатого", "тринадцатого",
              "четырнадцатого", "пятнадцатого", "шестнадцатого", "семнадцатого",
              "восемнадцатого", "девятнадцатого"]
_TENS_ORD = ["", "", "двадцатого", "тридцатого", "сорокового", "пятидесятого",
             "шестидесятого", "семидесятого", "восьмидесятого", "девяностого"]
_HUNDREDS_ORD = ["", "сотого", "двухсотого", "трёхсотого", "четырёхсотого",
                 "пятисотого", "шестисотого", "семисотого", "восьмисотого", "девятисотого"]

_MONTHS_GENITIVE = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
                    "августа", "сентября", "октября", "ноября", "декабря"]
_DATE_DAY_RE = re.compile(
    r"(?<!\d)([0-9]|[12][0-9]|3[01])\s+(" + "|".join(_MONTHS_GENITIVE) + r")\b"
)
_DATE_YEAR_RE = re.compile(r"(?<!\d)(\d{4})\s+(года|году)\b")

# Plain counts (points/documents/rubles) also decline by case, separately from
# the date-ordinal issue above: "не менее 20 баллов" needs genitive "не менее
# двадцати баллов", while "минимум 20 баллов" (no case-governing word) stays
# cardinal nominative "двадцать" -- confirmed both forms occur in real answers.
# Full Russian case declension is genuinely a solved-elsewhere problem (verb/
# preposition government, noun-numeral agreement), so this uses pymorphy3
# (Russian morphology, actively maintained fork of pymorphy2) to inflect our
# own generated cardinal words rather than hand-rolling a second grammar table.
_GENITIVE_TRIGGERS = ["не менее", "не более", "менее", "более", "свыше", "от", "до"]
_GENITIVE_TRIGGER_RE = re.compile(
    r"(?<![\wа-яА-ЯёЁ])(" + "|".join(_GENITIVE_TRIGGERS) + r")\s+(\d+)(?![.,]\d)(?![\wа-яА-ЯёЁ])",
    re.IGNORECASE,
)
_MORPH = None


def _get_morph():
    global _MORPH
    if _MORPH is None:
        import pymorphy3
        _MORPH = pymorphy3.MorphAnalyzer(lang="ru")
    return _MORPH


def _genitive_word(word: str) -> str:
    """Inflect one Russian word to genitive case. Prefers a NUMR (numeral)
    reading when the morphology is ambiguous -- e.g. "сто" parses as either
    the numeral 100 or an unrelated fixed noun abbreviation with equal
    probability; without this it silently keeps the wrong (nominative
    abbreviation) reading instead of declining to "ста"."""
    morph = _get_morph()
    parses = morph.parse(word)
    numeral_parses = [p for p in parses if "NUMR" in p.tag]
    p = numeral_parses[0] if numeral_parses else parses[0]
    inflected = p.inflect({"gent"})
    return inflected.word if inflected else word


def genitive_cardinal(n: int) -> str:
    """n -> Russian cardinal, genitive case (e.g. 20 -> "двадцати")."""
    phrase = int_to_ru_words(n)
    return " ".join(w if w == "минус" else _genitive_word(w) for w in phrase.split())


def _replace_genitive_trigger(match: re.Match) -> str:
    trigger, num = match.group(1), match.group(2)
    return f"{trigger} {genitive_cardinal(int(num))}"


def ordinal_genitive(n: int) -> str:
    """n -> Russian ordinal, genitive/dative case (masc/neut singular: "...ого").
    Only the last nonzero group of a compound number takes the ordinal ending;
    everything before it stays cardinal (e.g. 2026 -> "две тысячи двадцать
    шестого", 25 -> "двадцать пятого"). Covers 1..999_999, enough for any
    calendar day or year this project will speak."""
    thousands, rem = divmod(n, 1000)
    parts = []
    if thousands:
        if rem == 0:
            # exact multiple of 1000 (e.g. year 2000) -- rare for our use, crude fallback
            return f"{int_to_ru_words(thousands)}-тысячного"
        words = _three_digits(thousands, feminine=True)
        words.append(_plural(thousands, ("тысяча", "тысячи", "тысяч")))
        parts.append(" ".join(words))
    if rem:
        h, rest = divmod(rem, 100)
        if h:
            if rest == 0:
                parts.append(_HUNDREDS_ORD[h])
                return " ".join(parts)
            parts.append(_HUNDREDS[h])
        if rest:
            if rest < 10:
                parts.append(_ONES_ORD[rest])
            elif rest < 20:
                parts.append(_TEENS_ORD[rest - 10])
            else:
                t, o = divmod(rest, 10)
                if o == 0:
                    parts.append(_TENS_ORD[t])
                else:
                    parts.append(_TENS[t])
                    parts.append(_ONES_ORD[o])
    return " ".join(parts) if parts else "нулевого"


def _replace_date_day(match: re.Match) -> str:
    day, month = match.group(1), match.group(2)
    return f"{ordinal_genitive(int(day))} {month}"


def _replace_date_year(match: re.Match) -> str:
    year, word = match.group(1), match.group(2)
    return f"{ordinal_genitive(int(year))} {word}"


def _replace_time(match: re.Match) -> str:
    hour, minute = int(match.group(1)), int(match.group(2))
    hour_words = int_to_ru_words(hour)
    if minute == 0:
        return f"{hour_words} ноль ноль"
    tens, ones = divmod(minute, 10)
    minute_words = int_to_ru_words(minute) if tens else f"ноль {int_to_ru_words(ones)}"
    return f"{hour_words} {minute_words}"


def _replace_number(match: re.Match) -> str:
    raw = match.group(0)
    neg = raw.startswith("-")
    if neg:
        raw = raw[1:]
    if "." in raw or "," in raw:
        sep = "." if "." in raw else ","
        int_part, frac_part = raw.split(sep, 1)
        int_words = int_to_ru_words(int(int_part)) if int_part else "ноль"
        # read the fractional digits one by one (e.g. "3,5" -> "три целых пять")
        frac_words = " ".join(int_to_ru_words(int(d)) for d in frac_part)
        out = f"{int_words} целых {frac_words}"
    else:
        out = int_to_ru_words(int(raw))
    return ("минус " + out) if neg else out


_LETTER_RU = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи",
    "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн",
    "o": "оу", "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю",
    "v": "ви", "w": "дабл-ю", "x": "икс", "y": "уай", "z": "зет",
}

# Round-trip-verified (txt -> Silero -> wav -> faster-whisper -> txt) via
# scratchpad/tts_normalize_synth_test.py + stt_transcribe_roundtrip.py: bare
# Latin-script terms are either silently DROPPED ("IELTS или TOEFL" vanished
# entirely) or corrupt the whole surrounding sentence ("ИТ и Data Science"
# garbled everything after it). Cyrillic transliterations of English loanwords
# ("софт скиллы", "тайм менеджмент") came through fine, so the fix is to
# transliterate known terms rather than pass Latin script through. Extend this
# table as new admissions-context terms are found to fail.
_KNOWN_TERMS = {
    "ielts": "айэлтс", "toefl": "тоефл", "gre": "джи-ар-и", "sat": "эс-эй-ти",
    "gpa": "джи-пи-эй", "it": "айти", "cs": "си-эс", "mba": "эм-би-эй",
    "phd": "пи-эйч-ди", "data science": "дата сайенс",
    "machine learning": "машинное обучение", "deep learning": "глубокое обучение",
}
# longest phrase first so "data science" matches before a stray "data"/"science" would
_KNOWN_TERMS_RE = re.compile(
    r"(?<![\wа-яА-ЯёЁ])(" + "|".join(sorted((re.escape(k) for k in _KNOWN_TERMS), key=len, reverse=True))
    + r")(?![\wа-яА-ЯёЁ])",
    re.IGNORECASE,
)
_LATIN_WORD_RE = re.compile(r"(?<![\wа-яА-ЯёЁ])[A-Za-z]+(?![\wа-яА-ЯёЁ])")


def _replace_known_term(match: re.Match) -> str:
    return _KNOWN_TERMS[match.group(0).lower()]


def _replace_latin_fallback(match: re.Match) -> str:
    """Any Latin-script word not in _KNOWN_TERMS: spell it out letter-by-letter
    rather than let Silero drop it or garble the sentence around it. Not
    natural-sounding, but deterministic and verified not to corrupt output."""
    word = match.group(0)
    return "-".join(_LETTER_RU.get(ch.lower(), ch) for ch in word)


def transliterate_latin(text: str) -> str:
    text = _KNOWN_TERMS_RE.sub(_replace_known_term, text)
    return _LATIN_WORD_RE.sub(_replace_latin_fallback, text)


def normalize_for_tts(text: str) -> str:
    """Rewrite bare digit numbers and Latin-script terms in `text` into
    Silero-safe Russian, idempotent and safe to call on any answer text
    before handing it to Silero (numbers/words inside identifiers are left
    untouched by the boundary regexes)."""
    if not text:
        return text
    text = _TIME_RE.sub(_replace_time, text)
    text = _DATE_DAY_RE.sub(_replace_date_day, text)
    text = _DATE_YEAR_RE.sub(_replace_date_year, text)
    text = _GENITIVE_TRIGGER_RE.sub(_replace_genitive_trigger, text)
    text = _NUMBER_RE.sub(_replace_number, text)
    return transliterate_latin(text)
