from __future__ import annotations

import re
from typing import Optional


def parse_status(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "впервые" in t or "первый раз" in t:
        return "first_time"
    if "сотрудник" in t or "действующ" in t or "служу" in t:
        return "employee"
    if ("прям" in t and "набор" in t) or "прямой набор" in t:
        return "direct"
    if "после колледжа" in t or "после спо" in t:
        return "spo_path"
    return None


def parse_level(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "адъюнкт" in t or "аспирант" in t:
        return "adjunct"
    if re.search(r"\bмагистр\w*|\bмагистрат\w*", t):
        return "master"
    if re.search(r"\bбакалавр\w*|\bбакалавриат\w*", t):
        return "bachelor"
    if re.search(r"\bспециалитет\b|\bспец\b", t):
        return "specialist"
    if re.search(r"\bспо\b|\bколледж\w*|\bсредн(?:ее|его)\s+профессиональн", t):
        return "spo"
    return None


def parse_form(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "заочн" in t:
        return "part_time"
    if "очн" in t or "дневн" in t:
        return "full_time"
    if "дистанц" in t or "дот" in t or "онлайн" in t:
        return "distance"
    return None
