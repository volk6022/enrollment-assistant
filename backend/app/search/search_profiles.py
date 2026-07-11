from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..intents.schemas import IntentContext


@dataclass
class SearchProfile:
    top_k: int
    filters: dict[str, Any]
    expansions: list[str]
    preferred_doc_kinds: list[str]


PROFILE_BY_INTENT = {
    "programs_by_form": SearchProfile(8, {}, [
        "набор по формам обучения специальности направления подготовки",
        "очная форма обучения специальности направления сроки обучения",
        "заочная форма обучения специальности направления сроки обучения",
    ], ["rules_of_admission", "admission_attachment", "admission_order"]),
    "min_scores": SearchProfile(8, {"topic_scope": ["min_scores", "exams"]}, [
        "минимальные баллы егэ по предметам",
        "минимальное количество баллов егэ русский язык обществознание",
        "дополнительные вступительные испытания минимальные баллы",
    ], ["rules_of_admission", "admission_attachment", "exam_rules"]),
    "exams": SearchProfile(8, {"topic_scope": ["exams", "min_scores"]}, [
        "обязательные егэ дополнительные вступительные испытания формы проведения",
        "русский язык обществознание физическая подготовка вступительные испытания",
    ], ["rules_of_admission", "admission_attachment", "exam_rules"]),
    "without_ege": SearchProfile(8, {"topic_scope": ["eligibility", "exams"]}, [
        "кто может поступать без егэ после спо внутренние вступительные испытания",
        "спо 40 00 00 без егэ внутренние испытания институт самостоятельно",
    ], ["rules_of_admission", "exam_rules", "admission_order"]),
    "documents": SearchProfile(8, {"topic_scope": ["documents", "apply"]}, [
        "перечень документов для поступления заявление паспорт документ об образовании",
        "какие документы представляются при приеме",
    ], ["rules_of_admission", "admission_attachment", "admission_order"]),
    "apply": SearchProfile(8, {"topic_scope": ["apply", "documents"]}, [
        "куда подавать заявление кадровое подразделение по месту жительства прямой набор",
        "способы подачи документов лично почтой электронно",
    ], ["rules_of_admission", "admission_attachment", "admission_order"]),
    "benefits": SearchProfile(8, {"topic_scope": ["benefits", "eligibility"]}, [
        "льготы квоты особые права при поступлении",
        "льготы при поступлении федеральный закон об образовании",
    ], ["rules_of_admission", "benefits", "education_law"]),
    "citizenship": SearchProfile(6, {"topic_scope": ["eligibility"]}, [
        "гражданство российской федерации двойное гражданство поступление",
    ], ["rules_of_admission", "admission_order", "education_law"]),
    "health": SearchProfile(6, {}, [
        "состояние здоровья ввк военно врачебная комиссия поступление",
        "требования к состоянию здоровья поступающих",
    ], ["health_requirements", "rules_of_admission", "admission_order"]),
    "ege_validity": SearchProfile(6, {"topic_scope": ["exams", "eligibility"]}, [
        "срок действия результатов егэ 2026 какие годы действительны",
    ], ["rules_of_admission", "admission_attachment", "education_law"]),
    "after_9th_grade": SearchProfile(6, {"topic_scope": ["eligibility"]}, [
        "после 9 класса поступление возможно или нет",
        "база среднего общего образования 11 классов или спо",
    ], ["rules_of_admission", "admission_order"]),
    "paid_education": SearchProfile(6, {"topic_scope": ["eligibility"]}, [
        "платная договорная основа платное обучение осуществляется ли набор",
    ], ["rules_of_admission", "admission_order"]),
    "age_limits": SearchProfile(6, {"topic_scope": ["eligibility"]}, [
        "возраст поступления очная форма до 25 лет заочная форма ограничений нет",
    ], ["service_law", "rules_of_admission", "admission_order"]),
    "second_degree": SearchProfile(6, {"topic_scope": ["eligibility"]}, [
        "второе высшее образование заочная магистратура действующие сотрудники мвд",
    ], ["education_law", "rules_of_admission", "admission_order"]),
    "gender": SearchProfile(5, {"topic_scope": ["eligibility"]}, [
        "принимаются ли девушки поступление независимо от пола",
    ], ["education_law", "rules_of_admission"]),
    "dormitory": SearchProfile(6, {}, [
        "общежитие девушкам казарма юношам проживание стоимость общежития",
    ], ["rules_of_admission", "admission_attachment"]),
    "pass_score": SearchProfile(6, {"topic_scope": ["min_scores", "eligibility"]}, [
        "проходной балл конкурс региональная квота шанс поступления",
    ], ["rules_of_admission", "admission_attachment"]),
    "relatives_record": SearchProfile(6, {"topic_scope": ["eligibility"]}, [
        "судимость родственников препятствие для поступления направление территориальный орган",
    ], ["rules_of_admission", "admission_order"]),
    "where_apply": SearchProfile(8, {"topic_scope": ["apply", "documents"]}, [
        "куда подавать заявление кадровое подразделение по месту жительства прямой набор телефон",
    ], ["rules_of_admission", "admission_attachment", "admission_order"]),
    "transfer": SearchProfile(6, {"topic_scope": ["eligibility"]}, [
        "перевод из другого вуза не входящего в систему мвд",
    ], ["rules_of_admission", "admission_order"]),
    "accelerated": SearchProfile(6, {"topic_scope": ["eligibility"]}, [
        "сокращенная форма ускоренные сроки обучения предусмотрено или нет",
    ], ["education_law", "rules_of_admission", "admission_order"]),
    "deadlines": SearchProfile(6, {"topic_scope": ["deadlines", "apply"]}, [
        "сроки приема документов даты завершения приема",
        "до какого числа принимают документы",
    ], ["rules_of_admission", "admission_attachment", "admission_order"]),
    "contacts": SearchProfile(3, {"topic_scope": ["contacts"]}, [], ["contacts"]),
    "eligibility": SearchProfile(6, {"topic_scope": ["eligibility"]}, ["кто может поступать условия допуска к приему"], ["rules_of_admission", "admission_order", "education_law"]),
    "normative": SearchProfile(6, {}, ["правовое основание поступления"], ["education_law", "admission_order", "rules_of_admission"]),
}


def build_search_profile(ctx: IntentContext) -> SearchProfile:
    profile = PROFILE_BY_INTENT.get(ctx.intent, SearchProfile(6, {}, [], ["rules_of_admission", "admission_order"]))
    filters = dict(profile.filters)
    if ctx.level:
        filters["program_scope"] = [ctx.level, "all"]
    elif ctx.intent not in {"contacts", "normative", "dormitory", "gender", "age_limits", "paid_education", "pass_score", "relatives_record", "where_apply", "transfer", "accelerated", "citizenship", "health"}:
        filters["program_scope"] = ["all", "bachelor", "specialist", "master", "spo"]
    if ctx.study_form and ctx.intent in {"programs_by_form"}:
        filters["study_form"] = [ctx.study_form, "all"]
    return SearchProfile(profile.top_k, filters, profile.expansions, profile.preferred_doc_kinds)
