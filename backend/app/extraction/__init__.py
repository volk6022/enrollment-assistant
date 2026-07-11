from .apply import ApplyExtractor
from .benefits import BenefitsExtractor
from .documents import DocumentsExtractor
from .exams_info import ExamsInfoExtractor
from .min_scores import MinScoresExtractor
from .policy import PolicyExtractor
from .programs import ProgramsByFormExtractor
from .without_ege import WithoutEGEExtractor


EXTRACTOR_BY_INTENT = {
    "programs_by_form": ProgramsByFormExtractor,
    "min_scores": MinScoresExtractor,
    "documents": DocumentsExtractor,
    "apply": ApplyExtractor,
    "without_ege": WithoutEGEExtractor,
    "benefits": BenefitsExtractor,
    "exams": ExamsInfoExtractor,
    "citizenship": PolicyExtractor,
    "health": PolicyExtractor,
    "ege_validity": PolicyExtractor,
    "after_9th_grade": PolicyExtractor,
    "paid_education": PolicyExtractor,
    "age_limits": PolicyExtractor,
    "second_degree": PolicyExtractor,
    "gender": PolicyExtractor,
    "dormitory": PolicyExtractor,
    "pass_score": PolicyExtractor,
    "relatives_record": PolicyExtractor,
    "where_apply": PolicyExtractor,
    "transfer": PolicyExtractor,
    "accelerated": PolicyExtractor,
}
