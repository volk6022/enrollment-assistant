from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    backend_engine: str = os.getenv("BACKEND_ENGINE", "auto").strip().lower()
    yandex_folder_id: str = os.getenv("YC_FOLDER_ID", "").strip()
    yandex_ai_api_key: str = os.getenv("YC_AI_API_KEY", os.getenv("YC_SPEECHKIT_API_KEY", "")).strip()
    yandex_model_lite: str = os.getenv("YANDEXGPT_LITE_MODEL", "yandexgpt-lite/latest")
    yandex_model_pro: str = os.getenv("YANDEXGPT_PRO_MODEL", "yandexgpt/latest")
    yandex_embedding_model: str = os.getenv("YANDEX_EMBEDDING_MODEL", "text-search-query/latest")
    yandex_classifier_model: str = os.getenv("YANDEX_CLASSIFIER_MODEL", "")
    yandex_responses_endpoint: str = os.getenv(
        "YANDEX_RESPONSES_ENDPOINT", "https://ai.api.cloud.yandex.net/v1/responses"
    )
    yandex_embeddings_endpoint: str = os.getenv(
        "YANDEX_EMBEDDINGS_ENDPOINT", "https://ai.api.cloud.yandex.net/v1/embeddings"
    )
    yandex_text_generation_endpoint: str = os.getenv(
        "YANDEX_TEXT_GENERATION_ENDPOINT", "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    )
    yandex_text_classification_endpoint: str = os.getenv(
        "YANDEX_TEXT_CLASSIFICATION_ENDPOINT", "https://ai.api.cloud.yandex.net/v1/textClassification:classify"
    )
    yandex_vector_store_id: str = os.getenv("YANDEX_VECTOR_STORE_ID", "").strip()
    yandex_files_api_base: str = os.getenv("YANDEX_FILES_API_BASE", "https://ai.api.cloud.yandex.net/v1/files")
    yandex_vector_store_api_base: str = os.getenv(
        "YANDEX_VECTOR_STORE_API_BASE", "https://ai.api.cloud.yandex.net/v1"
    )
    yandex_timeout_seconds: int = int(os.getenv("YANDEX_TIMEOUT_SECONDS", "90"))
    yandex_extract_timeout_seconds: int = int(os.getenv("YANDEX_EXTRACT_TIMEOUT_SECONDS", "120"))
    contacts_path: str = os.getenv("CONTACTS_JSON_PATH", "/data/contacts.json")

    @property
    def yandex_enabled(self) -> bool:
        return bool(self.yandex_ai_api_key and self.yandex_folder_id)

    @property
    def effective_backend_engine(self) -> str:
        if self.backend_engine and self.backend_engine != "auto":
            return self.backend_engine
        if self.yandex_enabled and self.yandex_vector_store_id:
            return "yandex_v11"
        return "legacy_rag"


settings = Settings()
