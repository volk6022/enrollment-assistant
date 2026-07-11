from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.yandex.files_client import YandexFilesClient
from app.yandex.vector_store_client import YandexVectorStoreAdminClient


def main() -> None:
    root = Path("/data/npa/2025")
    if not root.exists():
        raise SystemExit(f"Corpus dir not found: {root}")
    docs = sorted(root.glob("*.docx"))
    if not docs:
        raise SystemExit("No .docx files found")
    files_client = YandexFilesClient()
    vs_client = YandexVectorStoreAdminClient()
    file_ids = []
    for path in docs:
        meta = files_client.upload(path)
        file_id = meta.get("id")
        print(f"uploaded {path.name}: {file_id}")
        if file_id:
            file_ids.append(file_id)
    if settings.yandex_vector_store_id:
        result = vs_client.attach_files(settings.yandex_vector_store_id, file_ids)
    else:
        result = vs_client.create("enrollment-assistant-v11", file_ids)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
