import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.npa_indexer import index_docx_folder

if __name__ == "__main__":
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    docs_dir = os.getenv("LEGAL_DOCX_DIR", os.getenv("NPA_DIR", "/data/npa/2025"))
    collection = os.getenv("QDRANT_COLLECTION_LEGAL", os.getenv("QDRANT_COLLECTION_NPA", "dvui_legal_2025"))

    recreate = os.getenv("RECREATE_LEGAL_COLLECTION", os.getenv("RECREATE_NPA_COLLECTION", "0")) == "1"

    res = index_docx_folder(
        npa_dir=docs_dir,
        collection=collection,
        qdrant_url=qdrant_url,
        embed_model=embed_model,
        recreate=recreate,
    )

    print(f"Docs: {res['documents']}")
    print(f"Chunks: {res['chunks']}")
    print(f"Collection: {res['collection']}")
    print("Legal corpus indexing done.")
