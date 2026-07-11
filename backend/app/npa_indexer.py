import os
import re
import io
import zipfile
import uuid
import datetime as dt
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .knowledge import resolve_document_profile
from .legal_hierarchy import classify_legal_force, decode_escaped_unicode


MONTHS_RU = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}
POINT_RE = re.compile(r"^(\d+(?:\.\d+){0,6})[.)]\s+")


def _uuid5_id(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


def _parse_ddmmyyyy(s: str) -> Optional[str]:
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", s)
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    try:
        return dt.date(y, mo, d).isoformat()
    except Exception:
        return None


def _parse_ru_date_words(s: str) -> Optional[str]:
    m = re.search(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", (s or "").lower())
    if not m:
        return None
    d = int(m.group(1))
    mo = MONTHS_RU.get(m.group(2))
    y = int(m.group(3))
    if not mo:
        return None
    try:
        return dt.date(y, mo, d).isoformat()
    except Exception:
        return None


def _normalize_filename_dates(name: str) -> str:
    return re.sub(r"(\d{2})_(\d{2})_(\d{4})", r"\1.\2.\3", name)


def _read_docx_xml(path: str, inner_name: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    z = zipfile.ZipFile(io.BytesIO(data))
    return z.read(inner_name)


def _extract_docx_paragraphs(path: str) -> List[str]:
    xml_bytes = _read_docx_xml(path, "word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras: List[str] = []
    for p in root.findall(".//w:p", ns):
        t = "".join([(n.text or "") for n in p.findall(".//w:t", ns)])
        t = re.sub(r"\s+", " ", (t or "").strip())
        if t:
            paras.append(t)
    return paras


def _extract_docx_core_properties(path: str) -> Dict[str, Optional[str]]:
    try:
        xml_bytes = _read_docx_xml(path, "docProps/core.xml")
    except Exception:
        return {"title": None}
    root = ET.fromstring(xml_bytes)
    ns = {"dc": "http://purl.org/dc/elements/1.1/"}
    node = root.find("dc:title", ns)
    return {"title": (node.text or "").strip() if node is not None and node.text else None}


def _guess_doc_meta_from_filename(filename: str) -> Dict[str, Any]:
    fn = _normalize_filename_dates(decode_escaped_unicode(os.path.basename(filename)))
    doc_type = None
    m_type = re.search(r"^(.+?)\s+от\s+(?:\d{2}\.\d{2}\.\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})", fn, flags=re.IGNORECASE)
    if m_type:
        doc_type = m_type.group(1).strip()

    issued_date = None
    m_issued = re.search(r"от\s+(\d{2}\.\d{2}\.\d{4})", fn, flags=re.IGNORECASE)
    if m_issued:
        issued_date = _parse_ddmmyyyy(m_issued.group(1))
    if not issued_date:
        m_words = re.search(r"от\s+(\d{1,2}\s+[а-яё]+\s+\d{4})", fn, flags=re.IGNORECASE)
        if m_words:
            issued_date = _parse_ru_date_words(m_words.group(1))

    revision_date = None
    m_rev = re.search(r"ред[_.\s]*от\s+(\d{2}\.\d{2}\.\d{4})", fn, flags=re.IGNORECASE)
    if m_rev:
        revision_date = _parse_ddmmyyyy(m_rev.group(1))
    if not revision_date:
        m_rev_words = re.search(r"ред[_.\s]*от\s+(\d{1,2}\s+[а-яё]+\s+\d{4})", fn, flags=re.IGNORECASE)
        if m_rev_words:
            revision_date = _parse_ru_date_words(m_rev_words.group(1))

    doc_number = None
    m_num = re.search(r"\sN\s*([0-9A-Za-zА-Яа-яЁё\-\/]+)", fn)
    if m_num:
        doc_number = m_num.group(1).strip().rstrip(").")

    meta = {
        "doc_type": doc_type,
        "doc_number": doc_number,
        "issued_date": issued_date,
        "revision_date": revision_date,
    }
    if revision_date:
        meta["revision_year"] = int(revision_date[:4])
    return meta


def _guess_revision_from_text(paras: List[str]) -> Optional[str]:
    joined = "\n".join(paras[:120])
    m = re.search(r"ред\.\s*от\s*(\d{2}\.\d{2}\.\d{4})", joined, flags=re.IGNORECASE)
    if m:
        return _parse_ddmmyyyy(m.group(1))
    if "Список изменяющих документов" not in joined:
        return None
    tail = joined[joined.index("Список изменяющих документов") :]
    dates = [_parse_ddmmyyyy(x) for x in re.findall(r"(\d{2}\.\d{2}\.\d{4})", tail)]
    dates = [x for x in dates if x]
    return max(dates) if dates else None


def _guess_title_from_text(paras: List[str]) -> str:
    head = [p for p in paras[:80] if "КонсультантПлюс" not in p]
    if not head:
        return "Нормативный акт"
    p0 = head[0]
    m = re.search(r"N\s*[0-9A-Za-zА-Яа-яЁё\-\/]+\s*\"?(.+)$", p0)
    if m:
        tail = m.group(1).strip().strip('"')
        if len(tail) >= 12:
            return tail[:240]
    candidates = [p for p in head if len(p) >= 6 and p.upper() == p and not p.endswith(".") and len(p) <= 160]
    if candidates:
        return " ".join(candidates[:3])[:240]
    return head[0][:240]


def _is_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if re.match(r"^(РАЗДЕЛ|ГЛАВА|ПРИЛОЖЕНИЕ)\b", s, flags=re.IGNORECASE):
        return True
    if re.match(r"^[IVXLCDM]+\.\s+", s):
        return True
    if s.upper() == s and len(s) <= 160 and not re.search(r"\d{2}\.\d{2}\.\d{4}", s):
        return True
    return False


def _split_long(text: str, max_chars: int = 1200, overlap: int = 180) -> List[str]:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return [t]
    out: List[str] = []
    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        part = t[start:end].strip()
        if part:
            out.append(part)
        if end >= len(t):
            break
        start = max(0, end - overlap)
    return out


def chunk_docx(paras: List[str], meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    section_path: List[str] = []
    chunks: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_point: Optional[str] = None
    buf_section: List[str] = []

    def flush():
        nonlocal buf, buf_point, buf_section
        if not buf:
            return
        joined = "\n".join(buf).strip()
        for part in _split_long(joined):
            chunks.append({"text": part, "point": buf_point, "section_path": list(buf_section)})
        buf = []
        buf_point = None
        buf_section = []

    for p in paras:
        line = (p or "").strip()
        if not line:
            continue
        if _is_heading(line):
            if buf:
                buf.append(line)
            else:
                if len(section_path) >= 4:
                    section_path = section_path[:3]
                section_path.append(line[:120])
            continue
        m = POINT_RE.match(line)
        if m:
            flush()
            buf_point = m.group(1)
            buf_section = list(section_path)
            buf = [line]
            continue
        if not buf:
            continue
        buf.append(line)
    flush()
    return chunks


def ensure_collection(client: QdrantClient, collection: str, embedder: TextEmbedding, recreate: bool = False) -> None:
    if recreate:
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            pass
    try:
        client.get_collection(collection_name=collection)
        return
    except Exception:
        pass
    vec = next(embedder.embed(["test"]))
    client.create_collection(collection_name=collection, vectors_config=VectorParams(size=len(vec), distance=Distance.COSINE))


def index_docx_folder(npa_dir: str, collection: str, qdrant_url: str, embed_model: str, recreate: bool = False, batch_size: int = 64) -> Dict[str, Any]:
    client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
    embedder = TextEmbedding(embed_model)
    ensure_collection(client, collection, embedder, recreate=recreate)

    files = sorted(os.path.join(npa_dir, f) for f in os.listdir(npa_dir) if f.lower().endswith(".docx"))
    total_docs = 0
    total_chunks = 0

    for path in files:
        total_docs += 1
        filename_raw = os.path.basename(path)
        filename = decode_escaped_unicode(filename_raw)
        paras = _extract_docx_paragraphs(path)
        core_meta = _extract_docx_core_properties(path)
        meta_fn = _guess_doc_meta_from_filename(filename_raw)
        title = core_meta.get("title") or _guess_title_from_text(paras)
        preview_text = "\n".join(paras[:25])
        issued_date = meta_fn.get("issued_date") or _parse_ru_date_words("\n".join(paras[:8])) or None
        revision_date = meta_fn.get("revision_date") or _guess_revision_from_text(paras)
        doc_type = meta_fn.get("doc_type")
        doc_number = meta_fn.get("doc_number")

        legal_force = classify_legal_force(title=title, doc_type=doc_type, source_name=filename, preview_text=preview_text)
        resolved = resolve_document_profile(
            source=filename,
            source_file=filename_raw,
            title=title,
            doc_type=doc_type,
            preview_text=preview_text,
            base_legal_force={"code": legal_force.code, "label": legal_force.label, "level": legal_force.level},
        )

        issued_date = resolved.get("issued_date") or issued_date
        revision_date = resolved.get("revision_date") or revision_date
        revision_year = int(str(revision_date)[:4]) if revision_date else None

        doc_payload_base = {
            "source_type": "legal_docx",
            "source": filename,
            "source_file": filename_raw,
            "doc_title": decode_escaped_unicode(resolved.get("title") or title),
            "doc_type": decode_escaped_unicode(doc_type or "") or None,
            "doc_number": resolved.get("doc_number") or doc_number,
            "issued_date": issued_date,
            "revision_date": revision_date,
            "revision_year": revision_year,
            "legal_force_code": resolved.get("legal_force_code") or legal_force.code,
            "legal_force_name": resolved.get("legal_force_name") or legal_force.label,
            "legal_force_level": resolved.get("legal_force_level") or legal_force.level,
            "doc_kind": resolved.get("doc_kind"),
            "program_scope": resolved.get("program_scope") or ["all"],
            "topic_scope": resolved.get("topic_scope") or [],
            "is_general_law": bool(resolved.get("is_general_law")),
            "is_local_admission_rule": bool(resolved.get("is_local_admission_rule")),
            "is_contact_source": bool(resolved.get("is_contact_source")),
        }

        chunks = chunk_docx(paras, doc_payload_base)
        total_chunks += len(chunks)

        for i0 in range(0, len(chunks), batch_size):
            part = chunks[i0 : i0 + batch_size]
            texts = [c["text"] for c in part]
            vectors = list(embedder.embed(texts))
            points: List[PointStruct] = []
            for j, (c, v) in enumerate(zip(part, vectors)):
                payload = dict(doc_payload_base)
                payload.update({"text": c["text"], "point": c.get("point"), "pages": None, "section_path": c.get("section_path") or []})
                pid = _uuid5_id(f"{filename_raw}|{c.get('point')}|{i0+j}")
                points.append(PointStruct(id=pid, vector=list(v), payload=payload))
            client.upsert(collection_name=collection, points=points)

    return {"documents": total_docs, "chunks": total_chunks, "collection": collection}


index_npa_folder = index_docx_folder
