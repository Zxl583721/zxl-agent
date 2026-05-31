import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} 必须是 JSON 数组或 JSONL。")
        return data
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSONL：{exc}") from exc
    return rows


def chunk_id(chunk: dict) -> str:
    metadata = chunk.get("metadata", {}) or {}
    return str(
        metadata.get("id")
        or metadata.get("chunk_id")
        or (
            metadata.get("source"),
            metadata.get("chapter"),
            metadata.get("start_page"),
            metadata.get("end_page"),
            metadata.get("chunk_index"),
        )
    )


def source_text(chunk_or_source: dict) -> str:
    metadata = chunk_or_source.get("metadata", chunk_or_source) or {}
    return " ".join(
        str(metadata.get(field, ""))
        for field in ("filename", "source", "chapter", "page_range", "start_page", "end_page")
    )


def source_hit_ids(chunks: list[dict], expected_sources: list[str]) -> list[str]:
    expected = [str(item).strip() for item in expected_sources if str(item).strip()]
    if not expected:
        return []

    hits = []
    for chunk in chunks:
        text = source_text(chunk)
        if any(item in text for item in expected):
            hits.append(chunk_id(chunk))
    return hits


def case_relevant_ids(case: dict, chunks: list[dict]) -> set[str]:
    explicit = {str(item) for item in case.get("relevant_chunk_ids", []) if str(item).strip()}
    if explicit:
        return explicit
    return set(source_hit_ids(chunks, case.get("expected_sources", [])))


def init_retriever(vector_dir: Path, collection_name: str | None = None):
    from src.retriever import ChromaRetriever

    kwargs = {"persist_dir": vector_dir}
    if collection_name:
        kwargs["collection_name"] = collection_name
    retriever = ChromaRetriever(**kwargs)
    if retriever.setup_error:
        raise RuntimeError(retriever.setup_error)
    if not retriever.has_documents():
        raise RuntimeError(f"{vector_dir} 中没有可用索引，请先运行 python build_index.py。")
    return retriever
