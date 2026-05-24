import argparse
import hashlib
import json
from pathlib import Path
from typing import TypeVar

from langchain_core.documents import Document

from src.document_loader import SUPPORTED_EXTENSIONS, clean_text, load_document_pages, load_file_pages
from src.model_provider import configured_embedding_model_name, get_embeddings
from src.retriever import COLLECTION_NAME
from src.section_splitter import group_pages_into_sections, sanitize_id_part, split_sections_into_chunks


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MANIFEST_PATH = VECTOR_STORE_DIR / "index_manifest.json"
PARENT_STORE_PATH = VECTOR_STORE_DIR / "parent_store.json"
EMBEDDING_MODEL = configured_embedding_model_name()
INDEX_VERSION = 2
T = TypeVar("T")


def build_documents(chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Load local documents and split them into chapter-aware LangChain documents."""
    pages = load_document_pages(DATA_DIR)
    sections = attach_parent_ids(group_pages_into_sections(pages))
    chunks = split_sections_into_chunks(
        sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "id": chunk["id"],
                **chunk["metadata"],
            },
        )
        for chunk in chunks
    ]


def build_documents_for_file(
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Document], dict[str, dict]]:
    """Load one file and split it into chapter-aware LangChain documents."""
    pages = []
    for page in load_file_pages(file_path):
        text = clean_text(page.get("text", ""))
        if text:
            pages.append(
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "page": page.get("page"),
                    "text": text,
                }
            )

    sections = attach_parent_ids(group_pages_into_sections(pages))
    parents = build_parent_entries(sections)
    chunks = split_sections_into_chunks(
        sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "id": chunk["id"],
                **chunk["metadata"],
            },
        )
        for chunk in chunks
    ]
    return documents, parents


def build_chunks(chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Load local documents and split them into the legacy chunk shape."""
    return [
        {
            "id": document.metadata["id"],
            "text": document.page_content,
            "metadata": {
                key: value for key, value in document.metadata.items() if key != "id"
            },
        }
        for document in build_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]


def batch_items(items: list[T], batch_size: int) -> list[list[T]]:
    """Split a list into small batches for embedding API calls."""
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def attach_parent_ids(sections: list[dict]) -> list[dict]:
    """Attach stable parent ids to section-level contexts."""
    sections_with_ids = []
    for section_index, section in enumerate(sections):
        source_path = Path(section["source"])
        safe_chapter = sanitize_id_part(section["chapter"])
        sections_with_ids.append(
            {
                **section,
                "section_index": section_index,
                "parent_id": f"{source_path.name}-{section_index}-{safe_chapter}",
            }
        )
    return sections_with_ids


def build_parent_entries(sections: list[dict]) -> dict[str, dict]:
    parents = {}
    for section in sections:
        parent_id = section["parent_id"]
        parents[parent_id] = {
            "text": section["content"],
            "metadata": {
                "parent_id": parent_id,
                "source": section["source"],
                "filename": section["filename"],
                "chapter": section["chapter"],
                "start_page": section["start_page"] or "",
                "end_page": section["end_page"] or "",
                "section_index": section["section_index"],
            },
        }
    return parents


def get_chroma_vectorstore(rebuild: bool = False):
    """Create a LangChain Chroma vector store for the current knowledge base."""
    try:
        import chromadb
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError("请先运行 pip install -r requirements.txt 安装 LangChain 和 Chroma。") from exc

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    if rebuild:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        collection_metadata={
            "embedding_model": EMBEDDING_MODEL,
        },
    )


def build_index(
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    batch_size: int = 16,
    rebuild: bool = False,
) -> bool:
    """Build a local Chroma vector index under vector_store/."""
    try:
        import chromadb  # noqa: F401
        import langchain_chroma  # noqa: F401
    except ImportError:
        print("索引写入失败：请先运行 pip install -r requirements.txt 安装 LangChain 和 Chroma。")
        return False

    has_manifest = MANIFEST_PATH.exists()
    manifest = load_manifest()
    files = discover_knowledge_files(DATA_DIR)
    if not files and not manifest.get("files"):
        supported_formats = "、".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"没有找到可索引的知识库文件。请先把 {supported_formats} 文件放入 data/ 目录。")
        return False

    if not has_manifest:
        rebuild = True
    if manifest.get("files") and not PARENT_STORE_PATH.exists():
        rebuild = True
        manifest = empty_manifest(chunk_size, chunk_overlap)
    if should_rebuild_manifest(manifest, chunk_size, chunk_overlap):
        rebuild = True
        manifest = empty_manifest(chunk_size, chunk_overlap)
    elif rebuild:
        manifest = empty_manifest(chunk_size, chunk_overlap)
    parent_store = empty_parent_store() if rebuild else load_parent_store()

    try:
        vectorstore = get_chroma_vectorstore(rebuild=rebuild)
    except RuntimeError as exc:
        print(f"索引写入失败：{exc}")
        return False

    current_file_keys = {relative_file_key(file_path) for file_path in files}
    previous_file_keys = set(manifest.get("files", {}))
    deleted_file_keys = sorted(previous_file_keys - current_file_keys)

    for file_key in deleted_file_keys:
        file_entry = manifest["files"].get(file_key, {})
        chunk_ids = file_entry.get("chunk_ids", [])
        delete_documents(vectorstore, chunk_ids)
        remove_parent_entries(parent_store, file_entry.get("parent_ids", []))
        manifest["files"].pop(file_key, None)
        print(f"已从索引删除失效文件：{file_key}，移除 {len(chunk_ids)} 个文本块。")

    changed_files = []
    unchanged_count = 0
    for file_path in files:
        file_key = relative_file_key(file_path)
        file_hash = hash_file(file_path)
        previous = manifest.get("files", {}).get(file_key)
        if previous and previous.get("hash") == file_hash:
            unchanged_count += 1
            continue
        changed_files.append((file_path, file_key, file_hash, previous))

    if not changed_files and not deleted_file_keys:
        print(f"索引已是最新状态：{len(files)} 个文件未变化。")
        save_manifest(manifest, chunk_size, chunk_overlap)
        return True

    print(
        f"索引增量更新：新增/修改 {len(changed_files)} 个文件，"
        f"删除 {len(deleted_file_keys)} 个文件，跳过 {unchanged_count} 个未变化文件。"
    )

    for file_path, file_key, file_hash, previous in changed_files:
        if previous:
            delete_documents(vectorstore, previous.get("chunk_ids", []))
            remove_parent_entries(parent_store, previous.get("parent_ids", []))

        try:
            documents, parents = build_documents_for_file(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except RuntimeError as exc:
            print(f"文档读取失败：{file_key}：{exc}")
            return False

        if not documents:
            manifest["files"].pop(file_key, None)
            print(f"跳过空文档：{file_key}")
            continue

        print(f"正在更新文件：{file_key}，切分出 {len(documents)} 个文本块。")
        if not add_documents(vectorstore, documents, batch_size):
            return False

        parent_store["parents"].update(parents)

        manifest["files"][file_key] = {
            "hash": file_hash,
            "chunk_ids": [document.metadata["id"] for document in documents],
            "parent_ids": list(parents),
            "chunk_count": len(documents),
        }

    save_parent_store(parent_store)
    save_manifest(manifest, chunk_size, chunk_overlap)
    print(f"Chroma 索引增量更新完成：{VECTOR_STORE_DIR}")
    return True


def add_documents(vectorstore, documents: list[Document], batch_size: int) -> bool:
    for batch_number, document_batch in enumerate(batch_items(documents, batch_size), start=1):
        print(f"正在写入第 {batch_number} 批，共 {len(document_batch)} 个文本块...")
        try:
            vectorstore.add_documents(
                documents=document_batch,
                ids=[document.metadata["id"] for document in document_batch],
            )
        except RuntimeError as exc:
            print(f"向量化失败：{exc}")
            print("请确认本地 Ollama 服务已启动，或在 .env 中配置正确的模型供应商。")
            return False
        except UnicodeEncodeError as exc:
            print(f"向量化失败：知识库文本中包含无法编码的特殊字符，已停止写入。错误：{exc}")
            print("请重新运行 python build_index.py，程序会在入库前清理这类字符。")
            return False

    return True


def delete_documents(vectorstore, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    try:
        vectorstore.delete(ids=chunk_ids)
    except Exception as exc:
        print(f"删除旧文本块时出现警告：{exc}")


def discover_knowledge_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return [
        file_path
        for file_path in sorted(data_dir.iterdir())
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def hash_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file_key(file_path: Path) -> str:
    return file_path.relative_to(DATA_DIR).as_posix()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return empty_manifest()
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty_manifest()


def load_parent_store() -> dict:
    if not PARENT_STORE_PATH.exists():
        return empty_parent_store()
    try:
        return json.loads(PARENT_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty_parent_store()


def save_parent_store(parent_store: dict) -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    PARENT_STORE_PATH.write_text(
        json.dumps(parent_store, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def empty_parent_store() -> dict:
    return {
        "index_version": INDEX_VERSION,
        "parents": {},
    }


def remove_parent_entries(parent_store: dict, parent_ids: list[str]) -> None:
    for parent_id in parent_ids:
        parent_store.get("parents", {}).pop(parent_id, None)


def save_manifest(manifest: dict, chunk_size: int, chunk_overlap: int) -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    manifest["index_version"] = INDEX_VERSION
    manifest["chunk_size"] = chunk_size
    manifest["chunk_overlap"] = chunk_overlap
    manifest["embedding_model"] = EMBEDDING_MODEL
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def empty_manifest(chunk_size: int | None = None, chunk_overlap: int | None = None) -> dict:
    return {
        "index_version": INDEX_VERSION,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBEDDING_MODEL,
        "files": {},
    }


def should_rebuild_manifest(manifest: dict, chunk_size: int, chunk_overlap: int) -> bool:
    if manifest.get("index_version") not in {None, INDEX_VERSION}:
        return True
    if manifest.get("index_version") is None and manifest.get("files"):
        return True
    has_settings = manifest.get("chunk_size") is not None and manifest.get("chunk_overlap") is not None
    if not has_settings and manifest.get("files"):
        return True
    return (
        manifest.get("chunk_size") not in {None, chunk_size}
        or manifest.get("chunk_overlap") not in {None, chunk_overlap}
        or manifest.get("embedding_model") not in {None, EMBEDDING_MODEL}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建本地 RAG 知识库向量索引。")
    parser.add_argument("--chunk-size", type=int, default=400, help="用于检索的 child 文本块最大字符数。")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="相邻 child 文本块的重叠字符数。")
    parser.add_argument("--batch-size", type=int, default=16, help="每次调用 embedding API 的文本数量。")
    parser.add_argument("--rebuild", action="store_true", help="忽略增量记录，重建整个 Chroma 索引。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
    )
