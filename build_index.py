import argparse
from pathlib import Path

from src.document_loader import load_txt_documents
from src.embedding import embed_texts
from src.retriever import COLLECTION_NAME
from src.text_splitter import split_text


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"


def build_chunks(chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Load local documents and split them into chunks."""
    documents = load_txt_documents(DATA_DIR)
    chunks = []

    for document in documents:
        text_chunks = split_text(
            document["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, text in enumerate(text_chunks):
            chunks.append(
                {
                    "id": f"{Path(document['source']).stem}-{index}",
                    "text": text,
                    "metadata": {
                        "source": document["source"],
                        "chunk_index": index,
                    },
                }
            )

    return chunks


def batch_items(items: list[str], batch_size: int) -> list[list[str]]:
    """Split a list into small batches for embedding API calls."""
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def get_chroma_collection():
    """Create a clean Chroma collection for the current knowledge base."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("请先运行 pip install -r requirements.txt 安装 chromadb。") from exc

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    # Rebuilding the index should reflect the latest files under data/.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "embedding_model": "embedding-3",
        },
    )


def build_index(chunk_size: int = 800, chunk_overlap: int = 100, batch_size: int = 16) -> None:
    """Build a local Chroma vector index under vector_store/."""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        print("索引写入失败：请先运行 pip install -r requirements.txt 安装 chromadb。")
        return

    chunks = build_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print("没有找到可索引的 .txt 文件。请先把知识库文件放入 data/ 目录。")
        return

    print(f"已加载并切分出 {len(chunks)} 个文本块，开始向量化...")

    vectors = []
    texts = [chunk["text"] for chunk in chunks]
    for batch_number, text_batch in enumerate(batch_items(texts, batch_size), start=1):
        print(f"正在处理第 {batch_number} 批，共 {len(text_batch)} 个文本块...")
        try:
            vectors.extend(embed_texts(text_batch))
        except RuntimeError as exc:
            print(f"向量化失败：{exc}")
            print("请确认已经安装依赖，并在 .env 中配置 ZHIPUAI_API_KEY。")
            return

    try:
        collection = get_chroma_collection()
    except RuntimeError as exc:
        print(f"索引写入失败：{exc}")
        return

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=vectors,
    )

    print(f"Chroma 索引构建完成：{VECTOR_STORE_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建本地 RAG 知识库向量索引。")
    parser.add_argument("--chunk-size", type=int, default=800, help="每个文本块的最大字符数。")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="相邻文本块的重叠字符数。")
    parser.add_argument("--batch-size", type=int, default=16, help="每次调用 embedding API 的文本数量。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
