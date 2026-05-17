import argparse
from pathlib import Path
from typing import TypeVar

from langchain_core.documents import Document

from src.document_loader import SUPPORTED_EXTENSIONS, load_documents
from src.langchain_zhipu import ZhipuEmbeddings
from src.retriever import COLLECTION_NAME
from src.text_splitter import split_text


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
T = TypeVar("T")


def build_documents(chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Load local documents and split them into LangChain documents."""
    documents = load_documents(DATA_DIR)
    langchain_documents = []

    for document in documents:
        text_chunks = split_text(
            document["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for index, text in enumerate(text_chunks):
            source_path = Path(document["source"])
            langchain_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "id": f"{source_path.name}-{index}",
                        "source": document["source"],
                        "chunk_index": index,
                    },
                )
            )

    return langchain_documents


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


def get_chroma_vectorstore():
    """Create a clean LangChain Chroma vector store for the current knowledge base."""
    try:
        import chromadb
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError("请先运行 pip install -r requirements.txt 安装 LangChain 和 Chroma。") from exc

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    # Rebuilding the index should reflect the latest files under data/.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=ZhipuEmbeddings(),
        collection_metadata={
            "embedding_model": "embedding-3",
        },
    )


def build_index(chunk_size: int = 800, chunk_overlap: int = 100, batch_size: int = 16) -> None:
    """Build a local Chroma vector index under vector_store/."""
    try:
        import chromadb  # noqa: F401
        import langchain_chroma  # noqa: F401
    except ImportError:
        print("索引写入失败：请先运行 pip install -r requirements.txt 安装 LangChain 和 Chroma。")
        return

    try:
        documents = build_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except RuntimeError as exc:
        print(f"文档读取失败：{exc}")
        return

    if not documents:
        supported_formats = "、".join(sorted(SUPPORTED_EXTENSIONS))
        print(f"没有找到可索引的知识库文件。请先把 {supported_formats} 文件放入 data/ 目录。")
        return

    try:
        vectorstore = get_chroma_vectorstore()
    except RuntimeError as exc:
        print(f"索引写入失败：{exc}")
        return

    print(f"已加载并切分出 {len(documents)} 个文本块，开始通过 LangChain 写入 Chroma...")

    for batch_number, document_batch in enumerate(batch_items(documents, batch_size), start=1):
        print(f"正在处理第 {batch_number} 批，共 {len(document_batch)} 个文本块...")
        try:
            vectorstore.add_documents(
                documents=document_batch,
                ids=[document.metadata["id"] for document in document_batch],
            )
        except RuntimeError as exc:
            print(f"向量化失败：{exc}")
            print("请确认已经安装依赖，并在 .env 中配置 ZHIPUAI_API_KEY。")
            return

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
