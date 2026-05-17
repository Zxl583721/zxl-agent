from collections import Counter
import math
from pathlib import Path
import re

from src.embedding import embed_text


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
COLLECTION_NAME = "personal_knowledge_base"


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class SimpleMemoryRetriever:
    """A lightweight local retriever.

    当前版本使用词频重叠做简单检索，便于项目先跑起来。
    后续可以替换为 FAISS、Chroma 或其他向量数据库。
    """

    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks or []

    def has_documents(self) -> bool:
        return bool(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks:
            return []

        query_counter = Counter(_tokenize(query))
        if not query_counter:
            return []

        scored_chunks = []
        for chunk in self.chunks:
            chunk_counter = Counter(_tokenize(chunk["text"]))
            score = self._cosine_similarity(query_counter, chunk_counter)
            if score > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                **chunk,
                "score": score,
            }
            for score, chunk in scored_chunks[:top_k]
        ]

    @staticmethod
    def _cosine_similarity(left: Counter, right: Counter) -> float:
        common_tokens = set(left) & set(right)
        dot_product = sum(left[token] * right[token] for token in common_tokens)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))

        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)


class ChromaRetriever:
    """Retrieve relevant chunks from a local Chroma vector database."""

    def __init__(self, persist_dir: str | Path, collection_name: str = COLLECTION_NAME):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.setup_error = ""
        self.collection = None

        try:
            import chromadb
        except ImportError:
            self.setup_error = "未安装 chromadb。请先运行 pip install -r requirements.txt。"
            return

        try:
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:
            self.setup_error = f"加载 Chroma 向量库失败：{exc}"

    def has_documents(self) -> bool:
        if self.collection is None:
            return False
        return self.collection.count() > 0

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if self.collection is None:
            return []

        query_embedding = embed_text(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            chunks.append(
                {
                    "text": document,
                    "metadata": metadata or {},
                    "score": 1 / (1 + distance),
                }
            )

        return chunks
