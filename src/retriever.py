from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

from src.model_provider import get_embeddings


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
LATIN_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
COLLECTION_NAME = "personal_knowledge_base"
RRF_K = 60
PARENT_STORE_FILE = "parent_store.json"


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _bm25_tokenize(text: str) -> list[str]:
    normalized = str(text).lower()
    tokens = LATIN_TOKEN_PATTERN.findall(normalized)
    tokens.extend(CJK_PATTERN.findall(normalized))
    return tokens


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
        self.vectorstore = None
        self.collection = None
        self.keyword_index = None
        self.last_vector_error = ""
        self.vector_disabled = False
        self.parent_store = self._load_parent_store()

        try:
            from langchain_chroma import Chroma
        except ImportError:
            self.setup_error = "未安装 LangChain Chroma。请先运行 pip install -r requirements.txt。"
            return

        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=get_embeddings(),
                persist_directory=str(self.persist_dir),
            )
            self.collection = self.vectorstore._collection
        except Exception as exc:
            self.setup_error = f"加载 Chroma 向量库失败：{exc}"

    def has_documents(self) -> bool:
        if self.collection is None:
            return False
        return self.collection.count() > 0

    def retrieve(self, query: str, top_k: int = 3, keyword_queries: list[str] | None = None) -> list[dict]:
        if self.vectorstore is None:
            return []

        recall_k = max(top_k * 2, top_k)
        vector_chunks = self._vector_retrieve(query, recall_k)
        keyword_chunks = self._keyword_retrieve_many([query, *(keyword_queries or [])], recall_k)
        fused_chunks = self._rrf_fuse(vector_chunks, keyword_chunks, top_k=recall_k)
        return self._expand_parent_chunks(fused_chunks, top_k=top_k)

    def _vector_retrieve(self, query: str, top_k: int) -> list[dict]:
        if self.vector_disabled:
            return []

        chunks = []
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        except Exception as exc:
            self.last_vector_error = f"{exc.__class__.__name__}: {exc}"
            self.vector_disabled = True
            return []

        for document, distance in results:
            chunks.append(
                {
                    "text": document.page_content,
                    "metadata": document.metadata or {},
                    "score": 1 / (1 + distance),
                }
            )

        return chunks

    def _keyword_retrieve(self, query: str, top_k: int) -> list[dict]:
        index = self._get_keyword_index()
        if not index["documents"]:
            return []

        query_tokens = _bm25_tokenize(query)
        if not query_tokens:
            return []

        query_terms = set(query_tokens)
        scored_chunks = []
        total_documents = len(index["documents"])
        average_length = index["average_length"] or 1.0
        k1 = 1.5
        b = 0.75

        for document in index["documents"]:
            score = 0.0
            token_counts = document["token_counts"]
            document_length = document["length"] or 1

            for term in query_terms:
                term_frequency = token_counts.get(term, 0)
                if term_frequency == 0:
                    continue

                document_frequency = index["document_frequency"].get(term, 0)
                idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = term_frequency + k1 * (1 - b + b * document_length / average_length)
                score += idf * (term_frequency * (k1 + 1)) / denominator

            if score > 0:
                scored_chunks.append((score, document))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "text": document["text"],
                "metadata": document["metadata"],
                "score": score,
                "keyword_score": score,
            }
            for score, document in scored_chunks[:top_k]
        ]

    def _keyword_retrieve_many(self, queries: list[str], top_k: int) -> list[dict]:
        merged_scores = defaultdict(float)
        merged_chunks = {}

        for query_index, query in enumerate(unique_queries(queries)):
            chunks = self._keyword_retrieve(query, top_k)
            channel_name = "keyword" if query_index == 0 else f"keyword_expanded_{query_index}"
            for rank, chunk in enumerate(chunks, start=1):
                chunk_id = chunk_identity(chunk)
                merged_scores[chunk_id] += 1 / (RRF_K + rank)
                merged_chunk = merged_chunks.setdefault(
                    chunk_id,
                    {
                        **chunk,
                        "keyword_score": 0.0,
                        "keyword_channels": [],
                    },
                )
                merged_chunk["keyword_score"] += chunk.get("keyword_score", chunk.get("score", 0.0))
                merged_chunk["keyword_channels"].append(channel_name)

        ranked_chunks = sorted(merged_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            {
                **merged_chunks[chunk_id],
                "score": fused_score,
                "keyword_fused_score": fused_score,
            }
            for chunk_id, fused_score in ranked_chunks[:top_k]
        ]

    def _get_keyword_index(self) -> dict:
        document_count = self.collection.count() if self.collection is not None else 0
        if self.keyword_index and self.keyword_index.get("document_count") == document_count:
            return self.keyword_index

        documents = []
        document_frequency = Counter()
        total_length = 0

        if self.collection is not None and document_count > 0:
            collection_data = self.collection.get(include=["documents", "metadatas"])
            texts = collection_data.get("documents", []) or []
            metadatas = collection_data.get("metadatas", []) or []
            ids = collection_data.get("ids", []) or []

            for index, text in enumerate(texts):
                metadata = metadatas[index] or {}
                if ids:
                    metadata = {"id": ids[index], **metadata}

                full_text = f"{metadata_source_text(metadata)}\n{text}"
                tokens = _bm25_tokenize(full_text)
                token_counts = Counter(tokens)
                total_length += len(tokens)
                document_frequency.update(set(tokens))
                documents.append(
                    {
                        "id": metadata.get("id") or f"keyword-{index}",
                        "text": text,
                        "metadata": metadata,
                        "token_counts": token_counts,
                        "length": len(tokens),
                    }
                )

        self.keyword_index = {
            "document_count": document_count,
            "average_length": total_length / len(documents) if documents else 0.0,
            "document_frequency": document_frequency,
            "documents": documents,
        }
        return self.keyword_index

    def _expand_parent_chunks(self, chunks: list[dict], top_k: int) -> list[dict]:
        parents = self.parent_store.get("parents", {})
        expanded_chunks = []
        seen_parent_ids = set()

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            parent_id = metadata.get("parent_id")
            parent = parents.get(parent_id) if parent_id else None

            if parent:
                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)

                parent_metadata = parent.get("metadata", {})
                expanded_chunks.append(
                    {
                        **chunk,
                        "text": parent.get("text", chunk.get("text", "")),
                        "child_text": chunk.get("text", ""),
                        "metadata": {
                            **metadata,
                            **parent_metadata,
                            "matched_child_chunk_index": metadata.get("chunk_index", ""),
                        },
                    }
                )
            else:
                expanded_chunks.append(chunk)

            if len(expanded_chunks) >= top_k:
                break

        return expanded_chunks

    def _load_parent_store(self) -> dict:
        parent_store_path = self.persist_dir / PARENT_STORE_FILE
        if not parent_store_path.exists():
            return {"parents": {}}
        try:
            return json.loads(parent_store_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"parents": {}}

    @staticmethod
    def _rrf_fuse(vector_chunks: list[dict], keyword_chunks: list[dict], top_k: int) -> list[dict]:
        fused_scores = defaultdict(float)
        merged_chunks = {}

        for result_list, source_name in ((vector_chunks, "vector"), (keyword_chunks, "keyword")):
            for rank, chunk in enumerate(result_list, start=1):
                chunk_id = chunk_identity(chunk)
                fused_scores[chunk_id] += 1 / (RRF_K + rank)
                merged_chunk = merged_chunks.setdefault(chunk_id, {**chunk, "retrieval_channels": []})
                merged_chunk["retrieval_channels"].append(source_name)
                if source_name == "vector":
                    merged_chunk["vector_score"] = chunk.get("score", 0.0)
                else:
                    merged_chunk["keyword_score"] = chunk.get("keyword_score", chunk.get("score", 0.0))

        ranked_chunks = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            {
                **merged_chunks[chunk_id],
                "score": fused_score,
                "hybrid_score": fused_score,
            }
            for chunk_id, fused_score in ranked_chunks[:top_k]
        ]


def chunk_identity(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    return str(
        metadata.get("id")
        or (
            metadata.get("source"),
            metadata.get("chapter"),
            metadata.get("start_page"),
            metadata.get("end_page"),
            metadata.get("chunk_index"),
        )
    )


def metadata_source_text(metadata: dict) -> str:
    return " ".join(
        str(metadata.get(key, ""))
        for key in ("filename", "source", "chapter")
    )


def unique_queries(queries: list[str]) -> list[str]:
    seen = set()
    result = []
    for query in queries:
        normalized = str(query).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
