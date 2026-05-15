from collections import Counter
import math
import re


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")


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


# TODO: 后续可新增 VectorRetriever：
# 1. 调用 src.embedding.embed_texts 生成向量。
# 2. 使用 FAISS/Chroma 持久化到 vector_store/。
# 3. 查询时对用户问题向量化并返回最相似的 chunks。

