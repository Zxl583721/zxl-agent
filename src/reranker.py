import math
import re
from collections import Counter


LATIN_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


class LightweightReranker:
    """Second-stage reranker for retrieved knowledge chunks.

    The retriever still does broad semantic recall. This reranker adds a
    deterministic lexical signal that works well for course terms, formulas,
    chapter titles, and abbreviations.
    """

    def __init__(
        self,
        vector_weight: float = 0.20,
        lexical_weight: float = 0.65,
        metadata_weight: float = 0.15,
    ):
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self.metadata_weight = metadata_weight

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        vector_scores = normalize_scores([float(chunk.get("score", 0.0)) for chunk in chunks])
        scored_chunks = []

        for index, chunk in enumerate(chunks):
            lexical_score = cosine_similarity(query_tokens, tokenize(chunk.get("text", "")))
            metadata_score = self._metadata_match_score(query_tokens, chunk.get("metadata", {}))
            rerank_score = (
                self.vector_weight * vector_scores[index]
                + self.lexical_weight * lexical_score
                + self.metadata_weight * metadata_score
            )
            scored_chunks.append(
                (
                    rerank_score,
                    {
                        **chunk,
                        "retrieval_rank": index + 1,
                        "rerank_score": rerank_score,
                    },
                )
            )

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    @staticmethod
    def _metadata_match_score(query_tokens: Counter, metadata: dict) -> float:
        metadata_text = " ".join(
            str(value)
            for key, value in metadata.items()
            if key in {"filename", "source", "chapter"}
        )
        return cosine_similarity(query_tokens, tokenize(metadata_text))


def tokenize(text: str) -> Counter:
    """Tokenize mixed Chinese/English text into a simple bag of terms."""
    normalized = str(text).lower()
    tokens = LATIN_TOKEN_PATTERN.findall(normalized)
    tokens.extend(CJK_PATTERN.findall(normalized))
    return Counter(tokens)


def cosine_similarity(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0

    common_tokens = set(left) & set(right)
    dot_product = sum(left[token] * right[token] for token in common_tokens)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]
