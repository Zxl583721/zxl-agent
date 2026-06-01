import math
import re
from collections import Counter
from pathlib import Path


LATIN_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
OUTLINE_SOURCE_MARKERS = ("前言", "课程安排", "参考及要求")
COURSE_SOURCE_HINTS = (
    (("雷达方程", "原理", "组成", "历史"), "第1讲"),
    (("噪声", "信号检测", "门限", "虚警"), "第2讲"),
    (("测距", "测速", "测角", "精度"), "第3讲"),
    (("杂波",), "第4讲"),
    (("传播", "损耗"), "第5讲"),
    (("天线", "相控阵", "波束", "扫描"), "第6讲"),
    (("发射", "接收"), "第7讲"),
    (("试验", "测量"), "第8讲"),
    (("对抗", "干扰"), "第9讲"),
)


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
        source_hint_weight: float = 0.20,
        outline_penalty: float = 0.20,
    ):
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self.metadata_weight = metadata_weight
        self.source_hint_weight = source_hint_weight
        self.outline_penalty = outline_penalty

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        vector_scores = normalize_scores([float(chunk.get("score", 0.0)) for chunk in chunks])
        scored_chunks = []

        for index, chunk in enumerate(chunks):
            lexical_score = cosine_similarity(query_tokens, tokenize(chunk.get("text", "")))
            metadata_score = self._metadata_match_score(query_tokens, chunk.get("metadata", {}))
            source_hint_score = self._source_hint_score(query, chunk.get("metadata", {}))
            outline_penalty = self._outline_penalty(chunk.get("metadata", {}))
            rerank_score = (
                self.vector_weight * vector_scores[index]
                + self.lexical_weight * lexical_score
                + self.metadata_weight * metadata_score
                + self.source_hint_weight * source_hint_score
                - outline_penalty
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

    def _source_hint_score(self, query: str, metadata: dict) -> float:
        source_text = metadata_source_text(metadata)
        for query_markers, source_marker in COURSE_SOURCE_HINTS:
            if source_marker in source_text and any(marker in query for marker in query_markers):
                return 1.0
        return 0.0

    def _outline_penalty(self, metadata: dict) -> float:
        source_text = metadata_source_text(metadata)
        if any(marker in source_text for marker in OUTLINE_SOURCE_MARKERS):
            return self.outline_penalty
        return 0.0


class BGEReranker:
    """Cross-encoder reranker backed by BAAI/bge-reranker-v2-m3."""

    def __init__(
        self,
        model_path: str | Path = "models/bge-reranker-v2-m3",
        *,
        use_fp16: bool = False,
        batch_size: int = 8,
    ):
        self.model_path = str(model_path)
        self.batch_size = batch_size
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "未安装 FlagEmbedding。请运行 ./.conda/bin/python -m pip install FlagEmbedding transformers torch"
            ) from exc

        if not Path(self.model_path).exists():
            raise RuntimeError(f"BGE reranker 模型目录不存在：{self.model_path}")

        self.model = FlagReranker(self.model_path, use_fp16=use_fp16)

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks or top_k <= 0:
            return []

        pairs = [[query, chunk.get("text", "")] for chunk in chunks]
        scores = self.model.compute_score(pairs, batch_size=self.batch_size)
        if isinstance(scores, (float, int)):
            scores = [float(scores)]

        scored_chunks = []
        for index, (chunk, score) in enumerate(zip(chunks, scores), start=1):
            rerank_score = float(score)
            scored_chunks.append(
                (
                    rerank_score,
                    {
                        **chunk,
                        "retrieval_rank": index,
                        "bge_rerank_score": rerank_score,
                    },
                )
            )

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]


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


def metadata_source_text(metadata: dict) -> str:
    return " ".join(
        str(metadata.get(key, ""))
        for key in ("filename", "source", "chapter")
    )
