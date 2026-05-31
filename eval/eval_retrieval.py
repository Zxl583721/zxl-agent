import argparse
from collections import defaultdict
from pathlib import Path
import time

from eval_utils import PROJECT_ROOT, case_relevant_ids, chunk_id, init_retriever, load_json_or_jsonl
from metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k, safe_mean
from src.query_expander import expand_keyword_queries
from src.reranker import LightweightReranker


DEFAULT_QUERIES = Path(__file__).with_name("retrieval_cases.jsonl")
DEFAULT_VECTOR_DIR = PROJECT_ROOT / "vector_store"


CONFIGS = (
    ("vector", "纯向量"),
    ("hybrid", "向量+BM25+RRF"),
    ("hybrid_expanded", "向量+BM25+RRF+QueryExp"),
    ("full", "向量+BM25+RRF+QueryExp+Rerank"),
)


def retrieve_by_config(retriever, query: str, config: str, top_k: int, candidate_k: int) -> list[dict]:
    if config == "vector":
        return retriever._vector_retrieve(query, top_k)
    if config == "hybrid":
        return retriever.retrieve(query, top_k=top_k, keyword_queries=[query])
    if config == "hybrid_expanded":
        return retriever.retrieve(query, top_k=top_k, keyword_queries=expand_keyword_queries(query))
    if config == "full":
        chunks = retriever.retrieve(
            query,
            top_k=max(candidate_k, top_k),
            keyword_queries=expand_keyword_queries(query),
        )
        return LightweightReranker().rerank(query, chunks, top_k=top_k)
    raise ValueError(f"未知配置：{config}")


def evaluate_config(retriever, cases: list[dict], config: str, top_k: int, candidate_k: int) -> dict:
    rows = []
    elapsed_ms = []

    for case in cases:
        query = str(case["query"]).strip()
        started = time.perf_counter()
        chunks = retrieve_by_config(retriever, query, config, top_k=top_k, candidate_k=candidate_k)
        elapsed_ms.append((time.perf_counter() - started) * 1000)

        retrieved_ids = [chunk_id(chunk) for chunk in chunks]
        relevant_ids = case_relevant_ids(case, chunks)
        rows.append(
            {
                "id": case.get("query_id", query),
                "type": case.get("type", "unknown"),
                "query": query,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": sorted(relevant_ids),
                "recall": recall_at_k(retrieved_ids, relevant_ids, top_k),
                "precision": precision_at_k(retrieved_ids, relevant_ids, top_k),
                "mrr": mrr_at_k(retrieved_ids, relevant_ids, top_k),
                "ndcg": ndcg_at_k(retrieved_ids, relevant_ids, top_k),
                "latency_ms": elapsed_ms[-1],
                "label_level": "chunk" if case.get("relevant_chunk_ids") else "source",
            }
        )

    return summarize(config, rows, elapsed_ms)


def summarize(config: str, rows: list[dict], elapsed_ms: list[float]) -> dict:
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)

    return {
        "config": config,
        "n": len(rows),
        "recall": safe_mean([row["recall"] for row in rows]),
        "precision": safe_mean([row["precision"] for row in rows]),
        "mrr": safe_mean([row["mrr"] for row in rows]),
        "ndcg": safe_mean([row["ndcg"] for row in rows]),
        "latency_ms": safe_mean(elapsed_ms),
        "rows": rows,
        "by_type": {
            query_type: {
                "n": len(items),
                "recall": safe_mean([item["recall"] for item in items]),
                "mrr": safe_mean([item["mrr"] for item in items]),
                "ndcg": safe_mean([item["ndcg"] for item in items]),
            }
            for query_type, items in sorted(by_type.items())
        },
    }


def print_summary(results: list[dict], top_k: int, verbose: bool) -> None:
    print(f"\n检索评测汇总 Top-{top_k}")
    print("=" * 86)
    print(f"{'配置':<32} {'N':>4} {'Recall':>9} {'Precision':>10} {'MRR':>9} {'NDCG':>9} {'Avg ms':>9}")
    print("-" * 86)
    for result in results:
        label = dict(CONFIGS)[result["config"]]
        print(
            f"{label:<32} {result['n']:>4} "
            f"{result['recall']:>8.2%} {result['precision']:>9.2%} "
            f"{result['mrr']:>8.2%} {result['ndcg']:>8.2%} {result['latency_ms']:>9.1f}"
        )

    full = next((item for item in results if item["config"] == "full"), None)
    if full:
        print("\nFull 配置按问题类型")
        print("-" * 60)
        for query_type, item in full["by_type"].items():
            print(
                f"{query_type:<20} n={item['n']:<3} "
                f"Recall={item['recall']:.2%} MRR={item['mrr']:.2%} NDCG={item['ndcg']:.2%}"
            )

    if verbose:
        print("\n未命中或低召回样例")
        print("-" * 60)
        for result in results:
            if result["config"] != "full":
                continue
            for row in result["rows"]:
                if row["recall"] < 1.0:
                    print(f"[{row['id']}] Recall={row['recall']:.2%} label={row['label_level']} query={row['query']}")
                    print(f"  relevant: {row['relevant_ids']}")
                    print(f"  retrieved: {row['retrieved_ids'][:top_k]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 检索消融评测：纯向量、混合检索、QueryExp、Rerank。")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="检索评测集 JSON/JSONL。")
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR, help="Chroma 向量库目录。")
    parser.add_argument("--collection-name", default="", help="Chroma collection 名称，默认使用项目配置。")
    parser.add_argument("--top-k", type=int, default=5, help="计算 Recall/MRR/NDCG 的 K。")
    parser.add_argument("--candidate-k", type=int, default=20, help="Reranker 前召回候选数。")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印低召回样例。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_json_or_jsonl(args.queries)
    if not cases:
        print(f"评估失败：{args.queries} 为空。")
        return 1

    retriever = init_retriever(args.vector_dir, args.collection_name or None)
    results = [
        evaluate_config(retriever, cases, config, top_k=args.top_k, candidate_k=args.candidate_k)
        for config, _ in CONFIGS
    ]
    print_summary(results, top_k=args.top_k, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
