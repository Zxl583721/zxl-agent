import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from eval_utils import PROJECT_ROOT, case_relevant_ids, chunk_id, init_retriever, load_json_or_jsonl
from metrics import recall_at_k, relative_gain, safe_mean
from src.query_expander import expand_keyword_queries
from src.query_rewriter import LLMQueryRewriter
from src.rag_agent import RAGAgent


DEFAULT_QUERIES = Path(__file__).with_name("multiturn_cases.jsonl")
DEFAULT_VECTOR_DIR = PROJECT_ROOT / "vector_store"


def similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def retrieve_ids(retriever, query: str, top_k: int) -> tuple[list[str], list[dict]]:
    chunks = retriever.retrieve(query, top_k=top_k, keyword_queries=expand_keyword_queries(query))
    return [chunk_id(chunk) for chunk in chunks], chunks


def evaluate_multiturn(retriever, cases: list[dict], top_k: int, use_llm_rewriter: bool) -> list[dict]:
    rewriter = LLMQueryRewriter() if use_llm_rewriter else None
    rows = []

    for case in cases:
        raw_question = str(case["current_question"]).strip()
        history = case.get("history") if isinstance(case.get("history"), list) else []
        expected = str(case.get("expected_standalone", "")).strip()

        raw_ids, raw_chunks = retrieve_ids(retriever, raw_question, top_k)
        raw_relevant_ids = case_relevant_ids(case, raw_chunks)
        raw_recall = recall_at_k(raw_ids, raw_relevant_ids, top_k)

        fallback = RAGAgent._resolve_question(raw_question, history)
        if use_llm_rewriter:
            rewritten = rewriter.rewrite(raw_question, history, fallback)
        else:
            rewritten = fallback

        rewritten_ids, rewritten_chunks = retrieve_ids(retriever, rewritten, top_k)
        rewritten_relevant_ids = case_relevant_ids(case, rewritten_chunks)
        relevant_ids = raw_relevant_ids or rewritten_relevant_ids
        rewritten_recall = recall_at_k(rewritten_ids, relevant_ids, top_k)

        rows.append(
            {
                "id": case.get("dialogue_id", raw_question),
                "type": case.get("query_type", "unknown"),
                "raw_question": raw_question,
                "rewritten_question": rewritten,
                "expected_standalone": expected,
                "rewrite_changed": rewritten.strip() != raw_question.strip(),
                "rewrite_similarity": similarity(rewritten, expected) if expected else 0.0,
                "raw_recall": raw_recall,
                "rewritten_recall": rewritten_recall,
                "recall_gain_pct": relative_gain(raw_recall, rewritten_recall),
                "raw_ids": raw_ids,
                "rewritten_ids": rewritten_ids,
                "relevant_ids": sorted(relevant_ids),
                "label_level": "chunk" if case.get("relevant_chunk_ids") else "source",
            }
        )
    return rows


def print_summary(rows: list[dict], top_k: int, verbose: bool) -> None:
    print(f"\n多轮 Query Rewriter 评测 Top-{top_k}")
    print("=" * 96)
    print(f"{'ID':<24} {'类型':<12} {'Raw R':>8} {'Rewrite R':>10} {'Gain':>9} {'Changed':>8} {'Sim':>7}")
    print("-" * 96)
    for row in rows:
        print(
            f"{row['id']:<24} {row['type']:<12} "
            f"{row['raw_recall']:>7.2%} {row['rewritten_recall']:>9.2%} "
            f"{row['recall_gain_pct']:>+8.1f}% {str(row['rewrite_changed']):>8} "
            f"{row['rewrite_similarity']:>7.2%}"
        )

    print("-" * 96)
    print(f"平均原始 Recall@{top_k}: {safe_mean([row['raw_recall'] for row in rows]):.2%}")
    print(f"平均改写 Recall@{top_k}: {safe_mean([row['rewritten_recall'] for row in rows]):.2%}")
    print(f"平均召回相对增益: {safe_mean([row['recall_gain_pct'] for row in rows]):+.1f}%")
    print(f"改写触发/变化率: {safe_mean([1.0 if row['rewrite_changed'] else 0.0 for row in rows]):.2%}")
    expected_rows = [row for row in rows if row["expected_standalone"]]
    if expected_rows:
        print(f"与期望 standalone 平均相似度: {safe_mean([row['rewrite_similarity'] for row in expected_rows]):.2%}")

    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    print("\n按指代类型")
    print("-" * 60)
    for query_type, items in sorted(by_type.items()):
        print(
            f"{query_type:<16} n={len(items):<3} "
            f"Raw={safe_mean([item['raw_recall'] for item in items]):.2%} "
            f"Rewrite={safe_mean([item['rewritten_recall'] for item in items]):.2%} "
            f"Gain={safe_mean([item['recall_gain_pct'] for item in items]):+.1f}%"
        )

    if verbose:
        print("\n改写详情")
        print("-" * 60)
        for row in rows:
            print(f"[{row['id']}] {row['type']} label={row['label_level']}")
            print(f"  raw:      {row['raw_question']}")
            print(f"  rewrite:  {row['rewritten_question']}")
            if row["expected_standalone"]:
                print(f"  expected: {row['expected_standalone']}")
            print(f"  relevant: {row['relevant_ids']}")
            print(f"  raw ids:  {row['raw_ids'][:top_k]}")
            print(f"  rw ids:   {row['rewritten_ids'][:top_k]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多轮追问 Query Rewriter 评测。")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="多轮评测集 JSON/JSONL。")
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR, help="Chroma 向量库目录。")
    parser.add_argument("--collection-name", default="", help="Chroma collection 名称，默认使用项目配置。")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rule-only", action="store_true", help="不用 LLM，只评估 RAGAgent 的规则 fallback 改写。")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_json_or_jsonl(args.queries)
    if not cases:
        print(f"评估失败：{args.queries} 为空。")
        return 1

    retriever = init_retriever(args.vector_dir, args.collection_name or None)
    rows = evaluate_multiturn(
        retriever,
        cases,
        top_k=args.top_k,
        use_llm_rewriter=not args.rule_only,
    )
    print_summary(rows, top_k=args.top_k, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
