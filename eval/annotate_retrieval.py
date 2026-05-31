import argparse
import json
from pathlib import Path

from eval_utils import PROJECT_ROOT, chunk_id, init_retriever, source_text
from src.query_expander import expand_keyword_queries


DEFAULT_VECTOR_DIR = PROJECT_ROOT / "vector_store"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索标注辅助：打印候选 chunk，便于复制 relevant_chunk_ids。")
    parser.add_argument("--query", required=True, help="要标注的查询。")
    parser.add_argument("--type", default="unknown", help="查询类型，如 term/formula/semantic/comparison。")
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--collection-name", default="")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    retriever = init_retriever(args.vector_dir, args.collection_name or None)
    chunks = retriever.retrieve(args.query, top_k=args.top_k, keyword_queries=expand_keyword_queries(args.query))

    print(f"Query: {args.query}\n")
    print(f"{'#':<3} {'score':>8}  source")
    print("-" * 110)
    for index, chunk in enumerate(chunks, start=1):
        preview = str(chunk.get("text", "")).replace("\n", " ")[:180]
        print(f"{index:<3} {float(chunk.get('score', 0.0)):>8.4f}  {source_text(chunk)[:80]}")
        print(f"    chunk_id: {chunk_id(chunk)}")
        print(f"    {preview}")

    template = {
        "query_id": "qXXX",
        "type": args.type,
        "query": args.query,
        "expected_sources": [],
        "relevant_chunk_ids": [chunk_id(chunk) for chunk in chunks[:3]],
    }
    print("\nJSONL 模板，人工删改 relevant_chunk_ids 后放入 eval/retrieval_cases.jsonl：")
    print(json.dumps(template, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
