import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_agent import RAGAgent  # noqa: E402
from src.retriever import ChromaRetriever  # noqa: E402


DEFAULT_EVAL_FILE = Path(__file__).with_name("qa_eval.json")
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"


def run_eval(eval_file: Path, answer: bool, disable_rewrite: bool) -> int:
    cases = load_cases(eval_file)
    agent = RAGAgent(ChromaRetriever(VECTOR_STORE_DIR))

    setup_error = getattr(agent.retriever, "setup_error", "")
    if setup_error:
        print(f"评估失败：{setup_error}")
        return 1
    if not agent.has_knowledge_base():
        print("评估失败：还没有可用的 Chroma 索引，请先运行 python build_index.py。")
        return 1

    results = []
    for case in cases:
        result = evaluate_case(agent, case, answer=answer, disable_rewrite=disable_rewrite)
        results.append(result)
        print_case_result(result, answer=answer)

    print_summary(results, answer=answer)
    return 0


def evaluate_case(agent: RAGAgent, case: dict, answer: bool, disable_rewrite: bool) -> dict:
    question = str(case["question"]).strip()
    history = case.get("history") if isinstance(case.get("history"), list) else []
    expected_sources = [str(item) for item in case.get("expected_sources", [])]
    expected_keywords = [str(item) for item in case.get("expected_keywords", [])]

    retrieval_question = (
        agent._resolve_question(question, history)
        if disable_rewrite
        else agent._rewrite_retrieval_question(question, history)
    )
    rewrite_triggered = (
        bool(history)
        and not disable_rewrite
        and agent._should_rewrite_question(question, history)
    )
    retrieved_chunks = agent.retriever.retrieve(retrieval_question, top_k=agent.CANDIDATE_CHUNKS)
    reranked_chunks = agent.reranker.rerank(
        retrieval_question,
        retrieved_chunks,
        top_k=agent.FINAL_CHUNKS,
    )
    retrieval_sources = agent._format_sources(reranked_chunks)
    retrieval_source_hit = source_hit(retrieval_sources, expected_sources)

    answer_text = ""
    answer_sources = []
    answer_source_hit = None
    keyword_coverage = None

    if answer:
        output = agent.answer_with_sources(question, history=history)
        answer_text = output.get("answer", "")
        answer_sources = output.get("sources", [])
        answer_source_hit = source_hit(answer_sources, expected_sources)
        keyword_coverage = keyword_hit_rate(answer_text, expected_keywords)

    return {
        "id": case.get("id", question),
        "question": question,
        "retrieval_question": retrieval_question,
        "rewrite_triggered": rewrite_triggered,
        "expected_sources": expected_sources,
        "retrieval_sources": retrieval_sources,
        "retrieval_source_hit": retrieval_source_hit,
        "answer": answer_text,
        "answer_sources": answer_sources,
        "answer_source_hit": answer_source_hit,
        "expected_keywords": expected_keywords,
        "keyword_coverage": keyword_coverage,
    }


def source_hit(sources: list[dict], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True

    source_text = "\n".join(
        " ".join(
            str(source.get(field, ""))
            for field in ("source", "chapter", "page_range")
        )
        for source in sources
    )
    return any(expected in source_text for expected in expected_sources)


def keyword_hit_rate(text: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    hits = sum(1 for keyword in expected_keywords if keyword in text)
    return hits / len(expected_keywords)


def print_case_result(result: dict, answer: bool) -> None:
    marker = "PASS" if result["retrieval_source_hit"] else "FAIL"
    print(f"\n[{marker}] {result['id']}")
    print(f"问题：{result['question']}")
    if result["retrieval_question"] != result["question"]:
        print(f"检索问题：{result['retrieval_question']}")
    print(f"触发改写：{'是' if result['rewrite_triggered'] else '否'}")
    print(f"期望来源：{', '.join(result['expected_sources']) or '未设置'}")
    print("检索来源：")
    for source in result["retrieval_sources"]:
        print(
            f"- {source.get('source', 'unknown')}；"
            f"章节：{source.get('chapter', '未识别章节')}；"
            f"页码：{source.get('page_range', '未知')}"
        )

    if answer:
        answer_hit = "PASS" if result["answer_source_hit"] else "FAIL"
        print(f"回答来源命中：{answer_hit}")
        print(f"关键词覆盖率：{result['keyword_coverage']:.2%}")


def print_summary(results: list[dict], answer: bool) -> None:
    total = len(results)
    retrieval_hits = sum(1 for result in results if result["retrieval_source_hit"])
    rewrite_count = sum(1 for result in results if result["rewrite_triggered"])
    print("\n评估汇总")
    print(f"- 用例数：{total}")
    print(f"- 检索来源命中率：{safe_rate(retrieval_hits, total):.2%}")
    print(f"- Query Rewrite 触发次数：{rewrite_count}")

    if answer:
        answer_hits = sum(1 for result in results if result["answer_source_hit"])
        avg_keyword_coverage = sum(result["keyword_coverage"] for result in results) / total if total else 0.0
        print(f"- 回答来源命中率：{safe_rate(answer_hits, total):.2%}")
        print(f"- 平均关键词覆盖率：{avg_keyword_coverage:.2%}")


def safe_rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def load_cases(eval_file: Path) -> list[dict]:
    cases = json.loads(eval_file.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("评估文件必须是 JSON 数组。")
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 RAG 检索和问答评估。")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="评估集 JSON 文件。")
    parser.add_argument("--answer", action="store_true", help="同时调用大模型生成回答并评估关键词覆盖。")
    parser.add_argument("--disable-rewrite", action="store_true", help="关闭 LLM 查询改写，仅使用规则式检索问题。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        run_eval(
            eval_file=args.eval_file,
            answer=args.answer,
            disable_rewrite=args.disable_rewrite,
        )
    )
