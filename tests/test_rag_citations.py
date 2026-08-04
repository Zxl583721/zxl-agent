from src.rag_agent import RAGAgent


SOURCES = [
    {"source_id": 1, "source": "旧版手册.pdf"},
    {"source_id": 2, "source": "新版手册.pdf"},
]


def test_conflicting_answer_requires_citations_for_every_retrieved_source():
    answer = "资料存在冲突：旧版给出的结论是 A。[资料 1]"

    completed = RAGAgent._ensure_source_citations(answer, SOURCES)

    assert completed.endswith("冲突资料：[资料 1]、[资料 2]")
    status = RAGAgent._citation_status(completed, SOURCES)
    assert status["conflict_detected"]
    assert status["conflict_citation_complete"]
    assert status["valid_cited_source_ids"] == [1, 2]


def test_non_conflicting_answer_keeps_its_specific_citation():
    answer = "该设备支持模式 A。[资料 1]"

    assert RAGAgent._ensure_source_citations(answer, SOURCES) == answer


def test_conflict_prompt_requires_disclosure_and_source_scoped_citations():
    prompt = RAGAgent._build_chain().first.messages[-1].prompt.template

    assert "资料存在冲突" in prompt
    assert "紧邻引用双方资料" in prompt
    assert "笼统引用替代逐项引用" in prompt
