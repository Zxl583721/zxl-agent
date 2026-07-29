from src.rag_agent import RAGAgent


HISTORY = [{"role": "user", "content": "请介绍相控阵雷达。"}]


def test_rewrite_requires_history():
    assert not RAGAgent._should_rewrite_question("这个系统有什么优点？", [])


def test_rewrite_detects_explicit_reference():
    assert RAGAgent._should_rewrite_question("这个系统有什么优点？", HISTORY)


def test_short_complete_question_does_not_trigger_rewrite():
    assert not RAGAgent._should_rewrite_question("雷达的工作原理是什么？", HISTORY)


def test_other_is_not_mistaken_for_pronoun():
    assert not RAGAgent._should_rewrite_question("其他方法有什么优点？", HISTORY)


def test_elliptical_question_still_triggers_rewrite():
    assert RAGAgent._should_rewrite_question("原理呢？", HISTORY)


def test_rewrite_result_distinguishes_attempt_from_change():
    class UnchangedRewriter:
        @staticmethod
        def rewrite(**kwargs):
            return kwargs["question"]

    agent = object.__new__(RAGAgent)
    agent.query_rewriter = UnchangedRewriter()

    result = agent._prepare_retrieval_question("原理呢？", HISTORY)

    assert result.rewrite_attempted
    assert not result.rewrite_applied
