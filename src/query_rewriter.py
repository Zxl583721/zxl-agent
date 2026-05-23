from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_zhipu import ZhipuChatModel
from src.zhipu_llm import MAX_HISTORY_MESSAGES


class LLMQueryRewriter:
    """Rewrite conversational follow-up questions into standalone retrieval queries."""

    ERROR_PREFIXES = (
        "未检测到",
        "未安装",
        "调用智谱 AI 接口失败",
        "智谱 AI 返回",
    )

    def __init__(self):
        self.chain = self._build_chain()

    def rewrite(self, question: str, history: list[dict] | None, fallback_question: str) -> str:
        history_text = self._format_history(history)
        if not history_text:
            return question

        try:
            rewritten_question = self.chain.invoke(
                {
                    "history": history_text,
                    "question": question,
                }
            )
        except Exception:
            return fallback_question

        rewritten_question = self._clean_rewritten_question(rewritten_question)
        if not rewritten_question or rewritten_question.startswith(self.ERROR_PREFIXES):
            return fallback_question

        return rewritten_question

    @staticmethod
    def _format_history(history: list[dict] | None) -> str:
        if not history:
            return ""

        lines = []
        for item in history[-MAX_HISTORY_MESSAGES:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if not content:
                continue

            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                lines.append(f"助手：{content}")

        return "\n".join(lines)

    @staticmethod
    def _clean_rewritten_question(text: str) -> str:
        cleaned = str(text).strip()
        cleaned = cleaned.removeprefix("```text").removeprefix("```").removesuffix("```").strip()

        for prefix in ("改写后问题：", "独立问题：", "检索问题："):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()

        if len(cleaned) > 500:
            return ""
        return cleaned

    @staticmethod
    def _build_chain():
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 RAG 系统的查询改写模块。你的任务是把多轮对话中的当前问题改写成一个"
                    "适合知识库检索的独立中文问题。只输出改写后的问题，不要解释。",
                ),
                (
                    "user",
                    """请根据【历史对话】理解【当前问题】中的指代、省略和上下文，并输出一个独立检索问题。

要求：
1. 如果当前问题已经完整，不需要改写，则原样输出当前问题。
2. 如果当前问题包含“它、这个、上述、前面、这些”等指代，请结合历史对话补全指代对象。
3. 不要回答问题，不要输出分析过程，不要添加知识库中没有的信息。
4. 输出必须只有一行独立问题。

【历史对话】
{history}

【当前问题】
{question}
""",
                ),
            ]
        )
        return prompt | ZhipuChatModel(temperature=0.1, max_tokens=256) | StrOutputParser()
