from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.langchain_zhipu import ZhipuChatModel
from src.zhipu_llm import DEFAULT_SYSTEM_PROMPT, MAX_HISTORY_MESSAGES


class RAGAgent:
    MAX_RETRIEVAL_HISTORY_MESSAGES = 4
    FOLLOW_UP_MARKERS = ("他", "它", "其", "这个", "那个", "上述", "前面", "刚才")

    def __init__(self, retriever):
        self.retriever = retriever
        self.chain = self._build_chain()

    def has_knowledge_base(self) -> bool:
        return self.retriever.has_documents()

    def answer(self, question: str, history: list[dict] | None = None) -> str:
        setup_error = getattr(self.retriever, "setup_error", "")
        if setup_error:
            return setup_error

        if not self.has_knowledge_base():
            return (
                "当前还没有加载到 Chroma 向量索引。请先把 txt、pdf、docx 或 pptx 文件放入 data/ 文件夹，"
                "然后运行 python build_index.py 构建索引。"
            )

        try:
            resolved_question = self._resolve_question(question, history)
            retrieved_chunks = self.retriever.retrieve(resolved_question, top_k=3)
        except RuntimeError as exc:
            return str(exc)

        if not retrieved_chunks:
            return "我没有在当前知识库中检索到相关内容，可以换个问法试试。"

        context = self._format_context(retrieved_chunks)
        return self.chain.invoke(
            {
                "chat_history": self._normalize_history(history),
                "context": context,
                "question": resolved_question,
            }
        )

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            context_parts.append(f"[资料 {index}] 来源：{source}\n{chunk['text']}")
        return "\n\n".join(context_parts)

    @staticmethod
    def _normalize_history(history: list[dict] | None) -> list[BaseMessage]:
        if not history:
            return []

        messages = []
        for item in history[-MAX_HISTORY_MESSAGES:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if not content:
                continue

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        return messages

    @classmethod
    def _resolve_question(cls, question: str, history: list[dict] | None) -> str:
        if not history:
            return question

        recent_user_messages = []
        for item in history:
            if not isinstance(item, dict):
                continue
            if item.get("role") != "user":
                continue

            content = str(item.get("content", "")).strip()
            if content:
                recent_user_messages.append(content)

        if not recent_user_messages:
            return question

        if cls._is_follow_up(question):
            last_topic = recent_user_messages[-1]
            return f"上一轮用户问题：{last_topic}\n当前追问：{question}\n请只回答当前追问中指代的对象。"

        context = "\n".join(recent_user_messages[-cls.MAX_RETRIEVAL_HISTORY_MESSAGES :])
        return f"{context}\n{question}"

    @classmethod
    def _is_follow_up(cls, question: str) -> bool:
        return any(marker in question for marker in cls.FOLLOW_UP_MARKERS)

    @staticmethod
    def _build_chain():
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DEFAULT_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    """请优先根据【知识库资料】回答用户问题。
如果资料中没有答案，请明确说明“知识库中没有找到相关信息”，不要编造。
如果用户问题是追问，请结合历史对话解析“他/它/其/这个”等指代，只回答该指代对象。
不要因为检索资料中出现了其他相关概念，就额外罗列未被问到的定义、公式或分类。

【知识库资料】
{context}

【用户问题】
{question}
""",
                ),
            ]
        )
        return prompt | ZhipuChatModel() | StrOutputParser()
