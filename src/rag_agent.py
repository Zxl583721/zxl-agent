import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.model_provider import get_chat_model
from src.query_expander import expand_keyword_queries
from src.query_rewriter import LLMQueryRewriter
from src.reranker import LightweightReranker
from src.zhipu_llm import DEFAULT_SYSTEM_PROMPT, MAX_HISTORY_MESSAGES


class RAGAgent:
    MAX_RETRIEVAL_HISTORY_MESSAGES = 4
    CANDIDATE_CHUNKS = 20
    FINAL_CHUNKS = 3
    FOLLOW_UP_MARKERS = ("他", "它", "其", "这个", "那个", "上述", "前面", "刚才")
    SOURCE_CITATION_PATTERN = re.compile(r"\[资料\s*(\d+)\]")
    CONTEXT_DEPENDENT_PATTERNS = (
        "有哪些影响",
        "有哪些因素",
        "有什么影响",
        "为什么",
        "怎么",
        "如何",
        "区别",
        "联系",
        "优缺点",
    )

    def __init__(self, retriever):
        self.retriever = retriever
        self.query_rewriter = LLMQueryRewriter()
        self.reranker = LightweightReranker()
        self.chain = self._build_chain()
        self.general_chain = self._build_general_chain()

    def has_knowledge_base(self) -> bool:
        return self.retriever.has_documents()

    def answer(self, question: str, history: list[dict] | None = None) -> str:
        return self.answer_with_sources(question, history=history)["answer"]

    def answer_general(self, question: str, history: list[dict] | None = None) -> dict:
        answer = self.general_chain.invoke(
            {
                "chat_history": self._normalize_history(history),
                "question": question,
            }
        )
        return {"answer": answer, "sources": [], "mode": "general"}

    def stream_general(self, question: str, history: list[dict] | None = None):
        yield {"type": "metadata", "sources": [], "mode": "general"}
        for token in self.general_chain.stream(
            {
                "chat_history": self._normalize_history(history),
                "question": question,
            }
        ):
            if token:
                yield {"type": "token", "content": str(token)}

    def answer_with_sources(self, question: str, history: list[dict] | None = None) -> dict:
        setup_error = getattr(self.retriever, "setup_error", "")
        if setup_error:
            return {"answer": setup_error, "sources": []}

        if not self.has_knowledge_base():
            return {
                "answer": (
                    "当前还没有加载到 Chroma 向量索引。请先把 txt、pdf、docx 或 pptx 文件放入 data/ 文件夹，"
                    "然后运行 python build_index.py 构建索引。"
                ),
                "sources": [],
            }

        try:
            rewrite_triggered = self._should_rewrite_question(question, history)
            retrieval_question = self._rewrite_retrieval_question(question, history)
            retrieved_chunks = self.retriever.retrieve(
                retrieval_question,
                top_k=self.CANDIDATE_CHUNKS,
                keyword_queries=expand_keyword_queries(retrieval_question),
            )
        except RuntimeError as exc:
            return {"answer": str(exc), "sources": []}

        if not retrieved_chunks:
            return {"answer": "我没有在当前知识库中检索到相关内容，可以换个问法试试。", "sources": []}

        reranked_chunks = self.reranker.rerank(
            retrieval_question,
            retrieved_chunks,
            top_k=self.FINAL_CHUNKS,
        )
        context = self._format_context(reranked_chunks)
        answer = self.chain.invoke(
            {
                "chat_history": self._normalize_history(history),
                "context": context,
                "question": question,
            }
        )
        sources = self._format_sources(reranked_chunks)
        answer = self._ensure_source_citations(answer, sources)
        return {
            "answer": answer,
            "sources": sources,
            "citation_status": self._citation_status(answer, sources),
            "retrieval_question": retrieval_question,
            "rewrite_triggered": rewrite_triggered,
        }

    def stream_with_sources(self, question: str, history: list[dict] | None = None):
        setup_error = getattr(self.retriever, "setup_error", "")
        if setup_error:
            yield {"type": "metadata", "sources": [], "mode": "knowledge"}
            yield {"type": "token", "content": setup_error}
            return

        if not self.has_knowledge_base():
            yield {"type": "metadata", "sources": [], "mode": "knowledge"}
            yield {
                "type": "token",
                "content": (
                    "当前还没有加载到 Chroma 向量索引。请先把 txt、pdf、docx 或 pptx 文件放入 data/ 文件夹，"
                    "然后运行 python build_index.py 构建索引。"
                ),
            }
            return

        try:
            rewrite_triggered = self._should_rewrite_question(question, history)
            retrieval_question = self._rewrite_retrieval_question(question, history)
            retrieved_chunks = self.retriever.retrieve(
                retrieval_question,
                top_k=self.CANDIDATE_CHUNKS,
                keyword_queries=expand_keyword_queries(retrieval_question),
            )
        except RuntimeError as exc:
            yield {"type": "metadata", "sources": [], "mode": "knowledge"}
            yield {"type": "token", "content": str(exc)}
            return

        if not retrieved_chunks:
            yield {"type": "metadata", "sources": [], "mode": "knowledge"}
            yield {"type": "token", "content": "我没有在当前知识库中检索到相关内容，可以换个问法试试。"}
            return

        reranked_chunks = self.reranker.rerank(
            retrieval_question,
            retrieved_chunks,
            top_k=self.FINAL_CHUNKS,
        )
        sources = self._format_sources(reranked_chunks)
        yield {
            "type": "metadata",
            "sources": sources,
            "mode": "knowledge",
            "retrieval_question": retrieval_question,
            "rewrite_triggered": rewrite_triggered,
        }

        for token in self.chain.stream(
            {
                "chat_history": self._normalize_history(history),
                "context": self._format_context(reranked_chunks),
                "question": question,
            }
        ):
            if token:
                yield {"type": "token", "content": str(token)}

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("filename") or metadata.get("source", "unknown")
            chapter = metadata.get("chapter", "未识别章节")
            page_range = _format_page_range(metadata.get("start_page"), metadata.get("end_page"))
            context_parts.append(
                f"[资料 {index}]\n"
                f"来源：{source}\n"
                f"章节：{chapter}\n"
                f"页码：{page_range}\n"
                f"内容：{chunk['text']}"
            )
        return "\n\n".join(context_parts)

    @staticmethod
    def _format_sources(chunks: list[dict]) -> list[dict]:
        sources = []
        seen = set()

        for source_index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("filename") or metadata.get("source", "unknown")
            chapter = metadata.get("chapter", "未识别章节")
            start_page = metadata.get("start_page") or ""
            end_page = metadata.get("end_page") or start_page
            key = (source, chapter, start_page, end_page)

            if key in seen:
                continue

            seen.add(key)
            sources.append(
                {
                    "source_id": source_index,
                    "source": source,
                    "chapter": chapter,
                    "start_page": start_page,
                    "end_page": end_page,
                    "page_range": _format_page_range(start_page, end_page),
                }
            )

        return sources

    @classmethod
    def _ensure_source_citations(cls, answer: str, sources: list[dict]) -> str:
        answer = str(answer).strip()
        if not answer or not sources or "知识库中没有找到相关信息" in answer:
            return answer

        status = cls._citation_status(answer, sources)
        if status["valid_citation_count"] > 0:
            return answer

        fallback_citations = "、".join(
            f"[资料 {source['source_id']}]"
            for source in sources[: cls.FINAL_CHUNKS]
            if source.get("source_id")
        )
        if not fallback_citations:
            return answer

        return f"{answer}\n\n依据：{fallback_citations}"

    @classmethod
    def _citation_status(cls, answer: str, sources: list[dict]) -> dict:
        valid_source_ids = {
            int(source["source_id"])
            for source in sources
            if str(source.get("source_id", "")).isdigit()
        }
        cited_source_ids = [
            int(match)
            for match in cls.SOURCE_CITATION_PATTERN.findall(str(answer))
            if match.isdigit()
        ]
        valid_cited_source_ids = sorted(
            source_id
            for source_id in set(cited_source_ids)
            if source_id in valid_source_ids
        )

        return {
            "has_citation": bool(cited_source_ids),
            "valid_citation_count": len(valid_cited_source_ids),
            "cited_source_ids": cited_source_ids,
            "valid_cited_source_ids": valid_cited_source_ids,
        }

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

    def _rewrite_retrieval_question(self, question: str, history: list[dict] | None) -> str:
        if not self._should_rewrite_question(question, history):
            return question

        fallback_question = self._resolve_question(question, history)
        return self.query_rewriter.rewrite(
            question=question,
            history=history,
            fallback_question=fallback_question,
        )

    @classmethod
    def _is_follow_up(cls, question: str) -> bool:
        return any(marker in question for marker in cls.FOLLOW_UP_MARKERS)

    @classmethod
    def _should_rewrite_question(cls, question: str, history: list[dict] | None) -> bool:
        if not history:
            return False

        stripped_question = question.strip()
        if not stripped_question:
            return False
        if cls._is_follow_up(stripped_question):
            return True

        # Short elliptical questions often depend on the previous topic.
        if len(stripped_question) <= 12:
            return True

        return any(pattern in stripped_question for pattern in cls.CONTEXT_DEPENDENT_PATTERNS)

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
回答中如果引用具体资料，请优先结合资料的来源、章节和页码判断上下文。
回答正文中需要引用资料时，只能使用【知识库资料】里已有的资料编号，例如 [资料 1]、[资料 2]。
不要在回答正文中编造、改写或额外生成参考来源名称、章节或页码；参考来源会由系统根据检索结果单独展示。
如果用户问题是追问，请结合历史对话解析“他/它/其/这个”等指代，只回答该指代对象。
历史对话只用于理解当前问题的指代和上下文，不要复述、总结或回答历史问题。
最终回答必须聚焦【用户问题】中的当前问题，不要把前几轮的问题或答案混入回答。
不要因为检索资料中出现了其他相关概念，就额外罗列未被问到的定义、公式或分类。
如果需要输出数学公式，请使用 Markdown 可渲染的数学格式：行内公式用 `$...$`，独立公式用 `$$...$$`。
不要把公式放进 ```latex``` 或其他代码块中，也不要把 LaTeX 当作普通源码文本输出。

【知识库资料】
{context}

【用户问题】
{question}
""",
                ),
            ]
        )
        return prompt | get_chat_model() | StrOutputParser()

    @staticmethod
    def _build_general_chain():
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是 zxl-agent 项目的中文通用聊天助手。
你可以根据自身通用知识回答问题，不需要检索知识库。
如果问题涉及不确定、实时变化或高风险内容，请说明不确定性，并建议用户核验最新或专业信息。
默认使用中文回答，表达清晰、直接。""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{question}"),
            ]
        )
        return prompt | get_chat_model() | StrOutputParser()


def _format_page_range(start_page, end_page) -> str:
    if not start_page and not end_page:
        return "未知"
    if not end_page or start_page == end_page:
        return str(start_page)
    return f"{start_page}-{end_page}"
