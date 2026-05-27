from collections.abc import Generator

from app.core.config import get_settings
from app.services.persistence_service import persistence_service
from src.conversation_store import ConversationStore
from src.rag_agent import RAGAgent
from src.retriever import ChromaRetriever
from src.zhipu_llm import MAX_HISTORY_MESSAGES


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.conversation_store = ConversationStore(self.settings.conversation_db_path)
        self.agent = self._create_agent()

    def _create_agent(self) -> RAGAgent:
        retriever = ChromaRetriever(self.settings.vector_store_dir)
        return RAGAgent(retriever=retriever)

    def reload_agent(self) -> None:
        self.agent = self._create_agent()

    def health(self) -> dict:
        setup_error = getattr(self.agent.retriever, "setup_error", "")
        has_index = self.agent.has_knowledge_base()
        document_count = 0

        if getattr(self.agent.retriever, "collection", None) is not None:
            document_count = self.agent.retriever.collection.count()

        return {
            "status": "ok",
            "ready": has_index and not setup_error,
            "has_index": has_index,
            "document_count": document_count,
            "message": setup_error or self._status_message(has_index, document_count),
        }

    def chat(
        self,
        question: str,
        conversation_id: str | None = None,
        mode: str = "knowledge",
        user_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> dict:
        question = question.strip()
        mode = mode.strip().lower()
        conversation_id = self.conversation_store.get_or_create_conversation(conversation_id)
        history = self.conversation_store.recent_messages(conversation_id, MAX_HISTORY_MESSAGES)

        if mode in {"general", "chat"}:
            result = self.agent.answer_general(question, history=history)
        else:
            result = self.agent.answer_with_sources(question, history=history)
            result["mode"] = "knowledge"

        answer = str(result.get("answer", "")).strip()
        self.conversation_store.add_message(conversation_id, "user", question, mode=result.get("mode", mode))
        self.conversation_store.add_message(
            conversation_id,
            "assistant",
            answer,
            mode=result.get("mode", mode),
            sources=result.get("sources", []),
        )
        persistence_service.record_chat_exchange(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            mode=result.get("mode", mode),
            sources=result.get("sources", []),
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        result["conversation_id"] = conversation_id
        result["user_id"] = user_id or self.settings.default_user_id
        result["knowledge_base_id"] = knowledge_base_id or self.settings.default_knowledge_base_id
        return result

    def stream_chat(
        self,
        question: str,
        conversation_id: str | None = None,
        mode: str = "knowledge",
        user_id: int | None = None,
        knowledge_base_id: int | None = None,
    ) -> Generator[dict, None, None]:
        question = question.strip()
        mode = mode.strip().lower()
        conversation_id = self.conversation_store.get_or_create_conversation(conversation_id)
        history = self.conversation_store.recent_messages(conversation_id, MAX_HISTORY_MESSAGES)
        answer_parts = []
        result_metadata = {"mode": mode, "sources": []}

        if mode in {"general", "chat"}:
            stream = self.agent.stream_general(question, history=history)
        else:
            stream = self.agent.stream_with_sources(question, history=history)

        yield {"type": "metadata", "conversation_id": conversation_id}
        for event in stream:
            event_type = event.get("type")
            if event_type == "metadata":
                result_metadata.update(event)
                yield {**event, "conversation_id": conversation_id}
                continue
            if event_type == "token":
                answer_parts.append(str(event.get("content", "")))
            yield event

        answer = "".join(answer_parts).strip()
        sources = result_metadata.get("sources", [])
        if sources:
            answer_with_citations = self.agent._ensure_source_citations(answer, sources)
            citation_suffix = answer_with_citations[len(answer) :]
            if citation_suffix:
                answer_parts.append(citation_suffix)
                answer = answer_with_citations
                yield {"type": "token", "content": citation_suffix}

        self.conversation_store.add_message(
            conversation_id,
            "user",
            question,
            mode=result_metadata.get("mode", mode),
        )
        self.conversation_store.add_message(
            conversation_id,
            "assistant",
            answer,
            mode=result_metadata.get("mode", mode),
            sources=sources,
        )
        persistence_service.record_chat_exchange(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            mode=result_metadata.get("mode", mode),
            sources=sources,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        yield {"type": "done", "conversation_id": conversation_id}

    @staticmethod
    def _status_message(has_index: bool, document_count: int) -> str:
        if has_index:
            return f"Chroma 索引已加载，共 {document_count} 个文本块。"
        return "还没有可用的 Chroma 索引。请先运行 python build_index.py。"


rag_service = RAGService()
