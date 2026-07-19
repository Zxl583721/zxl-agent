from collections.abc import Generator
from pathlib import Path

from app.core.config import BASE_DIR, get_settings
from app.services.persistence_service import persistence_service
from app.utils.files import kb_collection_name, tenant_vector_dir
from src.rag_agent import RAGAgent
from src.retriever import ChromaRetriever
from src.reranker import RerankerSelection, build_reranker
from src.zhipu_llm import MAX_HISTORY_MESSAGES


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._reranker_selection: RerankerSelection | None = None

    def initialize_reranker(self) -> RerankerSelection:
        """Load the configured reranker once during application startup."""
        if self._reranker_selection is not None:
            return self._reranker_selection

        model_path = self.settings.bge_reranker_model_path
        if not Path(model_path).is_absolute():
            model_path = str(BASE_DIR / model_path)
        self._reranker_selection = build_reranker(
            self.settings.reranker_provider,
            model_path=model_path,
            use_fp16=self.settings.bge_reranker_use_fp16,
            batch_size=self.settings.bge_reranker_batch_size,
            fallback_provider=self.settings.reranker_fallback,
        )
        return self._reranker_selection

    def _create_agent(self, *, user_id: int, knowledge_base_id: int) -> RAGAgent:
        metadata_filter = {"$and": [{"user_id": user_id}, {"knowledge_base_id": knowledge_base_id}]}
        retriever = ChromaRetriever(
            tenant_vector_dir(knowledge_base_id),
            collection_name=kb_collection_name(knowledge_base_id),
            metadata_filter=metadata_filter,
        )
        selection = self.initialize_reranker()
        return RAGAgent(
            retriever=retriever,
            reranker=selection.reranker,
            reranker_provider=selection.active_provider,
            reranker_setup_error=selection.setup_error,
        )

    def chat(
        self,
        question: str,
        conversation_id: str | None,
        mode: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> dict:
        question = question.strip()
        mode = mode.strip().lower()
        session = persistence_service.get_or_create_chat_session(
            conversation_id=conversation_id,
            question=question,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        conversation_id = session.conversation_id
        history = persistence_service.recent_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=MAX_HISTORY_MESSAGES,
        )
        agent = self._create_agent(user_id=user_id, knowledge_base_id=knowledge_base_id)

        if mode in {"general", "chat"}:
            result = agent.answer_general(question, history=history)
        else:
            result = agent.answer_with_sources(question, history=history)
            result["mode"] = "knowledge"

        answer = str(result.get("answer", "")).strip()
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
        result["user_id"] = user_id
        result["knowledge_base_id"] = knowledge_base_id
        return result

    def stream_chat(
        self,
        question: str,
        conversation_id: str | None,
        mode: str,
        user_id: int,
        knowledge_base_id: int,
    ) -> Generator[dict, None, None]:
        question = question.strip()
        mode = mode.strip().lower()
        session = persistence_service.get_or_create_chat_session(
            conversation_id=conversation_id,
            question=question,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        conversation_id = session.conversation_id
        history = persistence_service.recent_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=MAX_HISTORY_MESSAGES,
        )
        agent = self._create_agent(user_id=user_id, knowledge_base_id=knowledge_base_id)
        answer_parts = []
        result_metadata = {"mode": mode, "sources": []}

        if mode in {"general", "chat"}:
            stream = agent.stream_general(question, history=history)
        else:
            stream = agent.stream_with_sources(question, history=history)

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
            answer_with_citations = agent._ensure_source_citations(answer, sources)
            citation_suffix = answer_with_citations[len(answer) :]
            if citation_suffix:
                answer_parts.append(citation_suffix)
                answer = answer_with_citations
                yield {"type": "token", "content": citation_suffix}

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


rag_service = RAGService()


def get_rag_service() -> RAGService:
    return rag_service
