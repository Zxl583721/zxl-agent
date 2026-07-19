from pathlib import Path

from src.rag_agent import RAGAgent
from src.retriever import ChromaRetriever


VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"


def build_agent() -> RAGAgent:
    """Load the local Chroma vector database and build the RAG agent."""
    retriever = ChromaRetriever(VECTOR_STORE_DIR)
    return RAGAgent(retriever=retriever)


def main() -> None:
    agent = build_agent()
    history = []

    print("RAG Agent 已启动。输入问题开始提问，输入 exit 退出。")
    if getattr(agent.retriever, "setup_error", ""):
        print(f"提示：{agent.retriever.setup_error}")
    if getattr(agent, "reranker_setup_error", ""):
        print(f"错误提示：{agent.reranker_setup_error}")
    print(f"提示：当前 reranker={getattr(agent, 'reranker_provider', 'unknown')}")
    if not agent.has_knowledge_base():
        print("提示：当前还没有可用的 Chroma 索引。请先运行 python build_index.py。")

    while True:
        question = input("\n你：").strip()
        if question.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        if not question:
            continue

        answer = agent.answer(question, history=history)
        print(f"\nAgent：{answer}")
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )


if __name__ == "__main__":
    main()
