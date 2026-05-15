from pathlib import Path

from src.document_loader import load_txt_documents
from src.rag_agent import RAGAgent
from src.retriever import SimpleMemoryRetriever
from src.text_splitter import split_text


DATA_DIR = Path(__file__).parent / "data"


def build_agent() -> RAGAgent:
    """Load local txt files, split them, and build a simple in-memory RAG agent."""
    documents = load_txt_documents(DATA_DIR)
    chunks = []

    for document in documents:
        text_chunks = split_text(document["content"], chunk_size=800, chunk_overlap=100)
        for index, chunk in enumerate(text_chunks):
            chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": document["source"],
                        "chunk_index": index,
                    },
                }
            )

    retriever = SimpleMemoryRetriever(chunks)
    return RAGAgent(retriever=retriever)


def main() -> None:
    agent = build_agent()

    print("RAG Agent 已启动。输入问题开始提问，输入 exit 退出。")
    if not agent.has_knowledge_base():
        print("提示：当前 data/ 目录中还没有可用的 .txt 知识库文件。")

    while True:
        question = input("\n你：").strip()
        if question.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        if not question:
            continue

        answer = agent.answer(question)
        print(f"\nAgent：{answer}")


if __name__ == "__main__":
    main()

