from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.rag_agent import RAGAgent
from src.retriever import ChromaRetriever


BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"


def create_agent() -> RAGAgent:
    retriever = ChromaRetriever(VECTOR_STORE_DIR)
    return RAGAgent(retriever=retriever)


app = Flask(__name__)
agent = create_agent()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    setup_error = getattr(agent.retriever, "setup_error", "")
    has_index = agent.has_knowledge_base()
    document_count = 0

    if getattr(agent.retriever, "collection", None) is not None:
        document_count = agent.retriever.collection.count()

    return jsonify(
        {
            "ready": has_index and not setup_error,
            "has_index": has_index,
            "document_count": document_count,
            "message": setup_error or _status_message(has_index, document_count),
        }
    )


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    history = data.get("history")

    if not question:
        return jsonify({"answer": "请输入一个问题。"}), 400

    if not isinstance(history, list):
        history = []

    answer = agent.answer(question, history=history)
    return jsonify({"answer": answer})


def _status_message(has_index: bool, document_count: int) -> str:
    if has_index:
        return f"Chroma 索引已加载，共 {document_count} 个文本块。"
    return "还没有可用的 Chroma 索引。请先运行 python build_index.py。"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
