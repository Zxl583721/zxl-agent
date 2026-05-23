from pathlib import Path
import re

from flask import Flask, jsonify, render_template, request

from build_index import DATA_DIR, build_index
from src.document_loader import SUPPORTED_EXTENSIONS
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

    result = agent.answer_with_sources(question, history=history)
    return jsonify(result)


@app.post("/api/upload")
def upload():
    global agent

    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"message": "请选择要上传的文件。"}), 400

    saved_files = []
    rejected_files = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        original_name = uploaded_file.filename or ""
        filename = sanitize_filename(original_name)
        suffix = Path(filename).suffix.lower()

        if not filename or suffix not in SUPPORTED_EXTENSIONS:
            rejected_files.append(original_name or "未命名文件")
            continue

        save_path = DATA_DIR / filename
        uploaded_file.save(save_path)
        saved_files.append(filename)

    if not saved_files:
        supported_formats = "、".join(sorted(SUPPORTED_EXTENSIONS))
        return jsonify({"message": f"没有可入库的文件。当前支持：{supported_formats}。"}), 400

    if not build_index():
        return jsonify(
            {
                "message": "文件已保存，但自动构建索引失败。请检查依赖、API Key 或终端日志。",
                "files": saved_files,
                "rejected_files": rejected_files,
            }
        ), 500

    agent = create_agent()
    return jsonify(
        {
            "message": f"已上传并完成索引：{len(saved_files)} 个文件。",
            "files": saved_files,
            "rejected_files": rejected_files,
        }
    )


def _status_message(has_index: bool, document_count: int) -> str:
    if has_index:
        return f"Chroma 索引已加载，共 {document_count} 个文本块。"
    return "还没有可用的 Chroma 索引。请先运行 python build_index.py。"


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", name)
    return name.strip(" .")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
