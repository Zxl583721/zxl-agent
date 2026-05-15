from src.zhipu_llm import chat_with_zhipu


class RAGAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def has_knowledge_base(self) -> bool:
        return self.retriever.has_documents()

    def answer(self, question: str) -> str:
        if not self.has_knowledge_base():
            return (
                "当前还没有加载到知识库内容。请先把 .txt 文件放入 data/ 文件夹，"
                "然后重新运行 python main.py。"
            )

        retrieved_chunks = self.retriever.retrieve(question, top_k=3)
        if not retrieved_chunks:
            return "我没有在当前知识库中检索到相关内容，可以换个问法试试。"

        context = self._format_context(retrieved_chunks)
        prompt = self._build_prompt(question, context)
        return chat_with_zhipu(prompt)

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            context_parts.append(f"[资料 {index}] 来源：{source}\n{chunk['text']}")
        return "\n\n".join(context_parts)

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return f"""你是一个基于个人知识库的 RAG 问答助手。
请优先根据【知识库资料】回答用户问题。
如果资料中没有答案，请明确说明“知识库中没有找到相关信息”，不要编造。

【知识库资料】
{context}

【用户问题】
{question}

【回答】
"""

