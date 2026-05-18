from config import ZHIPUAI_API_KEY


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Create embeddings for a list of texts using ZhipuAI."""
    if not ZHIPUAI_API_KEY:
        raise RuntimeError("ZHIPUAI_API_KEY is missing. Please set it in .env.")
    texts = [clean_embedding_text(text) for text in texts]
    texts = [text for text in texts if text]

    if not texts:
        return []

    try:
        from zhipuai import ZhipuAI
    except ImportError as exc:
        raise RuntimeError("请先运行 pip install -r requirements.txt 安装 zhipuai。") from exc

    client = ZhipuAI(api_key=ZHIPUAI_API_KEY)

    # TODO: 如果智谱 SDK 的 embedding 模型名或返回结构有变化，只需要替换这里。
    # 常见模型名示例：embedding-3。
    response = client.embeddings.create(
        model="embedding-3",
        input=texts,
    )

    return [item.embedding for item in response.data]


def embed_text(text: str) -> list[float]:
    """Create one embedding vector for a single text."""
    vectors = embed_texts([text])
    return vectors[0] if vectors else []


def clean_embedding_text(text: str) -> str:
    """Drop invalid Unicode before sending text through the JSON API."""
    return str(text).encode("utf-8", errors="ignore").decode("utf-8").strip()
