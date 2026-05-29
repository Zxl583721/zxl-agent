from config import EMBEDDING_MODEL, EMBEDDING_PROVIDER, LLM_MODEL, LLM_PROVIDER


def get_chat_model(**kwargs):
    if LLM_PROVIDER == "ollama":
        from src.langchain_ollama import OllamaChatModel

        return OllamaChatModel(**kwargs)
    if LLM_PROVIDER == "zhipu":
        from src.langchain_zhipu import ZhipuChatModel

        model_kwargs = {"model_name": LLM_MODEL, **kwargs} if LLM_MODEL else kwargs
        return ZhipuChatModel(**model_kwargs)
    if LLM_PROVIDER == "deepseek":
        from src.langchain_deepseek import DeepSeekChatModel

        model_kwargs = {"model_name": LLM_MODEL, **kwargs} if LLM_MODEL else kwargs
        return DeepSeekChatModel(**model_kwargs)
    raise ValueError(f"不支持的 LLM_PROVIDER：{LLM_PROVIDER}")


def get_embeddings():
    if EMBEDDING_PROVIDER == "ollama":
        from src.langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model_name=EMBEDDING_MODEL)
    if EMBEDDING_PROVIDER == "zhipu":
        from src.langchain_zhipu import ZhipuEmbeddings

        return ZhipuEmbeddings()
    raise ValueError(f"不支持的 EMBEDDING_PROVIDER：{EMBEDDING_PROVIDER}")


def configured_embedding_model_name() -> str:
    return f"{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}"
