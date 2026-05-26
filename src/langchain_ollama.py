from typing import Any
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import EMBEDDING_BASE_URL, EMBEDDING_MODEL, LLM_BASE_URL, LLM_MODEL, OLLAMA_TIMEOUT_SECONDS
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from src.embedding import clean_embedding_text
from src.zhipu_llm import DEFAULT_SYSTEM_PROMPT


class OllamaEmbeddings(Embeddings):
    """LangChain-compatible embeddings for a local Ollama model."""

    model_name: str = EMBEDDING_MODEL

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return ollama_embed_texts(texts, model=self.model_name)

    def embed_query(self, text: str) -> list[float]:
        vectors = ollama_embed_texts([text], model=self.model_name)
        return vectors[0] if vectors else []


class OllamaChatModel(BaseChatModel):
    """LangChain-compatible chat model for Ollama."""

    model_name: str = LLM_MODEL
    temperature: float = 0.4
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "ollama-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        answer = self._chat(messages=messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    def _chat(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "stream": False,
            "think": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        try:
            response = _post_json(
                f"{_ollama_host(LLM_BASE_URL)}/api/chat",
                payload,
            )
        except RuntimeError as exc:
            return f"调用本地 Ollama 接口失败：{exc}"

        try:
            answer = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            return f"Ollama 返回结构异常，无法读取回答内容：{exc.__class__.__name__}: {exc}"

        return answer or "Ollama 返回了空回答。"

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        payload = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "stream": True,
            "think": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        try:
            for item in _post_json_lines(f"{_ollama_host(LLM_BASE_URL)}/api/chat", payload):
                token = item.get("message", {}).get("content", "")
                if not token:
                    continue
                if run_manager:
                    run_manager.on_llm_new_token(token)
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        except RuntimeError as exc:
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"调用本地 Ollama 接口失败：{exc}"))

    @staticmethod
    def _convert_messages(messages: list[BaseMessage]) -> list[dict]:
        converted = []
        for message in messages:
            role = _message_role(message)
            content = message.content
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content)
            converted.append({"role": role, "content": str(content)})

        if not any(message["role"] == "system" for message in converted):
            converted.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

        return converted


def ollama_embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    cleaned_texts = [clean_embedding_text(text) for text in texts]
    cleaned_texts = [text for text in cleaned_texts if text]
    if not cleaned_texts:
        return []

    try:
        response = _post_json(
            f"{EMBEDDING_BASE_URL}/api/embed",
            {
                "model": model,
                "input": cleaned_texts,
            },
        )
    except RuntimeError as exc:
        raise RuntimeError(f"调用本地 Ollama embedding 接口失败：{exc}") from exc

    embeddings = response.get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError("Ollama embedding 返回结构异常，缺少 embeddings 字段。")

    return embeddings


def _post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    request_headers = {
        "Content-Type": "application/json",
        **(headers or {}),
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 {url}：{exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"请求 {url} 超时") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"接口返回不是合法 JSON：{exc}") from exc


def _post_json_lines(url: str, payload: dict, headers: dict | None = None):
    request_headers = {
        "Content-Type": "application/json",
        **(headers or {}),
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                yield json.loads(line)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 {url}：{exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"请求 {url} 超时") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"接口返回不是合法 JSON 行：{exc}") from exc


def _ollama_host(base_url: str) -> str:
    return base_url.removesuffix("/v1").rstrip("/")


def _message_role(message: BaseMessage) -> str:
    message_type = getattr(message, "type", "")
    if message_type == "system":
        return "system"
    if message_type == "ai":
        return "assistant"
    return "user"
