from typing import Any
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, OLLAMA_TIMEOUT_SECONDS
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from src.zhipu_llm import DEFAULT_SYSTEM_PROMPT


class DeepSeekChatModel(BaseChatModel):
    """LangChain-compatible chat model for DeepSeek's OpenAI-compatible API."""

    model_name: str = LLM_MODEL or "deepseek-chat"
    temperature: float = 0.4
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

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
        if not DEEPSEEK_API_KEY:
            return "未检测到 DEEPSEEK_API_KEY。请在 .env 中填入你的 DeepSeek API Key。"

        payload = self._payload(messages=messages, stop=stop, stream=False, **kwargs)
        try:
            response = _post_json(f"{DEEPSEEK_BASE_URL}/chat/completions", payload)
        except RuntimeError as exc:
            return f"调用 DeepSeek 接口失败：{exc}"

        try:
            answer = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            return f"DeepSeek 返回结构异常，无法读取回答内容：{exc.__class__.__name__}: {exc}"

        return answer or "DeepSeek 返回了空回答。"

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        if not DEEPSEEK_API_KEY:
            yield ChatGenerationChunk(message=AIMessageChunk(content="未检测到 DEEPSEEK_API_KEY。请在 .env 中填入你的 DeepSeek API Key。"))
            return

        payload = self._payload(messages=messages, stop=stop, stream=True, **kwargs)
        try:
            for item in _post_sse(f"{DEEPSEEK_BASE_URL}/chat/completions", payload):
                try:
                    token = item["choices"][0]["delta"].get("content") or ""
                except (KeyError, IndexError, TypeError):
                    token = ""
                if not token:
                    continue
                if run_manager:
                    run_manager.on_llm_new_token(token)
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        except RuntimeError as exc:
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"调用 DeepSeek 接口失败：{exc}"))

    def _payload(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        stream: bool,
        **kwargs: Any,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model_name or "deepseek-chat",
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        return payload

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


def _post_json(url: str, payload: dict) -> dict:
    request = _request(url, payload)
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


def _post_sse(url: str, payload: dict):
    request = _request(url, payload)
    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                yield json.loads(data)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 {url}：{exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"请求 {url} 超时") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"接口返回不是合法 SSE JSON：{exc}") from exc


def _request(url: str, payload: dict) -> Request:
    return Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )


def _message_role(message: BaseMessage) -> str:
    message_type = getattr(message, "type", "")
    if message_type == "system":
        return "system"
    if message_type == "ai":
        return "assistant"
    return "user"
