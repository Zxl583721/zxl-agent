from typing import Any
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_TOKENS, LLM_MODEL, OLLAMA_TIMEOUT_SECONDS
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.zhipu_llm import DEFAULT_SYSTEM_PROMPT


class OpenAICompatibleChatModel(BaseChatModel):
    """LangChain-compatible chat model for OpenAI-compatible APIs."""

    model_name: str = LLM_MODEL
    temperature: float = 0.4
    max_tokens: int = LLM_MAX_TOKENS

    @property
    def _llm_type(self) -> str:
        return "openai-compatible-chat"

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
        if not LLM_API_KEY:
            return "未检测到 LLM_API_KEY。请在 .env 中填入你的模型服务 API Key。"

        payload = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        try:
            response = _post_json(
                _chat_completions_url(LLM_BASE_URL),
                payload,
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            )
        except RuntimeError as exc:
            return f"调用 OpenAI-compatible 接口失败：{exc}"

        try:
            answer = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            return f"OpenAI-compatible 返回结构异常，无法读取回答内容：{exc.__class__.__name__}: {exc}"

        return answer or "OpenAI-compatible 接口返回了空回答。"

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


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _message_role(message: BaseMessage) -> str:
    message_type = getattr(message, "type", "")
    if message_type == "system":
        return "system"
    if message_type == "ai":
        return "assistant"
    return "user"
