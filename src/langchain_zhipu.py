from typing import Any

from config import ZHIPUAI_API_KEY
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from src.embedding import embed_text, embed_texts
from src.zhipu_llm import DEFAULT_SYSTEM_PROMPT


class ZhipuEmbeddings(Embeddings):
    """LangChain-compatible wrapper around the existing Zhipu embedding calls."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_text(text)


class ZhipuChatModel(BaseChatModel):
    """LangChain-compatible chat model wrapper for ZhipuAI."""

    model_name: str = "glm-4-flash"
    temperature: float = 0.4
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "zhipu-chat"

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
        if not ZHIPUAI_API_KEY:
            return "未检测到 ZHIPUAI_API_KEY。请复制 .env.example 为 .env，并填入你的智谱 AI API Key。"

        try:
            from zhipuai import ZhipuAI
        except ImportError:
            return "未安装 zhipuai 依赖。请先运行 pip install -r requirements.txt。"

        client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
        payload = self._convert_messages(messages)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
        except Exception as exc:
            return f"调用智谱 AI 接口失败：{exc.__class__.__name__}: {exc}"

        try:
            answer = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            return f"智谱 AI 返回结构异常，无法读取回答内容：{exc.__class__.__name__}: {exc}"

        if not answer:
            return "智谱 AI 返回了空回答。"

        return answer

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ):
        if not ZHIPUAI_API_KEY:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="未检测到 ZHIPUAI_API_KEY。请复制 .env.example 为 .env，并填入你的智谱 AI API Key。"
                )
            )
            return

        try:
            from zhipuai import ZhipuAI
        except ImportError:
            yield ChatGenerationChunk(message=AIMessageChunk(content="未安装 zhipuai 依赖。请先运行 pip install -r requirements.txt。"))
            return

        client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
        payload = self._convert_messages(messages)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                stream=True,
            )
            for chunk in response:
                try:
                    token = chunk.choices[0].delta.content or ""
                except (AttributeError, IndexError, TypeError):
                    token = ""
                if not token:
                    continue
                if run_manager:
                    run_manager.on_llm_new_token(token)
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        except Exception as exc:
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"调用智谱 AI 接口失败：{exc.__class__.__name__}: {exc}"))

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


def _message_role(message: BaseMessage) -> str:
    message_type = getattr(message, "type", "")
    if message_type == "system":
        return "system"
    if message_type == "ai":
        return "assistant"
    return "user"
