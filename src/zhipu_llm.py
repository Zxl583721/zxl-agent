from config import ZHIPUAI_API_KEY


DEFAULT_SYSTEM_PROMPT = (
     "你是 zxl-agent 项目的中文知识库问答助手。\n"

    "你的主要任务是根据用户提供的知识库内容回答问题。\n"

    "回答时必须优先依据知识库内容，不要脱离知识库随意发挥。\n"

    "如果知识库中能找到答案，请用清晰、准确、适合初学者理解的方式回答。\n"

    "如果知识库内容不足以回答问题，请明确说明：知识库中没有足够信息。\n"

    "如果问题涉及代码、流程或概念，请尽量分步骤解释。\n"

    "不要编造知识库中不存在的事实、文件、结论或数据。\n"

    "默认使用中文回答。"
)

MAX_HISTORY_MESSAGES = 10


def _build_messages(
    prompt: str,
    history: list[dict] | None,
    system_prompt: str,
) -> list[dict]:
    """Build chat messages while keeping only recent valid history messages."""
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        valid_history = []
        for message in history[-MAX_HISTORY_MESSAGES:]:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and content:
                valid_history.append({"role": role, "content": str(content)})

        messages.extend(valid_history)

    messages.append({"role": "user", "content": prompt})
    return messages


def chat_with_zhipu(
    prompt: str,
    history: list[dict] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.4,
    max_tokens: int = 1024,
) -> str:
    """Send a prompt to ZhipuAI chat model and return the answer text."""
    if not ZHIPUAI_API_KEY:
        return "未检测到 ZHIPUAI_API_KEY。请复制 .env.example 为 .env，并填入你的智谱 AI API Key。"

    try:
        from zhipuai import ZhipuAI
    except ImportError:
        return "未安装 zhipuai 依赖。请先运行 pip install -r requirements.txt。"

    client = ZhipuAI(api_key=ZHIPUAI_API_KEY)
    messages = _build_messages(prompt, history, system_prompt)

    # 模型名可按需替换为你账号可用的智谱模型。
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
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
