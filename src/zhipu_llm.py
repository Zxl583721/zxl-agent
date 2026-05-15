from config import ZHIPUAI_API_KEY


def chat_with_zhipu(prompt: str) -> str:
    """Send a prompt to ZhipuAI chat model and return the answer text."""
    if not ZHIPUAI_API_KEY:
        return "未检测到 ZHIPUAI_API_KEY。请复制 .env.example 为 .env，并填入你的智谱 AI API Key。"

    try:
        from zhipuai import ZhipuAI
    except ImportError:
        return "未安装 zhipuai 依赖。请先运行 pip install -r requirements.txt。"

    client = ZhipuAI(api_key=ZHIPUAI_API_KEY)

    # 模型名可按需替换为你账号可用的智谱模型。
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content
