import openai

# 设置 API 密钥
openai.api_key = "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"

def chat_with_gpt4o_mini(messages, temperature=0.7, max_tokens=1000):
    """
    使用 GPT-4o Mini 模型生成回复。

    参数：
    - messages: 包含对话历史的列表，每个元素是一个字典，包含 'role' 和 'content'。
    - temperature: 控制回复的随机性，默认值为 0.7。
    - max_tokens: 回复的最大 token 数，默认值为 1000。

    返回：
    - 模型生成的回复文本。
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"发生错误: {e}"