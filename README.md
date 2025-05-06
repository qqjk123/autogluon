LLM Sample:
messages = [
    {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
    {"role": "user", "content": "你好，GPT-4o Mini！"},
    {"role": "assistant", "content": "你好！有什么我可以帮助您的吗？"},
    {"role": "user", "content": "你叫什么名字？"}
]

# 调用函数获取模型回复
response = chat_with_gpt4o_mini(messages)

# 输出模型的回复
print(response)
