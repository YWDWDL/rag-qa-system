# -*- coding: utf-8 -*-
"""
第 1 周动手 · 调通 DeepSeek API，多轮对话

这是你 RAG 项目的第一个零件：学会"写代码调大模型"。
读懂每一行 → 填上你的 API Key → 跑通 → 完成文末的"读懂证明"。

运行前准备：
  1. pip install openai
  2. 去 https://platform.deepseek.com 注册，创建 API Key（很便宜，充几块钱够用很久）
  3. 把下面的 api_key 换成你自己的
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# ============ 1. 初始化客户端（三要素） ============
load_dotenv()  # 读取同目录 .env 文件里的环境变量

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],  # ← 从环境变量读，key 不硬编码在代码里
    base_url="https://api.deepseek.com",      # ← 模型地址
)

# ============ 2. 多轮对话 ============
# 关键理解：模型"不记得"上一轮。要让对话连续，
# 每一轮都把【历史所有消息】重新发一遍，这就是 messages 的作用。
messages = [
    {"role": "system", "content": "你是一个只会说冷笑话的助手。"},
]

print("开始对话（输入 quit 退出）\n")
while True:
    user_input = input("你：")
    if user_input.strip().lower() == "quit":
        break

    # ① 把用户的话加进历史
    messages.append({"role": "user", "content": user_input})

    # ② 调 API：把整个 messages（含历史）发过去
    response = client.chat.completions.create(
        model="deepseek-chat",        # 用哪个模型
        messages=messages,            # 完整对话历史
        temperature=1,              # 随机性（0=严谨，1=发散）
    )

    # ③ 取出模型的回答
    answer = response.choices[0].message.content

    # ④ 把回答也加进历史（下一轮模型就知道自己说过什么）
    messages.append({"role": "assistant", "content": answer})

    print(f"AI：{answer}\n")


# =====================================================================
# 【读懂证明】改一改：
# 1. 把 temperature 改成 0.0，再问同样的问题，观察回答是否变得更"死板"
# 2. 把 system 提示词改成"你是一个只会说冷笑话的助手"，观察人设变化
# 3. 思考：如果把 messages 里"历史记录"那两行（append assistant）删掉，
#    多轮对话会发生什么？（提示：模型还会记得你上一句吗？）
# =====================================================================
