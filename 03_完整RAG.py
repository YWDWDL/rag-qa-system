# -*- coding: utf-8 -*-
"""
第 3 周动手：组装完整 RAG 系统（命令行版）

把 Week 1（调 LLM）+ Week 2（向量检索）串起来，变成一个能用的 RAG 问答系统。
跑通这个，你的作品集项目就"成型"了。

运行前准备：
  1. 已装 chromadb sentence-transformers openai python-dotenv（前两周装过了）
  2. .env 里已填 DEEPSEEK_API_KEY
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# ============ 1. 两个"引擎"：embedding（检索）+ LLM（生成） ============
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
llm = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ============ 2. 文档 → 向量化 → 存 Chroma（落盘！只做一次） ============
docs = [
    "无人机自主降落需要精确的位姿估计。",
    "模型预测控制（MPC）能在约束下优化降落轨迹。",
    "LSTM 可以预测移动平台的运动轨迹。",
    "RAG 是检索增强生成，先检索相关资料再生成答案。",
    "苹果是一种常见的水果。",
]

# PersistentClient = 落盘版（向量存到 ./chroma_data 文件夹）
# 对比你 Week 2 用的 Client()（内存版）：程序结束向量就没了
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("my_docs")

# 只在"第一次"写入（count()==0 表示还没存过），避免每次运行重复插入
if collection.count() == 0:
    for i, doc in enumerate(docs):
        collection.add(
            ids=[str(i)],
            embeddings=[embed_model.encode(doc).tolist()],
            documents=[doc],
        )
    print("首次运行：文档已向量化并落盘\n")
else:
    print(f"向量库已有 {collection.count()} 条，直接复用（不再重新向量化）\n")


# ============ 3. RAG 问答：检索 → 拼 prompt → 生成 ============
def ask(question):
    # ① 检索最相关的 3 条
    q_emb = embed_model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=3)
    context = "\n".join(results["documents"][0])

    # ② 拼提示词：system（设定规则）+ 资料 + 问题
    messages = [
        {"role": "system", "content": "你是问答助手。只根据【参考资料】回答，资料里没有就说'不知道'，不要编造。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"},
    ]

    # ③ 调 LLM 生成（temperature 调低到 0.3，少自由发挥、更贴资料）
    resp = llm.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=1,
    )
    return resp.choices[0].message.content


# ============ 4. 交互问答 ============
print("=" * 50)
print("RAG 问答系统就绪，开始提问（quit 退出）")
print("=" * 50)
while True:
    q = input("\n问题：")
    if q.strip().lower() == "quit":
        break
    print("答案：", ask(q))


# =====================================================================
# 【读懂证明】改一改：
# 1. 问一个和"苹果"相关的问题，观察它是否会说"不知道"（因为资料里没有相关内容）
# 2. 问一个和"无人机"相关、但 5 条资料里没有答案的问题，观察它会不会瞎编
# 3. 把 temperature 从 0.3 改成 1.0，再问第 2 题的问题，观察回答有什么不同
# =====================================================================
