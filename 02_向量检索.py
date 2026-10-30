# -*- coding: utf-8 -*-
"""
第 2 周动手：RAG 核心——向量检索

把"文档"切成块 → 每块转成向量 → 存进向量库 → 用问题检索最相关的块。
这是你 RAG 项目的"心脏"：理解了这一步，就理解了 RAG 是怎么"检索"的。

运行前准备：
  1. pip install chromadb sentence-transformers
  2. 首次运行会从 hf-mirror 下载 embedding 模型（约 470MB，等几分钟）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 国内镜像，加速下载模型

from sentence_transformers import SentenceTransformer
import chromadb

# ============ 1. 加载 Embedding 模型（把文字变成向量） ============
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ============ 2. 准备"文档"（这里用 5 句话模拟，实际是你的 PDF/文本切成的块） ============
docs = [
    "无人机自主降落需要精确的位姿估计。",
    "模型预测控制（MPC）能在约束下优化降落轨迹。",
    "LSTM 可以预测移动平台的运动轨迹。",
    "RAG 是检索增强生成，先检索相关资料再生成答案。",
    "苹果是一种常见的水果。",
]

# ============ 3. 向量化 + 存入 Chroma 向量库 ============
client = chromadb.Client()
collection = client.create_collection("my_docs")

for i, doc in enumerate(docs):
    emb = model.encode(doc).tolist()      # 文字 → 向量
    collection.add(
        ids=[str(i)],                     # 每条给个 id
        embeddings=[emb],                 # 向量
        documents=[doc],                  # 原文（方便取回）
    )
print(f"已把 {len(docs)} 条文档向量化并存入向量库\n")

# ============ 4. 检索：问题也转成向量，找"距离最近"的块 ============
question = "无人机降落用什么控制方法？"
q_emb = model.encode(question).tolist()          # 问题 → 向量
results = collection.query(
    query_embeddings=[q_emb],
    n_results=3,                                 # 取最相似的 2 条
)

print("问题：", question)
print("最相关的片段：")
for doc in results["documents"][0]:
    print("  -", doc)

print("\n看到没：检索出的都是和'无人机降落'相关的，而不是'苹果'——")
print("因为 embedding 把'语义相近'变成了'向量距离近'。")


# =====================================================================
# 【读懂证明】改一改：
# 1. 换一个问题："苹果能用来做什么？" 观察检索出的片段变化
# 2. 往 docs 里加几条你自己的话（比如毕设相关内容），再检索试试
# 3. 思考：为什么"苹果"那条没被"无人机降落"的问题检索出来？（提示：向量距离远）
# =====================================================================
