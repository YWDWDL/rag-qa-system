# -*- coding: utf-8 -*-
"""
作品集：RAG 知识库问答系统（Streamlit 网页版）

这就是你简历上的作品集项目——一个有网页界面的完整 RAG 系统。
运行：streamlit run 05_Streamlit版RAG.py   （会自动打开浏览器）

运行前准备：pip install streamlit
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# ============ 加载模型（@st.cache_resource：缓存，只加载一次） ============
@st.cache_resource
def load_models():
    embed = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    return embed, llm

embed, llm = load_models()

# ============ 向量库（落盘，复用，和你 03 一样） ============
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("my_docs")

# 示例文档（真实项目里可以换成"上传文件"功能）
docs = [
    "无人机自主降落需要精确的位姿估计。",
    "模型预测控制（MPC）能在约束下优化降落轨迹。",
    "LSTM 可以预测移动平台的运动轨迹。",
    "RAG 是检索增强生成，先检索相关资料再生成答案。",
    "苹果是一种常见的水果。",
    "苹果的颜色是彩色的。",
]
if collection.count() == 0:
    for i, doc in enumerate(docs):
        collection.add(ids=[str(i)], embeddings=[embed.encode(doc).tolist()], documents=[doc])


# ============ RAG 问答（和 03 一样：检索 → 拼提示词 → 生成） ============
def ask(question):
    q_emb = embed.encode(question).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=3)
    sources = results["documents"][0]                    # 检索出的来源（用于展示）
    context = "\n".join(sources)
    messages = [
        {"role": "system", "content": "你是问答助手。只根据【参考资料】回答，资料里没有就说'不知道'，不要编造。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"},
    ]
    resp = llm.chat.completions.create(model="deepseek-chat", messages=messages, temperature=0.3)
    return resp.choices[0].message.content, sources


# ============ 网页界面 ============
st.title("📚 RAG 知识库问答系统")
st.caption("基于向量检索 + 大模型生成 · 答案附原文引用")

question = st.text_input("输入你的问题：")

if question:
    with st.spinner("检索中..."):
        answer, sources = ask(question)

    st.markdown("### 答案")
    st.write(answer)

    st.markdown("### 参考来源")
    for i, s in enumerate(sources, 1):
        st.write(f"{i}. {s}")


# =====================================================================
# 【读懂证明】改一改：
# 1. 把 docs 换成你自己的内容（比如毕设相关的句子），再提问试试
# 2. 思考：@st.cache_resource 起了什么作用？如果去掉它，会发生什么？
# 3. 加分项：加一个 st.file_uploader 让用户上传 txt 文件，切块后加进向量库
# =====================================================================
