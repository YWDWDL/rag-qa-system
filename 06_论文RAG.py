# -*- coding: utf-8 -*-
"""
作品集升级：论文综述问答（读真实 .docx）

把《自主起降综述.docx》读进来 → 切块 → 向量化 → 入库 → 提问。
这是你 RAG 第一次"吃"真实文档，也是作品集里最出彩的实战。

运行：streamlit run 06_论文RAG.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from docx import Document

# ============ 加载模型（缓存，不重复加载） ============
@st.cache_resource
def load_models():
    embed = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    return embed, llm

embed, llm = load_models()

# ============ 读 .docx + 切块（缓存，只读一次文件） ============
@st.cache_data
def load_docx_chunks(path):
    doc = Document(path)                                  # 读 .docx
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]   # 取出所有非空段落
    full_text = "\n".join(paragraphs)                     # 拼成一整篇
    # 切块：500 字一块，重叠 50 字（重叠避免切断语义）
    chunk_size, overlap = 500, 50
    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunk = full_text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# ============ 向量库（论文专用，落盘） ============
client = chromadb.PersistentClient(path="./chroma_thesis")
collection = client.get_or_create_collection("thesis")

DOCX_PATH = "d:/桌面/自主起降综述.docx"
chunks = load_docx_chunks(DOCX_PATH)

if collection.count() == 0:
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            embeddings=[embed.encode(chunk).tolist()],
            documents=[chunk],
        )
    st.success(f"首次运行：已把综述切成 {len(chunks)} 块并向量化入库")
else:
    st.info(f"向量库已有 {collection.count()} 块，直接复用")


# ============ RAG 问答 ============
def ask(question):
    q_emb = embed.encode(question).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=3)
    sources = results["documents"][0]
    context = "\n".join(sources)
    messages = [
        {"role": "system", "content": "你是论文助手。只根据【参考资料】回答，资料里没有就说'不知道'，不要编造。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"},
    ]
    resp = llm.chat.completions.create(model="deepseek-chat", messages=messages, temperature=0.3)
    return resp.choices[0].message.content, sources


# ============ 网页界面 ============
st.title("📚 论文综述问答（自主起降）")
st.caption(f"已加载《自主起降综述.docx》，共 {len(chunks)} 块")

question = st.text_input("问关于综述的问题（如：综述里提到了哪些降落方法？）：")

if question:
    with st.spinner("检索中..."):
        answer, sources = ask(question)
    st.markdown("### 答案")
    st.write(answer)
    st.markdown("### 参考来源（综述原文片段）")
    for i, s in enumerate(sources, 1):
        st.write(f"{i}. {s[:200]}...")   # 只显示前 200 字，避免刷屏


# =====================================================================
# 【读懂证明】改一改：
# 1. 问几个关于综述的问题（如"提到了哪些预测方法""LSTM 和 MPC 分别干什么"），看答案质量
# 2. 把 chunk_size 从 500 改成 200 或 1000，观察检索效果变化
# 3. 思考：如果换成 PDF 文件，要装什么库？（提示：pypdf）
# =====================================================================
