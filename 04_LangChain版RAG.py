# -*- coding: utf-8 -*-
"""
第 4 周动手：LangChain 版 RAG（对照你手动的 03_完整RAG.py）

你手动版写了：检索 → 拼提示词 → 调 LLM。
LangChain 把这些步骤封装成"积木"，用 | 管道串起来。

运行前准备：
  pip install langchain langchain-openai langchain-huggingface langchain-chroma
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

# ============ LangChain 的"积木" ============
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# ① 模型（对应你手动的 llm）
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

# ② 向量化模型（对应你手动的 SentenceTransformer）
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# ③ 向量库（内存版，和 Week 2 一样，每次运行重新加文档）
vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
)

# ④ 加文档（对应你手动的 collection.add）
documents = [
    Document(page_content="无人机自主降落需要精确的位姿估计。"),
    Document(page_content="模型预测控制（MPC）能在约束下优化降落轨迹。"),
    Document(page_content="LSTM 可以预测移动平台的运动轨迹。"),
    Document(page_content="RAG 是检索增强生成，先检索相关资料再生成答案。"),
    Document(page_content="苹果是一种常见的水果。"),
]
vectorstore.add_documents(documents)

# ⑤ 检索器（对应你手动的 collection.query，k=3 取最相似 3 条）
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ⑥ 提示词模板（对应你手动的拼 messages）
prompt = ChatPromptTemplate.from_template(
    "你是问答助手。只根据【参考资料】回答，资料里没有就说'不知道'，不要编造。\n"
    "参考资料：\n{context}\n\n"
    "问题：{question}"
)

# ⑦ 链：用 | 管道把"检索→拼模板→模型→取文本"串起来（对应你手动的一步步）
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ============ 使用 ============
if __name__ == "__main__":
    question = "苹果是彩色的吗?"
    answer = chain.invoke(question)
    print("问题：", question)
    print("答案：", answer)


# =====================================================================
# 【读懂证明】改一改：
# 1. 对照你手动的 03，找出：LangChain 的 retriever/prompt/llm/chain 分别对应 03 里的哪段代码
# 2. 把问题换成"苹果是彩色的吗"，观察它是否也说"不知道"（防幻觉同样生效）
# 3. 思考：LangChain 帮你省掉了哪些"手动要写的代码"？
#    （提示：拼提示词、取输出文本这些，chain 都帮你做了）
# =====================================================================
