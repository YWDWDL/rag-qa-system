# RAG 知识库问答系统

基于**检索增强生成（RAG）**的本地知识库问答系统：上传文档 → 向量化入库 → 语义检索 → 大模型生成**带原文引用**的答案。

## 功能亮点

- 🔍 **语义检索**：sentence-transformers 向量化，Chroma 向量库落盘持久化
- 🧠 **大模型生成**：DeepSeek API，温度 0.3 贴合资料
- 🛡️ **防幻觉**：system 提示词约束"资料里没有就说不知道"，答案附参考来源
- 🖥️ **网页界面**：Streamlit，纯 Python 实现

## 技术栈

Python · LangChain · Chroma · DeepSeek API · sentence-transformers · Streamlit

## 快速上手

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（复制 .env.example 为 .env，填入你的 key）
# DEEPSEEK_API_KEY=sk-你的key

# 3. 运行（自动打开浏览器）
streamlit run 05_Streamlit版RAG.py
```

## 项目结构（从零到完整的学习路径）

| 文件 | 内容 |
|---|---|
| `01_调通API.py` | 调 DeepSeek API 多轮对话（入门） |
| `02_向量检索.py` | 向量化 + 相似检索（RAG 心脏） |
| `03_完整RAG.py` | 完整 RAG 系统（命令行版） |
| `04_LangChain版RAG.py` | LangChain 框架版（对照手动版） |
| `05_Streamlit版RAG.py` | Streamlit 网页版（通用 demo） |
| `06_论文RAG.py` | **论文综述问答（真实 .docx 实战，作品集亮点）** |

> 最出彩的是 `06_论文RAG.py`：把一份 19MB 的论文综述 .docx 读入、切块、向量化，实现专业问答。

## 核心链路

```
文档 → 切块 → 向量化 → Chroma 存储（落盘）
问题 → 向量化 → 语义检索 top-3 → 拼提示词 → DeepSeek 生成 → 带引用答案
```
