# 智能知识库问答助手 (Intelligent Knowledge Base QA Assistant)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Passing-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Shining-red.svg)](https://streamlit.io/)

这是一个基于LangChain和Streamlit构建的本地知识库问答应用。用户可以上传自己的PDF文档，应用会将其处理成一个可查询的知识库，并通过大语言模型（LLM）实现智能问答。

## ✨ 项目亮点 (Features)

* **前端交互界面**: 使用 `Streamlit` 构建了简洁、友好的Web用户界面。
* **核心RAG管线**: 基于 `LangChain` 实现完整的检索增强生成（RAG）流程，确保回答的准确性。
* **文档处理**: 支持上传PDF文件，并自动进行文本分割和向量化。
* **向量存储**: 使用 `ChromaDB` 在本地存储文档向量，实现高效检索。
* **模块化设计**: 将核心逻辑与界面代码分离，提高了代码的可读性和可维护性。

## 🚀 如何运行 (Getting Started)

请按照以下步骤在你的本地环境中运行本项目。

### 1. 克隆仓库
从该链接中复制project文件夹

'''https://github.com/12ddf/-LangChain-Streamlit-''''



### 2. 创建并激活conda环境

conda create -n my-rag-app python=3.9

conda activate my-rag-app

### 3.安装依赖
本项目的所有依赖都记录在 requirements.txt 文件中。运行以下命令进行安装：

pip install -r requirements.txt

### 4.安装API密钥
本项目需要使用大语言模型API。请按照以下步骤操作：
复制 .env.example 文件，并将其重命名为 .env。
在 .env 文件中填入你的API密钥和API基础地址。
#### .env 文件内容示例
SILICONFLOW_API_KEY="你的SiliconFlow API Key"

SILICONFLOW_BASE_URL="[https://api.siliconflow.cn/v1](https://api.siliconflow.cn/v1)"

### 5.启动应用
一切准备就绪！运行以下命令启动Streamlit应用：

streamlit run app.py





