# rag_core.py

import os
from dotenv import load_dotenv

# --- 导入所有需要的 LangChain 组件 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain():
    """
    封装整个RAG流程，创建一个可供调用的问答链。
    返回:
        一个可以调用 .invoke() 方法的 retrieval_chain 对象。
    """
    # 0. 加载环境变量
    load_dotenv()
    print("环境变量加载成功！")

    # --- 第一阶段：初始化模型 ---

    # 1. 初始化你的大语言模型 (LLM)
    #    注意：这里我假设你的大模型和Embedding模型使用相同的API配置
    #    你需要根据你使用的模型（如Qwen-Turbo等）来修改 model_name
    print("正在初始化大语言模型...")
    llm = ChatOpenAI(
        model_name="deepseek-ai/DeepSeek-V3.1",  # 或者其他你选用的聊天模型
        openai_api_base="https://api.siliconflow.cn/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0  # 我们希望它严格根据资料回答，所以温度设为0
    )
    print("大语言模型初始化完成！")

    # 2. 初始化你的Embeddings模型
    print("正在初始化Embeddings模型...")
    embeddings = OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-8B",  # 使用最新的模型
        openai_api_base="https://api.siliconflow.cn/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    print("Embeddings模型初始化完成！")

    # --- 第二阶段：文档处理与向量化 ---

    # 3. 加载并切分PDF文档
    loader = PyPDFLoader("knowledge.pdf")
    print("正在加载和切分PDF...")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"文档处理完成，共得到 {len(splits)} 个知识块。")

    # 4. 创建向量数据库
    print("正在创建向量数据库... 这个过程可能需要一些时间...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    print("向量数据库创建成功！")

    # --- 第三阶段：构建问答链 ---

    # 5. 从向量数据库创建检索器
    #    检索器的任务是：根据问题，从数据库中找出最相关的知识块
    retriever = vectorstore.as_retriever()
    print("检索器创建成功！")

    # 6. 定义Prompt模板
    prompt = ChatPromptTemplate.from_template("""
    你是一个智能问答助手。
    请你只根据下面提供的【背景知识】，来回答用户提出的【问题】。
    如果【背景知识】里没有相关内容，就直接说“我不知道”。

    【背景知识】:
    {context}

    【问题】: 
    {input}

    【你的回答】:
    """)

    # 7. 创建文档处理链和最终的检索链
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("\n--- RAG问答链已准备就绪！---\n")

    return retrieval_chain


# --- 主程序入口：用于直接运行此文件进行测试 ---
if __name__ == '__main__':
    # 1. 创建RAG链
    rag_chain = create_rag_chain()

    # 2. 提出问题并获取答案
    #    你可以修改这里的query来进行不同的测试
    query = "高预算有什么补剂推荐？"
    print(f"正在基于知识库回答问题：'{query}'")

    # .invoke() 会触发整个链条的运行
    response = rag_chain.invoke({"input": query})

    # 3. 打印最终答案
    final_answer = response.get("answer", "抱歉，未能生成答案。")  # 使用.get()更安全
    print("\n--- 最终答案 ---")
    print(final_answer)

