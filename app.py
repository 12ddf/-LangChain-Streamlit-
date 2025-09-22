# app.py

import streamlit as st
from ai_assistant import create_rag_chain  # 从我们的核心逻辑文件中导入函数
# ---页面设置 ---
st.set_page_config(page_title="我的专属AI知识库助手", page_icon="🤖")
st.title("🤖 知识库智能问答助手")

# --- 初始化 ---
# 使用 st.cache_resource 来缓存我们的RAG chain，避免每次都重新加载模型和文档
# 这会让应用响应速度更快
@st.cache_resource
def load_chain():
    return create_rag_chain()

# 在侧边栏创建一个说明
with st.sidebar:
    st.header("关于")
    st.markdown("""
    本项目基于 **LangChain** 和 **Streamlit** 构建。
    它能读取 `knowledge.pdf的内容，并根据其回答你的问题。
    """)

# 加载RAG chain
chain = load_chain()

# ---聊天界面 ---

# 初始化聊天历史 (使用session_state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 获取用户输入
if prompt := st.chat_input("请在这里输入你关于知识库的问题..."):
    # 将用户消息添加到历史记录并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 获取AI的回答    
    with st.chat_message("assistant"):
        with st.spinner("AI正在思考中..."):
            # 调用我们的RAG chain
            response = chain.invoke({"input": prompt})
            ai_answer = response["answer"]
            st.markdown(ai_answer)
    
    # 将AI的回答也添加到历史记录
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
