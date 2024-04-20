import PyPDF2
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

from config.keys import Keys
from utils.utils import extract_text_from_PDF, split_content_into_chunks
from utils.utils import save_chunks_into_vectorstore, get_chat_chain, process_user_input
from models.embedding_model import get_openaiEmbedding_model, get_huggingfaceEmbedding_model
from utils.utils import load_init_PDF_text
import streamlit as st

is_first_login=True

# 定义一个函数来验证用户名和密码
def authenticate(username, password):
    # 这里你可以根据需要实现你的身份验证逻辑
    return username == "a" and password == "a"


def main():
    global is_first_login
    # 初始化
    # session_state是Streamlit提供的用于存储会话状态的功能
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # 配置界面
    st.set_page_config(page_title="基于LangChain+LLM+Streamlit的问答系统",
                       page_icon=":robot:")

    container1 = st.empty()
    container2 = st.empty()
    container3 = st.empty()
    container4 = st.empty()
    container5 = st.empty()
    container6 = st.empty()
    #登陆界面
    container1.header("登录示例")

    # 添加输入框用于输入用户名和密码
    username = container2.text_input("用户名")
    password = container3.text_input("密码", type="password")

    # 添加登录按钮
    if container4.button("登录"):
        # 进行身份验证
        if authenticate(username, password):
            container5.success("登录成功！")
            # 将用户登录状态存储在会话中
            st.session_state.logged_in = True
        else:
            container6.error("用户名或密码错误！")

    # 如果用户已登录，则显示一些内容
    if "logged_in" in st.session_state and st.session_state.logged_in:
        container1.empty()
        container2.empty()
        container3.empty()
        container4.empty()
        container5.empty()
        container6.empty()

        st.header("登陆成功！欢迎来到基于LangChain+LLM+Streamlit的问答系统")

        if is_first_login:
            with st.spinner("正在处理本地知识库，请稍等..."):
                # 初始化对话链
                # 获取PDF文档内容（文本）
                texts = load_init_PDF_text()
                # 将获取到的文档内容进行切分
                content_chunks = split_content_into_chunks(texts)
                # 对每个chunk计算embedding，并存入到向量数据库
                # 根据model_type和model_name创建embedding model对象
                embedding_model = get_openaiEmbedding_model()
                # embedding_model = get_huggingfaceEmbedding_model(model_name="hkunlp/instructor-xl")
                # embedding_model = get_huggingfaceEmbedding_model(model_name="deepset/roberta-base-squad2")
                # 创建向量数据库对象，并将文本embedding后存入到里面
                vector_store = save_chunks_into_vectorstore(content_chunks, embedding_model)
                # 创建对话chain
                # 官网链接：https://python.langchain.com/docs/modules/memory/types/buffer
                st.session_state.conversation = get_chat_chain(vector_store)
                # global is_first_login = False
                st.write("本地知识库处理成功！")

        # 提供用户输入文本框
        user_input = st.text_input("请输入你的提问: ")
        # 处理用户输入，并返回响应结果
        if user_input:
            process_user_input(user_input)

        with st.sidebar:
            # 设置子标题
            st.subheader("自定义知识库")
            # 上传文档
            files = st.file_uploader("上传PDF文档，然后点击'提交并处理'，即可对该文档内容进行问答",
                                     accept_multiple_files=True)
            if st.button("提交并处理"):
                with st.spinner("请等待，处理中..."):
                    # 获取PDF文档内容（文本）
                    texts = extract_text_from_PDF(files)
                    # 将获取到的文档内容进行切分
                    content_chunks = split_content_into_chunks(texts)
                    # 对每个chunk计算embedding，并存入到向量数据库
                    # 根据model_type和model_name创建embedding model对象
                    embedding_model = get_openaiEmbedding_model()
                    # embedding_model = get_huggingfaceEmbedding_model(model_name="hkunlp/instructor-xl")
                    # embedding_model = get_huggingfaceEmbedding_model(model_name="deepset/roberta-base-squad2")
                    # 创建向量数据库对象，并将文本embedding后存入到里面
                    vector_store = save_chunks_into_vectorstore(content_chunks, embedding_model)
                    # 创建对话chain
                    st.session_state.conversation = get_chat_chain(vector_store)
                st.write("PDF处理成功！")


if __name__ == "__main__":
    main()