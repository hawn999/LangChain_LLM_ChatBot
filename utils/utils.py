
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS, Milvus, Pinecone, Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from models.llm_model import get_openai_model, get_huggingfacehub
import pinecone
import streamlit as st
from config.templates import bot_template, user_template
from config.keys import Keys
from PyPDF2 import PdfReader

def load_init_PDF_text():
    text=""
    file_path = '/Users/hwan/Downloads/LangChain_LLM_ChatBot/pdf/党建300问(1)_20240420211736.pdf'
    # 手动打开文件，指定编码为 UTF-8
    # file = open(file_path, 'r', encoding='utf-8')
    # 创建一个 PdfFileReader 对象
    pdf_reader = PdfReader(stream=file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    # file.close()
    return text



def extract_text_from_PDF(files):
    # 参考官网链接：https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
    # 加载多个PDF文件
    text = ""
    for pdf in files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_content_into_chunks(text):
    # 参考官网链接：https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/character_text_splitter
    text_spliter = CharacterTextSplitter(separator="\n",
                                         chunk_size=500,
                                         chunk_overlap=80,
                                         length_function=len)
    chunks = text_spliter.split_text(text)
    return chunks

def save_chunks_into_vectorstore(content_chunks, embedding_model):
    # 参考官网链接：https://python.langchain.com/docs/modules/data_connection/vectorstores/
    # pip install faiss-gpu (如果没有GPU，那么 pip install faiss-cpu)
    vectorstore = FAISS.from_texts(texts=content_chunks,
                                      embedding=embedding_model)
    return vectorstore

def get_chat_chain(vector_store):
    # ① 获取大语言模型LLM model
    llm = get_openai_model()
    # llm = get_huggingfacehub(model_name="deepset/roberta-base-squad2")

    # ② 存储历史记录
    # 参考官网链接：https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db
    # 用于缓存或者保存对话历史记录的对象
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    # ③ 对话链
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # 向量数据库进行检索
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def process_user_input(user_input):
    if st.session_state.conversation is not None:
        # 调用函数st.session_state.conversation，并把用户输入的内容作为一个问题传入，返回响应。
        response = st.session_state.conversation({'question': user_input})
        # session状态是Streamlit中的一个特性，允许在用户的多个请求之间保存数据。
        st.session_state.chat_history = response['chat_history']
        # 显示聊天记录
        # chat_history : 一个包含之前聊天记录的列表
        for i, message in enumerate(st.session_state.chat_history):
            # 用户输入
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True) # unsafe_allow_html=True表示允许HTML内容被渲染
            else:
                # 机器人响应
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
