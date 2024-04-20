
from config.keys import Keys
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
# from InstructorEmbedding import INSTRUCTOR

def get_openaiEmbedding_model():
    return OpenAIEmbeddings(openai_api_key=Keys.OPENAI_API_KEY)

def get_huggingfaceEmbedding_model(model_name):
    # return INSTRUCTOR(model_name=model_name)
    return HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True})

