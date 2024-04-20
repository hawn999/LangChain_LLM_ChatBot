
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from config.keys import Keys

def get_openai_model():
    llm_model = ChatOpenAI(openai_api_key=Keys.OPENAI_API_KEY)
    return llm_model

def get_huggingfacehub(model_name=None):
    llm_model = HuggingFaceEndpoint(repo_id=model_name,
                                    # temperature=0.8,
                                    timeout=5000,
                                    huggingfacehub_api_token=Keys.HUGGINGFACEHUB_API_TOKEN
                                    # ,model_kwargs={"temperature":0.8, "max_length":512}
                                    # ,model_kwargs = {"temperature": 0.5}
                                    )

    return llm_model