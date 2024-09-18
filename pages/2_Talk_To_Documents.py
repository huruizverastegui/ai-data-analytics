import streamlit as st
import openai
import llama_index
from llama_index.llms.openai import OpenAI
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

from azure.storage.blob import BlobServiceClient
from io import BytesIO

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import TokenTextSplitter
from helpers.azhelpers import upload_to_azure_storage, list_all_containers, list_all_files, Logger, download_blob_to_file
import pandas as pd
from llama_index.readers.azstorage_blob import AzStorageBlobReader

import os 
from dotenv import load_dotenv

from pandasai.llm import OpenAI
from pandasai import SmartDataframe



import streamlit as st
import pandas as pd
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai import SmartDatalake
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse
import os


load_dotenv()

logger = Logger().get_logger()
logger.info("App started")


azure_storage_account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
azure_storage_account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
connection_string_blob = os.environ["CONNECTION_STRING_BLOB"]


blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")

container_list = list_all_containers()
container_list = [container for container in container_list if container.startswith("data-analytics")]
container_name = st.sidebar.selectbox("Answering questions from Kowledge Base", container_list)


# select the csv file you want to talk to
blob_list = list_all_files(container_name)
blob_name = st.sidebar.selectbox("Answering questions from file", blob_list)

model_variable = st.sidebar.selectbox("Powered by", ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "llama3-70B"])


def clear_chat_history():
    st.session_state.messages = []

#Button for clearing history
st.sidebar.text("Click to Clear Chat history")
st.sidebar.button("CLEAR 🗑️",on_click=clear_chat_history)


# Get the API parameters for the Llama models hosted on Azure 
if model_variable == "llama3-8B":
    azure_api_base = os.environ["URL_AZURE_LLAMA3_8B"]
    azure_api_key = os.environ["KEY_AZURE_LLAMA3_8B"]
elif model_variable == "llama3-70B":
    azure_api_base = os.environ["URL_AZURE_LLAMA3_70B"]
    azure_api_key = os.environ["KEY_AZURE_LLAMA3_70B"]



openai.api_key = os.environ["OPEN_AI_KEY"]
st.header("Start chatting with your documents 💬 📚")

                   
llm = OpenAI(api_token=os.environ["OPEN_AI_KEY"] )

@st.cache_data(show_spinner=True)
def load_data(llm_model,container_name,blob_name):
    with st.spinner(text="Loading and indexing the provided data files – hang tight! This should take a couple of minutes."):
        
        # make sure there is a path where to download the files 
        if not os.path.exists('data-files'):
            os.makedirs('data-files')

        # make sure it is empty 
        # delete every file in data-files       
        # for file in os.listdir('data-files'):
        #     os.remove(os.path.join('data-files', file))
        
        # download the file to the path
        # if the file is not in the path, download it
        
        if not os.path.exists('data-files/'+blob_name):
            download_blob_to_file(blob_service_client,'data-analytics-test',blob_name)
        
        df_csv = pd.read_csv('data-files/'+blob_name)
        
        return df_csv

df_csv = load_data(model_variable,container_name,blob_name)
sdf_csv = SmartDataframe(df_csv,config={"llm": llm})

st.success("Documents loaded and indexed successfully!")



def get_agent(data,llm):
    """
    The function creates an agent on the dataframes exctracted from the uploaded files
    Args: 
        data: A Dictionary with the dataframes extracted from the uploaded data
        llm:  llm object based on the ll type selected
    Output: PandasAI Agent
    """
    agent = Agent(sdf_csv,config = {"llm":llm,"verbose": True, "response_parser": StreamlitResponse})

    return agent

def chat_window(analyst):
    with st.chat_message("assistant"):
        st.text("Explore your data with PandasAI?🧐")

    #Initilizing message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    #Displaying the message history on re-reun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            #priting the questions
            if 'question' in message:
                st.markdown(message["question"])
            #printing the code generated and the evaluated code
            elif 'response' in message:
                #getting the response
                st.write(message['response'])
                
            #retrieving error messages
            elif 'error' in message:
                st.text(message['error'])
    #Getting the questions from the users
    user_question = st.chat_input("What are you curious about? ")

    
    if user_question:
        #Displaying the user question in the chat message
        with st.chat_message("user"):
            st.markdown(user_question)
        #Adding user question to chat history
        st.session_state.messages.append({"role":"user","question":user_question})
       
        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)
                st.write(response)
                st.session_state.messages.append({"role":"assistant","response":response})
        
        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"

analyst = get_agent([df_csv],llm)

#starting the chat with the PandasAI agent
chat_window(analyst)