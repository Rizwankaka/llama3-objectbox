import streamlit as st 
import os 
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_objectbox.vectorstores import ObjectBox
import time 

from dotenv import load_dotenv
load_dotenv()

# loading the Groq and OpenAi key 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only. 
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
""")
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()   
        st.session_state.loader = PyPDFDirectoryLoader("./data") 
        st.session_state.docs = st.session_state.loader.load() 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=768)  


input_prompt = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Documents are embedded successfully")

response = None  # Initialize response variable

if input_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': input_prompt})

        print("Response time:", time.process_time() - start)
        st.write(response['answer'])
    else:
        st.write("Please embed documents first by clicking 'Documents Embedding' button.")

# with a streamlit expander 
with st.expander("document Similarity Search"):
    # finding relevant chunks 
    if response and 'context' in response:
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------")
    else:
        st.write("No response or context available. Please enter a question and submit.")