import os
from operator import itemgetter

import streamlit as st
import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

SYSTEM_MESSAGE = """You're a helpful assistant. Answer all questions to the best of your ability. If you don't know the answer let the user know to find help in the internet.

Available context:
{context}
"""
MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

ollama.pull(MODEL)
ollama.pull(EMBEDDING_MODEL)

mongo_client = MongoClient(MONGO_URI)
collection = mongo_client["bot"]["data"]
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

if collection.find_one({}) is None:
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Llama_(language_model)")
    md = MarkdownifyTransformer()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    docs = loader.load()
    converted_docs = md.transform_documents(docs)
    splits = text_splitter.split_documents(converted_docs)
    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        splits, embedding, collection=collection, index_name="default")
    vectorstore.create_vector_search_index(768)
else:
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=MONGO_URI, namespace="bot.data", embedding=embedding)
retriever = vectorstore.as_retriever()

chat = ChatOllama(model=MODEL)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_MESSAGE,
        ),
        ("human", "{input}"),
    ]
)

chain = {"context": itemgetter("input") | retriever, "input": itemgetter(
    "input")} | prompt_template | chat | StrOutputParser()

st.title("Chatbot")
st.caption("A Streamlit chatbot")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("ai"):
        st.write_stream(chain.stream({"input":prompt}))
