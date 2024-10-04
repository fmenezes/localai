import os
from operator import itemgetter

import streamlit as st
import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pymongo import MongoClient
from langchain_mongodb import MongoDBChatMessageHistory, MongoDBAtlasVectorSearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

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
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

chain = {"context": itemgetter("input") | retriever, "input": itemgetter(
    "input"), "history": itemgetter(
    "history")} | prompt_template | chat | StrOutputParser()


def get_session_history() -> BaseChatMessageHistory:
    return MongoDBChatMessageHistory(
        MONGO_URI, "user", database_name="bot")

history_chain = RunnableWithMessageHistory(
    chain, get_session_history, input_messages_key="input", history_messages_key="history")

st.title("Chatbot")
st.caption("A Streamlit chatbot")

history = get_session_history()
if len(history.messages) == 0:
    history.add_ai_message('Hello, how can I assist you today?')
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            st.write_stream(history_chain.stream({"input": prompt}))
