import streamlit as st
import ollama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

SYSTEM_MESSAGE = "You're a helpful assistant. Answer all questions to the best of your ability. If you don't know the answer let the user know to find help in the internet."

MODEL = "llama3.2"

with st.spinner("Loading"):
    ollama.pull(MODEL)

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

chain = prompt_template | chat | StrOutputParser()

st.title("Chatbot")
st.caption("A Streamlit chatbot")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("ai"):
        st.write_stream(chain.stream(prompt))
