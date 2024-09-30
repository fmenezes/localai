import streamlit as st

st.title("Chatbot")
st.caption("A Streamlit chatbot")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
