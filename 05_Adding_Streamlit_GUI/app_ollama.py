from langchain_ollama.chat_models import ChatOllama
import streamlit as st

llm = ChatOllama(model="deepseek-r1:8b")

st.title("Q & A With AI")

question = st.text_input("Your Question")

if question:
    response = llm.invoke(question)
    st.write(response.content)
