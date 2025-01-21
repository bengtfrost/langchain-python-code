from langchain_ollama.chat_models import ChatOllama
import streamlit as st

llm = ChatOllama(model="phi3.5:latest")

st.title("Q & A With AI")

question = st.text_input("Your Question")

if question:
    response = llm.invoke(question)
    st.write(response.content)
