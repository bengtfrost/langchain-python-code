from decouple import config
from langchain_mistralai import ChatMistral
import streamlit as st

MISTRAL_KEY = config("MISTRAL_KEY")

llm = ChatMistral(model="mistral-large-latest", mistral_api_key=MISTRAL_KEY)

st.title("Q & A With AI")

question = st.text_input("Ask a question:")
if question:
    response = llm.invoke(question)
    st.write(response.content)
