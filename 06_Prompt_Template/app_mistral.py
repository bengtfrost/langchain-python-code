from decouple import config
from langchain_mistralai import ChatMistral
import streamlit as st
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["country", "paragraph", "language"],
    template="""
    You are a currency expert. You give information about a specific currency used in a specific country.
    Avoid giving information about fictional places.
    If the country is fictional or non-existent, answer: I don't know.

    Answer the question: What is the currency of {country}?

    Answer in {paragraph} short paragraph in {language}
    """,
)

MISTRAL_KEY = config("MISTRAL_KEY")

llm = ChatMistral(model="mistral-large-latest", mistral_api_key=MISTRAL_KEY)

st.title("Currency Info")

country = st.text_input("Input Country")
paragraph = st.number_input("Input Number of Paragraphs", min_value=1, max_value=5)
language = st.text_input("Input Language")

if country and paragraph and language:
    response = llm.invoke(
        prompt.format(country=country, paragraph=paragraph, language=language)
    )
    st.write(response.content)
