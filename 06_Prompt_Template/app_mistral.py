from decouple import config
from langchain_mistralai import ChatMistral
import streamlit as st
from langchain.prompts import PromptTemplate

# Ensure that the MISTRAL_KEY is correctly set in the environment variables
MISTRAL_KEY = config("MISTRAL_KEY")

# Define the prompt template with the specified input variables and template
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

# Initialize the language model with the specified model and API key
llm = ChatMistral(model="mistral-large-latest", mistral_api_key=MISTRAL_KEY)

# Set the title of the Streamlit app
st.title("Currency Info")

# Capture user input using Streamlit's input widgets
country = st.text_input("Input Country")
paragraph = st.number_input("Input Number of Paragraphs", min_value=1, max_value=5)
language = st.text_input("Input Language")

# Validate user input and invoke the language model to get the response
if country and paragraph and language:
    response = llm.invoke(
        prompt.format(country=country, paragraph=paragraph, language=language)
    )
    # Display the response using Streamlit's write function
    st.write(response.content)
