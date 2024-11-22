from decouple import config
from langchain_mistralai import ChatMistral
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug

# Enable debug mode
set_debug(True)

# Ensure that the MISTRAL_KEY is correctly set in the environment variables
MISTRAL_KEY = config("MISTRAL_KEY")

# Initialize the language model with the specified model and API key
llm = ChatMistral(model="mistral-large-latest", mistral_api_key=MISTRAL_KEY)

# Define the title prompt template with the specified input variables and template
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an expert journalist.

    You need to come up with an interesting title for the following topic: {topic}

    Answer exactly with one title
    """,
)

# Define the essay prompt template with the specified input variables and template
essay_prompt = PromptTemplate(
    input_variables=["title"],
    template="""
    You are an expert nonfiction writer.

    You need to write a short essay of 350 words for the following title:

    {title}

    Make sure that the essay is engaging and makes the reader feel excited.
    """,
)

# Define the first chain to generate a title
first_chain = title_prompt | llm | StrOutputParser()

# Define the second chain to generate an essay based on the title
second_chain = essay_prompt | llm

# Define the overall chain to first generate a title and then an essay
overall_chain = first_chain | second_chain

# Set the title of the Streamlit app
st.title("Essay Writer")

# Capture user input using Streamlit's text input widget
topic = st.text_input("Input Topic")

# Validate user input and invoke the overall chain to get the response
if topic:
    response = overall_chain.invoke({"topic": topic})
    # Display the response using Streamlit's write function
    st.write(response.content)
