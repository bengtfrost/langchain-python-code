from decouple import config
from langchain_mistralai import ChatMistral
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.globals import set_debug

set_debug(True)

# Load the Mistral API key from environment variables
MISTRAL_KEY = config("MISTRAL_KEY")

# Initialize the language model
llm = ChatMistral(model="mistral-large-latest", mistral_api_key=MISTRAL_KEY)

# Define the prompt template for generating a title
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an expert journalist.

    You need to come up with an interesting title for the following topic: {topic}

    Answer exactly with one title
    """,
)

# Define the prompt template for generating an essay
essay_prompt = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""
    You are an expert nonfiction writer.

    You need to write a short essay of 350 words for the following title:

    {title}

    Make sure that the essay is engaging and makes the reader feel {emotion}.

    Format the output as a JSON object with three keys: 'title', 'emotion', 'essay' and fill them with respective values
    """,
)

def create_overall_chain(emotion):
    """
    Create a chain that first generates a title and then an essay based on the title and emotion.

    Args:
        emotion (str): The emotion to be conveyed in the essay.

    Returns:
        Chain: The overall chain for generating a title and an essay.
    """
    first_chain = title_prompt | llm | StrOutputParser()
    second_chain = essay_prompt | llm | JsonOutputParser()

    overall_chain = (
        first_chain
        | (lambda title: {"title": title, "emotion": emotion})
        | second_chain
    )
    return overall_chain

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Essay Writer Updated")

    # Input fields for topic and emotion
    topic = st.text_input("Input Topic")
    emotion = st.text_input("Input Emotion")

    # Validate inputs
    if topic and emotion:
        try:
            overall_chain = create_overall_chain(emotion)
            response = overall_chain.invoke({"topic": topic})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both a topic and an emotion.")

if __name__ == "__main__":
    main()
