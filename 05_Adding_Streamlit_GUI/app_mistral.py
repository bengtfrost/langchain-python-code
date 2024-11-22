from decouple import config
from langchain_mistralai import ChatMistral
import streamlit as st

# Ensure that the MISTRAL_KEY is correctly set in the environment variables
MISTRAL_KEY = config("MISTRAL_KEY")

# Initialize the language model with the specified model and API key
llm = ChatMistral(model="mistral-large-latest", mistral_api_key=MISTRAL_KEY)

# Set the title of the Streamlit app
st.title("Q & A With AI")

# Capture user input using Streamlit's text input widget
question = st.text_input("Ask a question:")
if question:
    # Invoke the language model to get the response
    response = llm.invoke(question)
    # Display the response using Streamlit's write function
    st.write(response.content)
