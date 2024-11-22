# Import necessary libraries
from decouple import config
from langchain_mistralai import ChatMistral
import streamlit as st

# Adding History
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os, base64

# Load API key from environment variables
MISTRAL_KEY = config("MISTRAL_KEY")

# Initialize the language model
llm = ChatMistral(
    model="mistral-large-latest", mistral_api_key=MISTRAL_KEY
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can describe images."),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            [
                {"type": "text", "text": "{input}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

# Initialize the chat message history
history = StreamlitChatMessageHistory()

# Function to encode the image
def encode_image(image_path):
    """
    Encode the image to base64.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to process the image
def process_image(file):
    """
    Process the uploaded image file.

    Args:
        file (UploadedFile): The uploaded image file.
    """
    with st.spinner("Processing image..."):
        data = file.read()
        file_name = os.path.join("./", file.name)
        with open(file_name, "wb") as f:
            f.write(data)
        image = encode_image(file_name)
        st.session_state.encoded_image = image
        st.success("Image encoded. Ask your questions")

# Create the chain with history
chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Function to clear the chat history
def clear_history():
    """
    Clear the chat history from the session state.
    """
    if "langchain_messages" in st.session_state:
        del st.session_state["langchain_messages"]

# Streamlit app title
st.title("Chat With Image")

# Input for image upload
uploaded_file = st.file_uploader("Upload your image: ", type=["jpg", "png"])
add_file = st.button("Submit Image", on_click=clear_history)

if uploaded_file and add_file:
    process_image(uploaded_file)

# Display chat history
for message in st.session_state["langchain_messages"]:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Input for user question
question = st.chat_input("Your Question")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    if "encoded_image" in st.session_state:
        image = st.session_state["encoded_image"]
        response = chain_with_history.stream(
            {"input": question, "image": image}, config={"configurable": {"session_id": "any"}}
        )
        with st.chat_message("assistant"):
            st.write_stream(response)
    else:
        st.error("No image is uploaded. Upload your image first.")
