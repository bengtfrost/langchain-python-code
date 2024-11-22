from decouple import config
from langchain_mistralai import ChatMistral
import streamlit as st

# Import necessary modules for chat history management
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Sidebar configuration for model selection and API key input
st.sidebar.title("Configuration")
model = st.sidebar.selectbox("Select Model", ["mistral-large-latest"], index=0)
MISTRAL_KEY = st.sidebar.text_input("Mistral API Key", type="password")

# Validate user input for API key and model selection
if not MISTRAL_KEY:
    st.error("Please provide a Mistral API Key.")
elif not model:
    st.error("Please select a model.")
else:
    # Initialize the language model with the specified model and API key
    llm = ChatMistral(model=model, mistral_api_key=MISTRAL_KEY)

    # Define the prompt template with the specified input variables and template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI chatbot having a conversation with a human. Use the following context to understand the human question. Do not include emojis in your answer.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create the chat chain
    chain = prompt | llm

    # Initialize the chat message history
    history = StreamlitChatMessageHistory()

    # Create the chat chain with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Set the page title
    st.title("Q & A With AI")

    # Display chat history
    for message in st.session_state["langchain_messages"]:
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # Get user input
    question = st.chat_input("Ask a question:")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        try:
            # Generate response
            response = chain_with_history.stream(
                {"input": question}, config={"configurable": {"session_id": "any"}}
            )
            with st.chat_message("assistant"):
                st.write_stream(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Add clear history button
if st.sidebar.button("Clear History"):
    history.clear()
    st.session_state["langchain_messages"] = []
