from decouple import config
import os
import time
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

# Adding History
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Ensure that the MISTRAL_KEY and HF_TOKEN are correctly set in the environment variables
MISTRAL_KEY = config("MISTRAL_KEY")
HF_TOKEN = config("HF_TOKEN")

# Validate environment variables
if not MISTRAL_KEY or not HF_TOKEN:
    raise ValueError("MISTRAL_KEY and HF_TOKEN must be set in the environment variables.")

# Set the HF_TOKEN environment variable
# This token is used for authentication with the Hugging Face API
os.environ["HF_TOKEN"] = HF_TOKEN

# Initialize the language model with the specified model and API key
llm = ChatMistralAI(
    model="mistral-large-latest", mistral_api_key=MISTRAL_KEY
)

# Load and split the document
loader = TextLoader("11_Chat_With_Document/ai-discussion.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Initialize embeddings and vector store
embeddings = MistralAIEmbeddings(
    mistral_api_key=MISTRAL_KEY, model="mistral-embed"
)
vector_store = Chroma.from_documents(chunks, embeddings)

# Initialize retriever
retriever = vector_store.as_retriever()

# Define contextualize prompt
contextualize_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# Define system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Initialize question-answer chain and RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize chat history
history = StreamlitChatMessageHistory()

# Create conversational RAG chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Streamlit UI
st.title("Chat With Document")

# Initialize the chat history in the session state if it does not exist
if "langchain_messages" not in st.session_state:
    st.session_state["langchain_messages"] = []

# Ensure the chat history is not empty before accessing it
if st.session_state["langchain_messages"]:
    for message in st.session_state["langchain_messages"]:
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

# Get user input
question = st.chat_input("Your Question: ")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    # Retry logic for rate limit handling
    max_retries = 5
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            answer_chain = conversational_rag_chain.pick("answer")
            response = answer_chain.stream(
                {"input": question}, config={"configurable": {"session_id": "any"}}
            )
            with st.chat_message("assistant"):
                st.write_stream(response)
            break
        except Exception as e:
            if "Requests rate limit exceeded" in str(e):
                st.write(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise
