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
from langchain.chains import create_retrieval_chain

# Load the MISTRAL_KEY and HF_TOKEN from the environment variables
MISTRAL_KEY = config("MISTRAL_KEY")
HF_TOKEN = config("HF_TOKEN")

# Set the HF_TOKEN environment variable
# This token is used for authentication with the Hugging Face API
os.environ["HF_TOKEN"] = HF_TOKEN

llm = ChatMistralAI(
    model="mistral-large-latest", mistral_api_key=MISTRAL_KEY
)

loader = TextLoader("11_Chat_With_Document/ai-discussion.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(documents)

# Mistral embeddings are using HF_TOKEN for the rag
embeddings = MistralAIEmbeddings(
    mistral_api_key=MISTRAL_KEY, model="mistral-embed"
)
vector_store = Chroma.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

st.title("Chat with Document")

question = st.text_input("Ask Your Question:")
if question:
    max_retries = 5
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            response = rag_chain.invoke({"input": question})
            if "answer" in response and response["answer"]:
                st.write(response["answer"])
                break
            else:
                st.write("No answer found. Please try again.")
        except Exception as e:
            if "Requests rate limit exceeded" in str(e):
                st.write(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise
        time.sleep(1)  # Add a delay of 1 second to handle the rate limit
