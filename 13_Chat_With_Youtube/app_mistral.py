from decouple import config
from langchain_mistralai import ChatMistral, MistralEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import YoutubeLoader

MISTRAL_KEY = config("MISTRAL_KEY")

llm = ChatMistral(
    model="mistral/mistral-large-2407", mistral_api_key=MISTRAL_KEY
)

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
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

history = StreamlitChatMessageHistory()

def display_transcript(docs):
    """
    Display the transcript of the YouTube video.

    Args:
        docs (list): List of documents containing the transcript.
    """
    transcript = "\n".join([doc.page_content for doc in docs])
    st.text_area("Video Transcript", transcript, height=300)

def process_youtube_url(url):
    """
    Process the YouTube URL to extract transcript and create a conversational RAG chain.

    Args:
        url (str): The YouTube video URL.
    """
    try:
        with st.spinner("Processing YouTube video..."):
            loader = YoutubeLoader.from_youtube_url(url)

            docs = loader.load()
            if docs:
                display_transcript(docs)
                if st.button("Confirm and Process"):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(docs)
                    embeddings = MistralEmbeddings(
                        mistral_api_key=MISTRAL_KEY, model="models/embedding-001"
                    )
                    vector_store = Chroma.from_documents(chunks, embeddings)
                    retriever = vector_store.as_retriever()
                    history_aware_retriever = create_history_aware_retriever(
                        llm, retriever, contextualize_prompt
                    )
                    rag_chain = create_retrieval_chain(
                        history_aware_retriever, question_answer_chain
                    )
                    conversational_rag_chain = RunnableWithMessageHistory(
                        rag_chain,
                        lambda session_id: history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer",
                    )
                    st.session_state.crc = conversational_rag_chain
                    st.success("Video processed. Ask your questions")
            else:
                st.error("Video has no transcript. Please try another video")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def clear_history():
    """
    Clear the chat history from the session state.
    """
    if "langchain_messages" in st.session_state:
        del st.session_state["langchain_messages"]

st.title("Ask YouTube Video")

youtube_url = st.text_input("Input YouTube URL")
submit_button = st.button("Submit Video", on_click=clear_history)
if youtube_url and submit_button:
    process_youtube_url(youtube_url)

if "langchain_messages" not in st.session_state:
    st.session_state["langchain_messages"] = []

for message in st.session_state["langchain_messages"]:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

question = st.chat_input("Your Question: ")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    if "crc" in st.session_state:
        crc = st.session_state["crc"]
        answer_chain = crc.pick("answer")
        response = answer_chain.stream(
            {"input": question}, config={"configurable": {"session_id": "any"}}
        )
        with st.chat_message("assistant"):
            st.write_stream(response)
    else:
        st.error("No video is linked. Input your video URL first.")
