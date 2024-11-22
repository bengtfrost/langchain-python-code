from decouple import config
from langchain import hub
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_mistralai import ChatMistralAI

# Ensure that the MISTRAL_KEY is correctly set in the environment variables
MISTRAL_KEY = config("MISTRAL_KEY")

# Initialize the language model with the specified model and API key
llm = ChatMistralAI(
    model="mistral-large-latest", mistral_api_key=MISTRAL_KEY
)

# Load the prompt from the hub
prompt = hub.pull("hwchase17/react")

# Load the tools with the specified list and language model
tools = load_tools(["wikipedia", "ddg-search", "llm-math"], llm)

# Create the agent with the language model, tools, and prompt
agent = create_react_agent(llm, tools, prompt)

# Create the agent executor with the agent, tools, verbose mode, and error handling
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Set the title of the Streamlit app
st.title("AI Agent")

# Capture user input using Streamlit's chat input widget
question = st.chat_input("Give me a task: ")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        for response in agent_executor.stream({"input": question}):
            # Agent Action
            if "actions" in response:
                for action in response["actions"]:
                    st.write(f"Calling Tool: `{action.tool}` with input `{action.tool_input}`")
            # Observation
            elif "steps" in response:
                for step in response["steps"]:
                    st.write(f"Tool Result: `{step.observation}`")
            # Final result
            elif "output" in response:
                st.write(f'Final Output: {response["output"]}')
            else:
                raise ValueError("Unexpected response format")
            st.write("---")
