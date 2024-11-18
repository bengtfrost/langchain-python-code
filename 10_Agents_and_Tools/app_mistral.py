from decouple import config
from langchain import hub
import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_mistralai import ChatMistralAI

MISTRAL_KEY = config("MISTRAL_KEY")

llm = ChatMistralAI(
    model="mistral-large-latest", mistral_api_key=MISTRAL_KEY
)

prompt = hub.pull("hwchase17/react")

tools = load_tools(["wikipedia", "ddg-search", "llm-math"], llm)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

st.title("AI Agent")

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