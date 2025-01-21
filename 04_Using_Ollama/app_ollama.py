from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model="phi3.5:latest")

print("Q & A With AI")
print("=============")

question = "What's the currency of Thailand?"
print("Question: " + question)

response = llm.invoke(question)

print("Answer: " + response.content)
