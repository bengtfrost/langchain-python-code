from decouple import config
from langchain_mistralai import ChatMistral

MISTRAL_KEY = config("MISTRAL_KEY")

llm = ChatMistral(model="mistral-large-latest", mistral_api_key=MISTRAL_KEY)

print("Q & A With AI")
print("=============")

question = "What's the currency of Thailand?"
print("Question: " + question)

response = llm.invoke(question)

print("Answer: " + response.content)
