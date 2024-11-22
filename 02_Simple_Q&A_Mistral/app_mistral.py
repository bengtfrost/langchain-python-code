from decouple import config
from langchain_mistralai import ChatMistral

# Ensure that the MISTRAL_KEY is correctly set in the environment variables
MISTRAL_KEY = config("MISTRAL_KEY")

# Initialize the language model with the specified model and API key
llm = ChatMistral(model="mistral-large-latest", mistral_api_key=MISTRAL_KEY)

# Print the header for the Q&A session
print("Q & A With AI")
print("=============")

# Hardcoded question for the example
question = "What's the currency of Thailand?"
print("Question: " + question)

# Invoke the language model to get the response
response = llm.invoke(question)

# Print the answer
print("Answer: " + response.content)
