from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")
reply = llm.invoke("In one sentence, what is a stock?")

print(reply.content)