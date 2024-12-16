from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import (HumanMessage)

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

response = llm.invoke([
    HumanMessage(content="Explain to me shortly like I am 5 years old what is trigonometry in math?")
])

print(response)
