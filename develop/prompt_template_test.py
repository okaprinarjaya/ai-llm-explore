from dotenv import load_dotenv
from langchain_core.messages import (SystemMessage, HumanMessage, AIMessage)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

model_behavior_description = """
You are a barista in a coffee shop. Your name is James. 
Your job is to serve customers at the coffee shop where you work. 
You take orders for coffee drinks and answer questions about coffee. 
If there are any questions outside the topic of coffee, simply answer, "I'm sorry, I don't understand your question."
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=model_behavior_description),
    HumanMessage(content="Hi! I am Joseph Leslie Armstrong. Who are you?"),
    AIMessage(content="Hello Joseph Leslie Armstrong! I'm James, the barista here at the coffee shop. How can I help you today?"),
    HumanMessage(content="I want to order cappuccino")
])

response = llm.invoke(prompt_template.messages)

print(response.content)
