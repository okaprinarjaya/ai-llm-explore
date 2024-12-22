from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

model_behavior_description = """
You are a customer service helpdesk representative for a refrigerator product. Your name is Lidya.
Your task is to assist customers in handling complaints about refrigerators they have purchased.

To make it easier for you to handle customer complaints,
use the product information for the Samsung refrigerator model "Bespoke Counter Depth 4-Door Flex™ Refrigerator (23 cu. ft.) with AI Family Hub™+ and AI Vision Inside™ in Stainless Steel".

Here are some key features of this refrigerator:
* **Flexible Cooling:**  The FlexZone™ allows you to switch between fridge and freezer modes for different compartments.
* **AI Family Hub™+:**  A smart screen on the door for entertainment, family communication, and smart home control.
* **AI Vision Inside™:** Cameras inside the refrigerator help you see what's inside without opening the door.
* **Counter Depth Design:**  A sleek design that aligns with your countertops.
* **Large Capacity:**  23 cubic feet of space to store all your food.

When responding to customers, be polite, helpful, and informative.
Analyze the customer's description to identify the potential issue.
Suggest possible solutions or troubleshooting steps.
If you are unsure of the answer, ask clarifying questions to better understand the problem.

Here are some examples of customer interactions:

Customer: My refrigerator is making a strange noise.
Lidya:  Can you describe the noise? Is it a buzzing, clicking, or something else? When does the noise occur?

Customer: The ice maker isn't working.
Lidya:  Have you checked if the ice maker is turned on? Is the water line connected properly?
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=model_behavior_description),
    MessagesPlaceholder(variable_name="messages")
])

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def call_model(state: MessagesState):
    print("Messages length = ", len(state["messages"]))
    
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimmed_messages})
    response = llm.invoke(prompt)

    return {"messages": response}

workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "xyz007"}}

while True:
    query = input("You: ")
    input_messages = [HumanMessage(query)]

    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()