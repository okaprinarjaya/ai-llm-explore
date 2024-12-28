from dotenv import load_dotenv
from typing import Callable

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition

from langgraph_tool01_state import State
from langgraph_tool01_agents import PrimaryAgent, CoffeeBaristaAgent

load_dotenv()


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig) -> State:
        result = self.runnable.invoke(state)
        return {"messages": result}


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {
                        assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {
                        assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state
        }

    return entry_node


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
memory = MemorySaver()


def route_primary_assistant(state: State):
    route = tools_condition(state)
    if route == END:
        return END

    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == "tool_coffee_barista":
            return "enter_coffee_barista"
        return END
    return END


def route_coffee_barista(state: State):
    route = tools_condition(state)
    if route == END:
        return END

    return "primary_assistant"


# Build graph
graph_builder = StateGraph(State)

graph_builder.add_edge(START, "primary_assistant")
graph_builder.add_edge("enter_coffee_barista", "coffee_barista")
graph_builder.add_edge("primary_assistant", END)

graph_builder.add_node("primary_assistant",
                       Assistant(PrimaryAgent(llm).create()))
graph_builder.add_node("coffee_barista", Assistant(
    CoffeeBaristaAgent(llm).runnable))
graph_builder.add_node("enter_coffee_barista", create_entry_node(
    "Coffee Barista Assistant", "coffee_barista"))

graph_builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant
)

graph = graph_builder.compile(checkpointer=memory)


# Run
config = {"configurable": {"thread_id": "xyz007"}}
while True:
    message = input("You: ")
    events = graph.stream(
        {"messages": HumanMessage(content=message)}, config=config)
    for event in events:
        for value in event.values():
            # message_last = value["messages"][-1]
            print(value)
