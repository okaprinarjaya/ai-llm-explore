from dotenv import load_dotenv
import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        
        return {"messages": outputs}

tool_tavily_search = TavilySearchResults(max_results=2)
tools = [tool_tavily_search]

memory = MemorySaver()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

# Node
def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Node
tool_node = BasicToolNode(tools=tools)

# Edge decision to next node
def edge_route_decision(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    
    return END

# Design / setup the graph (workflow)
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    edge_route_decision,
    {"tools": "tools", END: END}
)

graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(human_input: str):
    events = graph.stream({"messages": [HumanMessage(content=human_input)]}, config=config)
    for event in events:
        for value in event.values():
            print("AI Assistant:", value["messages"][-1].content)

while True:
    human_input = input("Human (You): ")

    if human_input.lower() in ["quit", "exit", "q"]:
        print("Bye!")
        break
    if len(human_input) < 5:
        continue
    else:
        stream_graph_updates(human_input)
