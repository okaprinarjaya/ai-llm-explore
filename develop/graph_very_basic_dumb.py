from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

def satu(state, config: RunnableConfig):
    messages = state["messages"]
    messages.append("satu command")

    return {"messages": messages}

def dua(state):
    messages = state["messages"]
    messages.append("dua command")

    return {"messages": messages}

def tiga(state):
    messages = state["messages"]
    messages.append("tiga command")

    return {"messages": messages}

graph_builder = StateGraph(dict)

graph_builder.add_node("satu", satu)
graph_builder.add_node("dua", dua)
graph_builder.add_node("tiga", tiga)

graph_builder.add_edge(START, "satu")
graph_builder.add_edge("satu", "dua")
graph_builder.add_edge("dua", "tiga")
graph_builder.add_edge("tiga", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "xyz007"}}

results = graph.invoke({"messages": []}, config=config)

print(results["messages"])

# while True:
#     str_input = input("Input: ")
#     results = graph.invoke({"messages": []}, config=config)

#     print(results["messages"])
