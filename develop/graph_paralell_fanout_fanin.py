import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]

class MyNode:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State):
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}
    
graph_builder = StateGraph(State)

graph_builder.add_node("a", MyNode("I'm A"))
graph_builder.add_node("b", MyNode("I'm B"))
graph_builder.add_node("c", MyNode("I'm C"))
graph_builder.add_node("d", MyNode("I'm D"))

graph_builder.add_edge(START, "a")
graph_builder.add_edge("a", "b")
graph_builder.add_edge("a", "c")
graph_builder.add_edge("b", "d")
graph_builder.add_edge("c", "d")
graph_builder.add_edge("d", END)

graph = graph_builder.compile()

graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "foobar"}})
