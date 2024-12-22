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
graph_builder.add_node("a", MyNode("I am A"))
graph_builder.add_edge(START, "a")

graph_builder.add_node("b", MyNode("I am B"))
graph_builder.add_node("b2", MyNode("I am B2"))
graph_builder.add_node("c", MyNode("I am C"))
graph_builder.add_node("d", MyNode("I am D"))
graph_builder.add_edge("a", "b")
graph_builder.add_edge("a", "c")
graph_builder.add_edge("b", "b2")
graph_builder.add_edge(["b2", "c"], "d")
graph_builder.add_edge("d", END)

graph = graph_builder.compile()

output = graph.invoke({"aggregate": []})
print(output)