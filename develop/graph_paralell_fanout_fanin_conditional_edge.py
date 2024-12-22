import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    which: str

class MyNode:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: State):
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}

def decision_edge_to(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]

intermediates = ["b", "c", "d"]

graph_builder = StateGraph(State)
graph_builder.add_node("a", MyNode("I am A"))
graph_builder.add_node("b", MyNode("I am B"))
graph_builder.add_node("c", MyNode("I am C"))
graph_builder.add_node("d", MyNode("I am D"))
graph_builder.add_node("e", MyNode("I am E"))

graph_builder.add_edge(START, "a")

graph_builder.add_conditional_edges(
    "a",
    decision_edge_to,
    intermediates
)

for node in intermediates:
    graph_builder.add_edge(node, "e")

graph_builder.add_edge("e", END)

graph = graph_builder.compile()

output = graph.invoke({"aggregate": [], "which": "cd"})

print(output)
