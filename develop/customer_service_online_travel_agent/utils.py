from typing import Callable

from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

from state_cs_ota import State

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls

    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"]
            ) for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools=tools).with_fallbacks([RunnableLambda(handle_tool_error)], exception_key="error")

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node


def print_with_tools_information(_printed: set, message, msg_repr):
    print(msg_repr)
    _printed.add(message.id)

def print_final_information_only(_printed: set, message, msg_repr):
    if isinstance(message, ToolMessage):
        _printed.add(message.id)
    if isinstance(message, HumanMessage):
        print(msg_repr)
        _printed.add(message.id)
    if isinstance(message, AIMessage) and not message.tool_calls:
        print(msg_repr)
        _printed.add(message.id)

def print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")

    if current_state:
        print("Currently in: ", current_state[-1])

    messages = event.get("messages")

    if messages:
        if isinstance(messages, list):
            message = messages[-1]
        if message and message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + "... (truncated)"
            
            print_with_tools_information(_printed, message=message, msg_repr=msg_repr)
            # print_final_information_only(_printed, message=message, msg_repr=msg_repr)
