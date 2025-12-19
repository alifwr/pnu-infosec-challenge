from langgraph.graph import MessagesState, END
from typing import Literal


def is_input_safe(state: MessagesState) -> Literal[END, "generate_query_or_respond"]:
    if state["is_input_safe"]:
        return "generate_query_or_respond"
    return END


def is_output_safe(state: MessagesState) -> Literal[END, "generate_query_or_respond"]:
    if state["is_output_safe"]:
        return END
    return "generate_query_or_respond"
