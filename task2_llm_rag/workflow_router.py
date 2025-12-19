from langgraph.graph import END
from typing import Literal

from state import AgentState


def is_input_safe(
    state: AgentState,
) -> Literal["block_unsafe_content", "generate_query_or_respond"]:
    if state["is_input_safe"]:
        return "generate_query_or_respond"
    return "block_unsafe_content"


def is_output_safe(
    state: AgentState,
) -> Literal["block_unsafe_content", "generate_query_or_respond"]:
    if state["is_output_safe"]:
        return END
    return "generate_query_or_respond"
