from langgraph.graph import StateGraph, MessagesState, START, END

from nodes import check_request_safety, check_response_safety, generate_query_or_respond

from .workflow_router import is_input_safe, is_output_safe

workflow = StateGraph(MessagesState)

workflow.add_node("check_request_safety", check_request_safety)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("check_response_safety", check_response_safety)

workflow.add_edge(START, "check_request_safety")
workflow.add_conditional_edges(
    "check_request_safety", is_input_safe, ["generate_query_or_respond", END]
)
workflow.add_edge("generate_query_or_respond", "check_response_safety")
workflow.add_conditional_edges(
    "check_response_safety", is_output_safe, ["generate_query_or_respond", END]
)
