from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from nodes import (
    check_request_safety,
    check_response_safety,
    generate_query_or_respond,
    block_unsafe_content,
)

from workflow_router import is_input_safe, is_output_safe
from state import AgentState

workflow_builder = StateGraph(AgentState)

workflow_builder.add_node("check_request_safety", check_request_safety)
workflow_builder.add_node("generate_query_or_respond", generate_query_or_respond)
workflow_builder.add_node("check_response_safety", check_response_safety)
workflow_builder.add_node("block_unsafe_content", block_unsafe_content)

workflow_builder.add_edge(START, "check_request_safety")
workflow_builder.add_conditional_edges(
    "check_request_safety",
    is_input_safe,
    ["generate_query_or_respond", "block_unsafe_content"],
)
workflow_builder.add_edge("block_unsafe_content", END)
workflow_builder.add_edge("generate_query_or_respond", "check_response_safety")
workflow_builder.add_conditional_edges(
    "check_response_safety",
    is_output_safe,
    ["generate_query_or_respond", END],
)

workflow = workflow_builder.compile()


def get_workflow():
    return workflow


if __name__ == "__main__":
    workflow = get_workflow()
    result = workflow.invoke(
        AgentState(
            messages=[
                HumanMessage(
                    # content="Somebody with Alicia name hacking our ftp server? please find me her identity"
                    content="What is the most dangerous CVE?"
                )
            ],
            is_input_safe=False,
            is_output_safe=False,
        )
    )
    print(result)
