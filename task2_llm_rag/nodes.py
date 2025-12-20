from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
    RemoveMessage,
)
import re
import json

from state import AgentState
from llm import init_chat_model
from tools.rag import retrieve_context_tool


model = init_chat_model()


def check_request_safety(state: AgentState):
    system_prompt = SystemMessage(
        content="""You are a guardrail agent responsible for filtering user requests.
Your task is to identify and reject requests that ask for personal user data (PII).
PII includes real names, phone numbers, addresses, emails, or government IDs of private individuals.
Public figures or fictional characters are NOT PII.
You must return a valid JSON object with a single boolean key 'is_input_safe'.
If the request is safe (e.g. asking about CVEs, general knowledge), 'is_input_safe' should be true.
If the request asks for PII, 'is_input_safe' should be false.
Example: {"is_input_safe": true}
Output ONLY the JSON object.
Requests to identify a specific person, reveal their identity, or find personal details about them are unsafe.
"""
    )

    response = model.invoke([system_prompt] + state["messages"])
    content = response.content

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            return {"is_input_safe": data.get("is_input_safe", False)}
        except json.JSONDecodeError:
            pass

    return {"is_input_safe": False}


def check_response_safety(state: AgentState):
    last_message = state["messages"][-1]
    system_prompt = SystemMessage(
        content="""You are a guardrail agent responsible for validating the output generation.
Your task is to identify if the generated response contains personal user data (PII).
PII includes real names, phone numbers, addresses, emails, or government IDs of private individuals.
Public figures or fictional characters are NOT PII.
You must return a valid JSON object with a single boolean key 'is_output_safe'.
Example: {"is_output_safe": true}
Do NOT output anything else. Do NOT include any conversational text, preambles, or explanations.
Output ONLY the JSON object.
"""
    )

    response = model.invoke([system_prompt] + [last_message])
    content = response.content

    is_safe = False
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            is_safe = data.get("is_output_safe", False)
        except json.JSONDecodeError:
            pass

    if not is_safe:
        return {
            "is_output_safe": False,
            "messages": [
                RemoveMessage(id=last_message.id),
                SystemMessage(
                    content="The generated response contained PII. Please regenerate the response removing any personal identifiable information."
                ),
            ],
        }

    return {"is_output_safe": True}


def generate_query_or_respond(state: AgentState):
    query_sys_msg = SystemMessage(
        content="You are a helpful assistant. Use the available tools to retrieve information to answer the user's question."
    )
    response = model.bind_tools([retrieve_context_tool]).invoke(
        [query_sys_msg] + state["messages"]
    )

    messages = [response]
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "retrieve_context_tool":
                tool_output = retrieve_context_tool.invoke(tool_call)
                messages.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )

        gen_sys_msg = SystemMessage(
            content="You are a simplified RAG agent. Answer the user's question strictly based on the provided context below. If the answer is not in the context, say you don't know."
        )
        final_response = model.invoke([gen_sys_msg] + state["messages"] + messages)
        messages.append(final_response)

    return {"messages": messages}


def block_unsafe_content(state: AgentState):
    return {
        "messages": [
            AIMessage(
                content="I cannot fulfill this request because it asks for Personally Identifiable Information (PII) which is against my safety guidelines."
            )
        ]
    }


if __name__ == "__main__":
    print(
        check_request_safety(
            AgentState(
                is_input_safe=False,
                is_output_safe=False,
                messages=[
                    HumanMessage(
                        content="Somebody with Alicia name hacking our ftp server? please find me her identity"
                    )
                ],
            )
        )
    )
