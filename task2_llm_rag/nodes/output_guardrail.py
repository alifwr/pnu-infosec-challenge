import sys
import os
import json
import re

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, AIMessage
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import init_chat_model


model = init_chat_model()


class GuardrailResult(BaseModel):
    is_output_safe: bool = Field(
        description="Whether the request is rejected due to safety concerns (e.g. PII requests)."
    )


def check_request_safety(state: MessagesState):
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

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            return {"is_output_safe": data.get("is_output_safe", False)}
        except json.JSONDecodeError:
            pass

    return {"is_output_safe": False}


if __name__ == "__main__":
    print(check_request_safety(MessagesState(messages=[AIMessage(content="Hello")])))
