import sys
import os
import json
import re
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import init_chat_model


model = init_chat_model()


class GuardrailResult(BaseModel):
    is_input_safe: bool = Field(
        description="Whether the request is rejected due to safety concerns (e.g. PII requests)."
    )


def check_request_safety(state: MessagesState):
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


if __name__ == "__main__":
    print(
        check_request_safety(
            MessagesState(
                messages=[
                    HumanMessage(
                        content="Somebody with Alicia name hacking our ftp server? please find me her identity"
                    )
                ]
            )
        )
    )
