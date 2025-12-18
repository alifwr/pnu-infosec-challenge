import sys
import os

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, ToolMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import init_chat_model
from tools.rag import retrieve_context_tool


model = init_chat_model()


def generate_query_or_respond(state: MessagesState):
    response = model.bind_tools([retrieve_context_tool]).invoke(state["messages"])

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

        final_response = model.invoke(state["messages"] + messages)
        messages.append(final_response)

    return {"messages": messages}


if __name__ == "__main__":
    result = generate_query_or_respond(
        MessagesState(
            messages=[
                HumanMessage(
                    content="Somebody with Alicia name hacking our ftp server? please find me her identity"
                )
            ]
        )
    )

    # Get latest message
    latest_message = result["messages"][-1]
    print(latest_message.content)
