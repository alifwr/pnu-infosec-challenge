from fastapi import FastAPI
from langchain_core.messages import HumanMessage, AIMessage

from schemas.api import ChatRequest, ChatResponse, DocumentResponse, RAGRequest
from tools.rag import retrieve_documents

from workflow import get_workflow

app = FastAPI()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [
        HumanMessage(content=m.content)
        if m.role == "user"
        else AIMessage(content=m.content)
        for m in request.messages
    ]

    workflow = get_workflow()

    result = workflow.invoke(
        {"messages": messages, "is_input_safe": True, "is_output_safe": True}
    )

    response_messages = []
    for m in result["messages"]:
        print(m.type)
        msg_dict = {
            "role": "assistant" if m.type == "ai" else "user",
            "content": str(m.content),
        }
        if m.type == "tool":
            msg_dict["role"] = "tool"
            msg_dict["tool_call_id"] = m.tool_call_id
            msg_dict["name"] = m.name
        elif m.type == "ai":
            if getattr(m, "tool_calls", None):
                msg_dict["tool_calls"] = m.tool_calls

        response_messages.append(msg_dict)

    return ChatResponse(messages=response_messages)


@app.post("/rag-query", response_model=list[DocumentResponse])
async def get_rag_query(request: RAGRequest):
    docs = retrieve_documents(request.query)
    return [
        DocumentResponse(content=doc["content"], metadata=doc["metadata"])
        for doc in docs
    ]
