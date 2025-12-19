from fastapi import FastAPI

from schema import ChatRequest, ChatResponse

from workflow import get_workflow

app = FastAPI()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = request.messages
    workflow = get_workflow()

    result = workflow.invoke(messages)
    return ChatResponse(messages=result)
