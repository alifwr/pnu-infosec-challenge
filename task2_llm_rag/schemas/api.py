from pydantic import BaseModel


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]


class ChatResponse(BaseModel):
    messages: list[dict[str, str]]
