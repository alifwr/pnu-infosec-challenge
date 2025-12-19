from typing import Optional, Any, Literal
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str


class MessageWithTool(Message):
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    messages: list[MessageWithTool]
