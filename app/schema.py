# app/schema.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: str
    message: str
    thread_id: str | None = None

class ChatResponse(BaseModel):
    thread_id: str
    message_index: int
    response: str

class MessageItem(BaseModel):
    role: str
    content: str
    index: int

class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[MessageItem]

class ErrorResponse(BaseModel):
    error: str
    detail: str


class MemoryItem(BaseModel):
    key: str
    value: dict


class MemoriesResponse(BaseModel):
    user_id: str
    memories: list[MemoryItem]
