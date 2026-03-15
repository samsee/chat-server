# app/main.py
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage

from app.config import settings
from app.schema import (
    ChatRequest, ChatResponse, HistoryResponse,
    MessageItem, ErrorResponse,
)
from app.graph import create_graph

_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    checkpointer = PostgresSaver.from_conn_string(settings.database_url)
    checkpointer.setup()
    _graph = create_graph(checkpointer=checkpointer)
    yield

app = FastAPI(title="Chat Server", lifespan=lifespan)

def get_graph():
    return _graph

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, graph=Depends(get_graph)):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config,
    )

    messages = result["messages"]
    ai_message = messages[-1]
    message_index = len(messages) - 1

    return ChatResponse(
        thread_id=thread_id,
        message_index=message_index,
        response=ai_message.content,
    )

@app.get(
    "/history/{thread_id}",
    response_model=HistoryResponse,
    responses={404: {"model": ErrorResponse}},
)
def history(thread_id: str, graph=Depends(get_graph)):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    if not state.values:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "detail": f"Thread '{thread_id}' not found"},
        )

    messages = []
    for i, msg in enumerate(state.values["messages"]):
        role = "human" if isinstance(msg, HumanMessage) else "ai"
        messages.append(MessageItem(role=role, content=msg.content, index=i))

    return HistoryResponse(thread_id=thread_id, messages=messages)
