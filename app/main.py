# app/main.py
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage

from app.config import settings
from app.schema import (
    ChatRequest, ChatResponse, HistoryResponse,
    MessageItem, ErrorResponse, MemoryItem, MemoriesResponse,
    ForkRequest, ForkResponse,
)
from app.graph import create_graph

_graph = None
_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _store
    checkpointer = PostgresSaver.from_conn_string(settings.database_url)
    checkpointer.setup()
    _store = InMemoryStore()
    _graph = create_graph(checkpointer=checkpointer, store=_store)
    yield

app = FastAPI(title="Chat Server", lifespan=lifespan)

def get_graph():
    return _graph

def get_store():
    return _store

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, graph=Depends(get_graph)):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id, "user_id": req.user_id}}

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

@app.get("/memories/{user_id}", response_model=MemoriesResponse)
def get_memories(user_id: str, store=Depends(get_store)):
    items = store.search(("memories", user_id))
    memories = [
        MemoryItem(key=item.key, value=item.value)
        for item in items
    ]
    return MemoriesResponse(user_id=user_id, memories=memories)

@app.delete("/memories/{user_id}")
def delete_memories(user_id: str, store=Depends(get_store)):
    items = store.search(("memories", user_id))
    for item in items:
        store.delete(("memories", user_id), item.key)
    return {"status": "ok", "deleted": len(items)}

@app.post(
    "/fork",
    response_model=ForkResponse,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
def fork_conversation(req: ForkRequest, graph=Depends(get_graph)):
    source_config = {"configurable": {"thread_id": req.source_thread_id}}
    state = graph.get_state(source_config)

    if not state.values:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "detail": f"Thread '{req.source_thread_id}' not found"},
        )

    messages = state.values["messages"]

    if req.message_index < 0 or req.message_index >= len(messages):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_index", "detail": f"message_index must be 0-{len(messages)-1}"},
        )

    # 새 thread에 해당 인덱스까지의 메시지를 주입
    new_thread_id = str(uuid.uuid4())
    new_config = {"configurable": {"thread_id": new_thread_id}}
    truncated_messages = messages[:req.message_index + 1]

    graph.update_state(
        new_config,
        {"messages": truncated_messages},
    )

    return ForkResponse(
        new_thread_id=new_thread_id,
        forked_from=req.source_thread_id,
        message_index=req.message_index,
    )
