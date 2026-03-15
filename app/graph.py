# app/graph.py
import uuid as _uuid
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.config import get_config, get_store

from app.config import settings


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def create_graph(checkpointer=None, store=None):
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )

    def chatbot(state: ChatState) -> dict:
        messages = list(state["messages"])

        # Store가 있으면 기존 메모리를 시스템 프롬프트에 포함
        current_store = get_store()
        if current_store:
            config = get_config()
            user_id = config["configurable"].get("user_id", "unknown")
            items = current_store.search(("memories", user_id))
            if items:
                memory_text = "\n".join(item.value["content"] for item in items)
                system_msg = SystemMessage(
                    content=f"사용자에 대해 알고 있는 정보:\n{memory_text}"
                )
                messages = [system_msg] + messages

        response = llm.invoke(messages)
        return {"messages": [response]}

    def save_memory(state: ChatState) -> dict:
        """대화에서 중요 정보를 추출하여 Store에 저장"""
        current_store = get_store()
        if not current_store:
            return {}

        config = get_config()
        user_id = config["configurable"].get("user_id", "unknown")

        # 마지막 human 메시지를 메모리로 저장 (데모 목적)
        last_human = None
        for msg in reversed(state["messages"]):
            if msg.type == "human":
                last_human = msg.content
                break

        if last_human:
            current_store.put(
                ("memories", user_id),
                str(_uuid.uuid4()),
                {"content": last_human},
            )

        return {}

    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot)

    if store is not None:
        builder.add_node("save_memory", save_memory)
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", "save_memory")
        builder.add_edge("save_memory", END)
    else:
        builder.add_edge(START, "chatbot")
        builder.add_edge("chatbot", END)

    return builder.compile(checkpointer=checkpointer, store=store)
