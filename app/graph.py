# app/graph.py
from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from app.config import settings


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def create_graph(checkpointer=None):
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )

    def chatbot(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = (
        StateGraph(ChatState)
        .add_node("chatbot", chatbot)
        .add_edge(START, "chatbot")
        .add_edge("chatbot", END)
        .compile(checkpointer=checkpointer)
    )

    return graph
