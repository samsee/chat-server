# tests/test_graph.py
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import AIMessage
from app.graph import create_graph

def test_graph_single_message(mock_ollama):
    """그래프가 메시지를 처리하고 응답을 반환하는지 확인"""
    checkpointer = InMemorySaver()
    graph = create_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test-1"}}
    result = graph.invoke(
        {"messages": [("human", "안녕하세요")]},
        config
    )

    assert len(result["messages"]) == 2
    assert result["messages"][-1].type == "ai"

def test_graph_remembers_history(mock_ollama):
    """같은 thread_id로 여러 번 호출하면 히스토리가 누적되는지 확인"""
    checkpointer = InMemorySaver()
    graph = create_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test-2"}}

    graph.invoke({"messages": [("human", "내 이름은 철수야")]}, config)
    result = graph.invoke({"messages": [("human", "내 이름이 뭐야?")]}, config)

    assert len(result["messages"]) == 4  # 2 human + 2 ai

def test_graph_separate_threads(mock_ollama):
    """다른 thread_id는 독립적인 히스토리를 갖는지 확인"""
    checkpointer = InMemorySaver()
    graph = create_graph(checkpointer=checkpointer)

    config1 = {"configurable": {"thread_id": "thread-a"}}
    config2 = {"configurable": {"thread_id": "thread-b"}}

    graph.invoke({"messages": [("human", "A 대화")]}, config1)
    result = graph.invoke({"messages": [("human", "B 대화")]}, config2)

    assert len(result["messages"]) == 2  # B 스레드는 독립적


def test_graph_with_store_saves_memory(mock_ollama):
    """Store가 주입되면 대화에서 메모리를 추출하여 저장하는지 확인"""
    mock_ollama.invoke.side_effect = lambda msgs: AIMessage(
        content="안녕하세요 철수님! 반갑습니다."
    )

    checkpointer = InMemorySaver()
    store = InMemoryStore()
    graph = create_graph(checkpointer=checkpointer, store=store)

    config = {"configurable": {"thread_id": "mem-1", "user_id": "user1"}}
    graph.invoke(
        {"messages": [("human", "내 이름은 철수야")]},
        config,
    )

    items = store.search(("memories", "user1"))
    assert isinstance(items, list)
    assert len(items) > 0


def test_graph_without_store(mock_ollama):
    """Store 없이도 그래프가 정상 동작하는지 확인 (1단계 호환)"""
    checkpointer = InMemorySaver()
    graph = create_graph(checkpointer=checkpointer, store=None)

    config = {"configurable": {"thread_id": "no-store"}}
    result = graph.invoke(
        {"messages": [("human", "안녕")]},
        config,
    )
    assert len(result["messages"]) == 2


def test_graph_with_store_loads_memory(mock_ollama):
    """기존 메모리가 있을 때 시스템 프롬프트에 포함되는지 확인"""
    checkpointer = InMemorySaver()
    store = InMemoryStore()

    # 사전에 메모리 저장
    store.put(("memories", "user1"), "fact-1", {"content": "이름은 철수"})

    graph = create_graph(checkpointer=checkpointer, store=store)

    config = {"configurable": {"thread_id": "mem-load", "user_id": "user1"}}
    graph.invoke(
        {"messages": [("human", "안녕")]},
        config,
    )

    # LLM에 전달된 메시지에 시스템 프롬프트가 포함되었는지 확인
    call_args = mock_ollama.invoke.call_args[0][0]
    # 첫 번째 메시지가 SystemMessage이고 메모리 내용을 포함
    assert call_args[0].type == "system"
    assert "철수" in call_args[0].content
