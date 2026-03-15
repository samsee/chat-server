# tests/test_graph.py
from langgraph.checkpoint.memory import InMemorySaver
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
