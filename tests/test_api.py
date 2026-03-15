# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.main import app, get_graph, get_store
from app.graph import create_graph

@pytest.fixture(autouse=True)
def override_graph(mock_ollama):
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    test_graph = create_graph(checkpointer=checkpointer, store=store)

    def _get_graph():
        return test_graph

    def _get_store():
        return store

    app.dependency_overrides[get_graph] = _get_graph
    app.dependency_overrides[get_store] = _get_store
    yield
    app.dependency_overrides.clear()

async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert "status" in resp.json()

async def test_chat_creates_thread():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json={
            "user_id": "user1",
            "message": "안녕"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert "response" in data
        assert data["message_index"] >= 0

async def test_chat_with_thread_id():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post("/chat", json={
            "user_id": "user1",
            "thread_id": "test-thread",
            "message": "안녕"
        })
        assert resp1.status_code == 200

        resp2 = await client.post("/chat", json={
            "user_id": "user1",
            "thread_id": "test-thread",
            "message": "잘 지내?"
        })
        assert resp2.status_code == 200
        assert resp2.json()["message_index"] > resp1.json()["message_index"]

async def test_history():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/chat", json={
            "user_id": "user1",
            "thread_id": "hist-thread",
            "message": "테스트"
        })

        resp = await client.get("/history/hist-thread")
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread_id"] == "hist-thread"
        assert len(data["messages"]) >= 2

async def test_history_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/history/nonexistent")
        assert resp.status_code == 404


async def test_get_memories():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/memories/user1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "user1"
        assert isinstance(data["memories"], list)


async def test_delete_memories():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/memories/user1")
        assert resp.status_code == 200
