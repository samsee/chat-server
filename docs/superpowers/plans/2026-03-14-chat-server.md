# Chat Server API Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FastAPI + LangGraph + Ollama + PostgreSQL 기반 채팅 서버 API 구현 (메모리 데모용)

**Architecture:** LangGraph StateGraph에 PostgresSaver checkpointer를 연결하여 대화 히스토리를 자동 관리. ChatOllama로 qwen3.5:cloud 모델 호출. 3단계 점진적 구현 (히스토리 → 크로스 세션 메모리 → 대화 분기).

**Tech Stack:** Python, FastAPI, LangGraph, langchain-ollama, langgraph-checkpoint-postgres, PostgreSQL (Docker), uvicorn

---

## File Structure

```
chat-server/
├── app/
│   ├── __init__.py       # 빈 파일
│   ├── main.py           # FastAPI 앱, 라우터, lifespan
│   ├── graph.py          # LangGraph StateGraph 정의
│   ├── config.py         # 설정 (DB URL, 모델명)
│   └── schema.py         # Pydantic 요청/응답 모델
├── tests/
│   ├── __init__.py
│   ├── conftest.py       # 공통 fixture (mock_ollama 등)
│   ├── test_config.py    # 설정 테스트
│   ├── test_schema.py    # 스키마 검증 테스트
│   ├── test_graph.py     # 그래프 단위 테스트
│   └── test_api.py       # API 통합 테스트
├── .gitignore
├── pyproject.toml        # pytest 설정
├── requirements.txt
└── README.md
```

---

## Chunk 1: 프로젝트 셋업 및 1단계 (대화 히스토리 메모리)

### Task 1: 프로젝트 초기화

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `app/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: requirements.txt 작성**

```
fastapi>=0.115.0
uvicorn>=0.34.0
pydantic-settings>=2.0.0
langchain-ollama>=0.3.0
langgraph>=0.3.0
langgraph-checkpoint-postgres>=2.0.0
psycopg[binary]>=3.2.0
httpx>=0.28.0
pytest>=8.0.0
pytest-asyncio>=0.25.0
```

- [ ] **Step 2: pyproject.toml 작성**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: .gitignore 작성**

```
__pycache__/
*.pyc
.env
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: 빈 __init__.py 파일 생성**

`app/__init__.py` 및 `tests/__init__.py` — 빈 파일

- [ ] **Step 5: 의존성 설치**

Run: `pip install -r requirements.txt`

- [ ] **Step 6: git 초기화 및 커밋**

```bash
git init
git add .gitignore requirements.txt pyproject.toml app/__init__.py tests/__init__.py
git commit -m "chore: initialize project with dependencies"
```

---

### Task 2: config.py — 설정

**Files:**
- Create: `app/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_config.py
from app.config import Settings

def test_default_settings():
    settings = Settings()
    assert settings.ollama_model == "qwen3.5:cloud"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert "postgresql" in settings.database_url

def test_custom_model():
    settings = Settings(ollama_model="llama3")
    assert settings.ollama_model == "llama3"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `cannot import name 'Settings'`

- [ ] **Step 3: config.py 구현**

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://chat:chat@localhost:5432/chat_memory"
    ollama_model: str = "qwen3.5:cloud"
    ollama_base_url: str = "http://localhost:11434"

    model_config = {"env_prefix": "CHAT_"}

settings = Settings()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add application settings with env var support"
```

---

### Task 3: schema.py — 요청/응답 모델

**Files:**
- Create: `app/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_schema.py
from app.schema import ChatRequest, ChatResponse, HistoryResponse, MessageItem, ErrorResponse

def test_chat_request_with_thread_id():
    req = ChatRequest(user_id="user1", message="hello", thread_id="t1")
    assert req.thread_id == "t1"

def test_chat_request_without_thread_id():
    req = ChatRequest(user_id="user1", message="hello")
    assert req.thread_id is None

def test_chat_response():
    resp = ChatResponse(thread_id="t1", message_index=1, response="hi")
    assert resp.message_index == 1

def test_history_response():
    msgs = [MessageItem(role="human", content="hi", index=0)]
    resp = HistoryResponse(thread_id="t1", messages=msgs)
    assert len(resp.messages) == 1

def test_error_response():
    err = ErrorResponse(error="not_found", detail="Thread not found")
    assert err.error == "not_found"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL

- [ ] **Step 3: schema.py 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_schema.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/schema.py tests/test_schema.py
git commit -m "feat: add Pydantic request/response schemas"
```

---

### Task 4: graph.py — LangGraph 그래프 정의

**Files:**
- Create: `app/graph.py`
- Create: `tests/conftest.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: conftest.py 작성**

```python
# tests/conftest.py
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

@pytest.fixture
def mock_ollama():
    """ChatOllama를 mock하여 실제 Ollama 서버 없이 테스트"""
    with patch("app.graph.ChatOllama") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Mock 응답입니다.")
        mock_cls.return_value = mock_llm
        yield mock_llm
```

- [ ] **Step 2: 테스트 작성**

```python
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
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL — `cannot import name 'create_graph'`

- [ ] **Step 4: graph.py 구현**

```python
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
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/test_graph.py -v`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add app/graph.py tests/test_graph.py tests/conftest.py
git commit -m "feat: add LangGraph chat graph with checkpointer support"
```

---

### Task 5: main.py — FastAPI 앱 및 엔드포인트

**Files:**
- Create: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: API 테스트 작성**

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from langgraph.checkpoint.memory import InMemorySaver

from app.main import app, get_graph
from app.graph import create_graph

@pytest.fixture(autouse=True)
def override_graph(mock_ollama):
    checkpointer = InMemorySaver()
    test_graph = create_graph(checkpointer=checkpointer)

    def _get_graph():
        return test_graph

    app.dependency_overrides[get_graph] = _get_graph
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_api.py -v`
Expected: FAIL

- [ ] **Step 3: main.py 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add FastAPI endpoints for chat and history"
```

---

### Task 6: README.md 및 1단계 완료

**Files:**
- Create: `README.md`

- [ ] **Step 1: README.md 작성**

````markdown
# Chat Server API

FastAPI + LangGraph + Ollama + PostgreSQL 기반 채팅 서버.

## 실행 방법

### 1. PostgreSQL (Docker)

```bash
docker pull postgres:17
docker run -d \
  --name chat-postgres \
  -e POSTGRES_USER=chat \
  -e POSTGRES_PASSWORD=chat \
  -e POSTGRES_DB=chat_memory \
  -p 5432:5432 \
  postgres:17
```

### 2. Ollama

```bash
ollama pull qwen3.5:cloud
ollama serve
```

### 3. 서버 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API

- `GET /health` — 상태 확인
- `POST /chat` — 메시지 전송
- `GET /history/{thread_id}` — 대화 히스토리 조회
- `GET /memories/{user_id}` — 사용자 장기 메모리 조회
- `DELETE /memories/{user_id}` — 사용자 메모리 초기화
- `POST /fork` — 대화 분기

## 환경 변수

- `CHAT_DATABASE_URL` — PostgreSQL 접속 URL (기본: `postgresql://chat:chat@localhost:5432/chat_memory`)
- `CHAT_OLLAMA_MODEL` — Ollama 모델명 (기본: `qwen3.5:cloud`)
- `CHAT_OLLAMA_BASE_URL` — Ollama 서버 URL (기본: `http://localhost:11434`)
````

- [ ] **Step 2: 전체 테스트 실행**

Run: `pytest -v`
Expected: ALL PASS

- [ ] **Step 3: 커밋**

```bash
git add README.md
git commit -m "docs: add README with setup instructions"
```

---

## Chunk 2: 2단계 (크로스 세션 메모리)

### Task 7: graph.py 확장 — Store 기반 장기 메모리

**Files:**
- Modify: `app/graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 1: 테스트 추가**

```python
# tests/test_graph.py 에 추가
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import AIMessage

def test_graph_with_store_saves_memory(mock_ollama):
    """Store가 주입되면 대화에서 메모리를 추출하여 저장하는지 확인"""
    mock_ollama.invoke.return_value = AIMessage(
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_graph.py::test_graph_with_store_saves_memory -v`
Expected: FAIL

- [ ] **Step 3: graph.py를 Store 지원으로 전면 수정**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_graph.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/graph.py tests/test_graph.py
git commit -m "feat: add Store-based long-term memory to graph"
```

---

### Task 8: schema.py 및 main.py — 메모리 엔드포인트

**Files:**
- Modify: `app/schema.py`
- Modify: `app/main.py`
- Modify: `tests/test_api.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: API 테스트 추가 (테스트 먼저)**

```python
# tests/test_api.py 에 추가
from app.main import get_store

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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_api.py::test_get_memories -v`
Expected: FAIL

- [ ] **Step 3: 스키마 추가**

```python
# app/schema.py 에 추가
class MemoryItem(BaseModel):
    key: str
    value: dict

class MemoriesResponse(BaseModel):
    user_id: str
    memories: list[MemoryItem]
```

- [ ] **Step 4: main.py 전체 수정 (store 추가, chat에 user_id 포함)**

```python
# app/main.py — 전체 파일
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
```

- [ ] **Step 5: conftest.py의 override fixture 수정 (store 포함)**

```python
# tests/conftest.py — 전체 파일
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

@pytest.fixture
def mock_ollama():
    """ChatOllama를 mock하여 실제 Ollama 서버 없이 테스트"""
    with patch("app.graph.ChatOllama") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Mock 응답입니다.")
        mock_cls.return_value = mock_llm
        yield mock_llm
```

```python
# tests/test_api.py — override_graph fixture 수정
from langgraph.store.memory import InMemoryStore
from app.main import app, get_graph, get_store

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
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 7: 커밋**

```bash
git add app/schema.py app/main.py tests/test_api.py tests/conftest.py
git commit -m "feat: add memory endpoints (GET/DELETE /memories/{user_id})"
```

---

## Chunk 3: 3단계 (대화 분기)

### Task 9: Fork 스키마 및 엔드포인트

**Files:**
- Modify: `app/schema.py`
- Modify: `app/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Fork 테스트 작성 (테스트 먼저)**

```python
# tests/test_api.py 에 추가

async def test_fork_conversation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 원본 대화 생성
        await client.post("/chat", json={
            "user_id": "user1", "thread_id": "fork-src", "message": "첫 번째"
        })
        await client.post("/chat", json={
            "user_id": "user1", "thread_id": "fork-src", "message": "두 번째"
        })

        # message_index=1에서 분기 (첫 번째 AI 응답까지)
        resp = await client.post("/fork", json={
            "source_thread_id": "fork-src",
            "message_index": 1
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["forked_from"] == "fork-src"
        assert data["message_index"] == 1
        new_thread_id = data["new_thread_id"]

        # 분기된 대화의 히스토리 확인
        hist = await client.get(f"/history/{new_thread_id}")
        assert hist.status_code == 200
        assert len(hist.json()["messages"]) == 2  # index 0, 1만 복사됨

async def test_fork_invalid_thread():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/fork", json={
            "source_thread_id": "nonexistent",
            "message_index": 0
        })
        assert resp.status_code == 404

async def test_fork_invalid_index():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/chat", json={
            "user_id": "user1", "thread_id": "fork-idx", "message": "hello"
        })
        resp = await client.post("/fork", json={
            "source_thread_id": "fork-idx",
            "message_index": 99
        })
        assert resp.status_code == 400
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_api.py::test_fork_conversation -v`
Expected: FAIL

- [ ] **Step 3: Fork 스키마 추가**

```python
# app/schema.py 에 추가
class ForkRequest(BaseModel):
    source_thread_id: str
    message_index: int

class ForkResponse(BaseModel):
    new_thread_id: str
    forked_from: str
    message_index: int
```

- [ ] **Step 4: Fork 엔드포인트 구현**

```python
# app/main.py 에 추가
from app.schema import ForkRequest, ForkResponse

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
    # 새 thread에는 기존 상태가 없으므로 add_messages 리듀서가
    # 빈 리스트에 추가하는 것과 같아 정상 동작함
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
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 6: 전체 테스트 실행**

Run: `pytest -v`
Expected: ALL PASS

- [ ] **Step 7: 커밋**

```bash
git add app/main.py app/schema.py tests/test_api.py
git commit -m "feat: add /fork endpoint for conversation branching"
```

---

### Task 10: 최종 정리

- [ ] **Step 1: 전체 테스트 최종 확인**

Run: `pytest -v`
Expected: ALL PASS

- [ ] **Step 2: 최종 커밋**

```bash
git add -A
git commit -m "chore: final cleanup"
```
