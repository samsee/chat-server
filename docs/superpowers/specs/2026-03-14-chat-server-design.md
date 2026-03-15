# Chat Server API 설계 문서

## 목적

FastAPI + LangGraph + Ollama + PostgreSQL을 사용한 채팅 서버 API.
LangGraph의 메모리 시스템(대화 히스토리, 크로스 세션 메모리, 대화 분기)을 학습/데모하는 것이 목적.

## 기술 스택

- **FastAPI** — HTTP API 서버
- **LangGraph** — 대화 그래프 및 메모리 관리
- **langchain-ollama** — LLM 호출 (`ChatOllama`, 모델: `qwen3.5:cloud`, config에서 변경 가능)
- **langgraph-checkpoint-postgres** — PostgreSQL 체크포인터 (`PostgresSaver`)
- **PostgreSQL** — 메모리 영구 저장 (Docker 컨테이너로 실행)

## 아키텍처

```
Client (HTTP) → FastAPI → LangGraph StateGraph → ChatOllama (qwen3.5:cloud)
                                ↕
                       PostgresSaver (checkpointer)
                       + Store (2단계)
                                ↕
                          PostgreSQL (Docker)
```

## 프로젝트 구조

```
chat-server/
├── app/
│   ├── main.py          # FastAPI 앱, 엔드포인트
│   ├── graph.py         # LangGraph 그래프 정의
│   ├── config.py        # 설정 (DB URL, 모델명 등)
│   └── schema.py        # Pydantic 요청/응답 모델
├── requirements.txt
└── README.md            # 실행 방법 (Docker, Ollama 등)
```

## 사용자 구분

인증 없음. 요청 시 `user_id` 파라미터로 사용자를 구분한다.

## 공통 사항

### 에러 응답 형식

모든 에러는 다음 형식으로 응답한다:

```json
{
  "error": "not_found",
  "detail": "Thread 'thread-abc' not found"
}
```

### LangGraph 그래프 구조

**State 스키마:**
```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

**1단계 그래프:** `chatbot` 노드 1개 (메시지를 받아 LLM 호출 후 응답 반환)

**2단계 그래프:** `chatbot` 노드 + `save_memory` 노드 추가 (LLM 응답 후 중요 정보 추출하여 Store에 저장)

## 단계별 구현 로드맵

### 1단계: 대화 히스토리 메모리

같은 세션(thread) 내에서 이전 대화 맥락을 기억.

- LangGraph `PostgresSaver`를 checkpointer로 사용
- `thread_id`로 세션 식별
- 같은 `thread_id`로 요청하면 이전 대화를 기억
- 다른 `thread_id`면 새 대화 시작

**엔드포인트:**

| Method | Path                   | 설명                     |
|--------|------------------------|--------------------------|
| GET    | `/health`              | 서버/DB/Ollama 상태 확인 |
| POST   | `/chat`                | 메시지 전송 및 응답 받기 |
| GET    | `/history/{thread_id}` | 대화 히스토리 조회       |

**POST /chat:**

```json
// Request
{
  "user_id": "user1",
  "thread_id": "thread-abc",   // 생략 시 자동 생성
  "message": "안녕하세요"
}

// Response
{
  "thread_id": "thread-abc",
  "message_index": 1,
  "response": "안녕하세요! 무엇을 도와드릴까요?"
}
```

- `message_index`: 현재 응답의 인덱스 (3단계 fork에서 분기 지점으로 사용)

**GET /history/{thread_id}:**

```json
{
  "thread_id": "thread-abc",
  "messages": [
    {"role": "human", "content": "안녕하세요", "index": 0},
    {"role": "ai", "content": "안녕하세요! 무엇을 도와드릴까요?", "index": 1}
  ]
}
```

**데이터 흐름 (POST /chat):**

1. FastAPI가 요청 수신
2. `thread_id`를 config로 LangGraph 그래프 invoke
3. PostgresSaver가 해당 `thread_id`의 기존 상태를 DB에서 로드
4. 기존 메시지 + 새 메시지를 ChatOllama에 전달
5. 응답을 상태에 추가, PostgresSaver가 자동으로 DB에 저장
6. 응답 반환

### 2단계: 크로스 세션 메모리

세션이 끝나도 사용자에 대한 정보를 영구 저장하여 다음 대화에서도 활용.

**메모리 저장 방식:**
- LLM이 대화에서 중요 정보를 자동 추출 (이름, 선호도, 관심사 등)
- LangGraph `InMemoryStore` 사용 (데모 목적, 영구 저장이 필요하면 커스텀 PostgreSQL store로 교체 가능)
- Store namespace: `("memories", <user_id>)` — 사용자별 메모리 격리

**그래프 변경:**
- `chatbot` 노드에서 Store를 주입받아 기존 메모리를 시스템 프롬프트에 포함
- `save_memory` 노드를 추가하여 LLM 응답 후 중요 정보를 추출, Store에 저장

**추가 엔드포인트:**

| Method | Path                    | 설명                       |
|--------|-------------------------|----------------------------|
| GET    | `/memories/{user_id}`   | 사용자의 저장된 메모리 조회 |
| DELETE | `/memories/{user_id}`   | 사용자의 메모리 초기화      |

**GET /memories/{user_id}:**

```json
{
  "user_id": "user1",
  "memories": [
    {"key": "name", "value": "김철수"},
    {"key": "preference", "value": "매운 음식을 좋아함"}
  ]
}
```

**데이터 흐름 (POST /chat, 2단계):**

1. FastAPI가 요청 수신
2. Store에서 해당 `user_id`의 기존 메모리 조회
3. 기존 메모리를 시스템 프롬프트에 포함하여 LLM 호출
4. LLM 응답 후 `save_memory` 노드에서 새로운 중요 정보 추출
5. 추출된 정보를 Store에 저장
6. 응답 반환

### 3단계: 대화 분기

특정 대화 시점에서 새 세션으로 분기(fork).

- 해당 `message_index`까지의 메시지를 가져와 새 `thread_id`의 초기 상태로 주입
- 이후 새 `thread_id`로 대화를 이어가면 독립된 세션

**엔드포인트:**

| Method | Path    | 설명                           |
|--------|---------|--------------------------------|
| POST   | `/fork` | 특정 시점에서 새 세션으로 분기  |

**POST /fork:**

```json
// Request
{
  "source_thread_id": "thread-abc",
  "message_index": 3
}

// Response
{
  "new_thread_id": "thread-xyz",
  "forked_from": "thread-abc",
  "message_index": 3
}
```

- `message_index`: `/history`에서 확인한 메시지 인덱스. 해당 인덱스까지의 메시지를 포함하여 분기.
- 존재하지 않는 `source_thread_id`나 범위를 벗어난 `message_index`는 에러 반환.

## 인프라 실행

**PostgreSQL (Docker):**

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

**Ollama:**

```bash
ollama pull qwen3.5:cloud
ollama serve
```
