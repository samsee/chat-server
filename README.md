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
