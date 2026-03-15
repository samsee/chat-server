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
