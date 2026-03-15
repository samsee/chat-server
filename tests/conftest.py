# tests/conftest.py
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

@pytest.fixture
def mock_ollama():
    """ChatOllama를 mock하여 실제 Ollama 서버 없이 테스트"""
    with patch("app.graph.ChatOllama") as mock_cls:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = lambda msgs: AIMessage(content="Mock 응답입니다.")
        mock_cls.return_value = mock_llm
        yield mock_llm
