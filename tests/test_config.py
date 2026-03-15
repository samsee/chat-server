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
