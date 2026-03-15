# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://chat:chat@localhost:5432/chat_memory"
    ollama_model: str = "qwen3.5:cloud"
    ollama_base_url: str = "http://localhost:11434"

    model_config = {"env_prefix": "CHAT_"}

settings = Settings()
