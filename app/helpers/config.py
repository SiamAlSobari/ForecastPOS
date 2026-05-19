from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    api_token: str

    # LLM API Keys (minimal salah satu harus diisi untuk fitur AI Insights)
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()