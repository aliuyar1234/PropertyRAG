"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "PropertyRAG"
    debug: bool = False

    # Database
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/propertyrag"
    )

    # OpenAI
    openai_api_key: str = Field(default="")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"

    # Embedding dimensions for text-embedding-3-small
    embedding_dimensions: int = 1536

    # Chunking
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens

    # Retrieval
    retrieval_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
