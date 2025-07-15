import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # ChromaDB Configuration
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")
    chroma_collection_name: str = Field(default="rag_documents", env="CHROMA_COLLECTION_NAME")
    
    # Server Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8001, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # RAG Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_retrieval_docs: int = Field(default=5, env="MAX_RETRIEVAL_DOCS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Embedding Model
    embedding_model: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=3072, env="EMBEDDING_DIMENSIONS")
    
    # LLM Model
    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    
    class Config:
        # Look for .env file in the project root
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    return Settings()