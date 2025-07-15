from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_index: int
    total_chunks: int
    created_at: datetime = Field(default_factory=datetime.now)
    file_type: str
    size: int
    
    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # Convert datetime to string for ChromaDB
        if 'created_at' in data:
            data['created_at'] = data['created_at'].isoformat()
        # Remove None values for ChromaDB compatibility
        return {k: v for k, v in data.items() if v is not None}


class Document(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None


class QueryRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=20)
    include_metadata: bool = True
    similarity_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class RetrievedDocument(BaseModel):
    content: str
    metadata: DocumentMetadata
    similarity_score: float


class RAGResponse(BaseModel):
    answer: str
    sources: List[RetrievedDocument]
    query: str
    response_time: float
    model_used: str
    total_tokens: Optional[int] = None


class UploadResponse(BaseModel):
    message: str
    document_id: str
    chunks_created: int
    status: str = "success"


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    chroma_connected: bool
    openai_configured: bool


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = ["\n\n", "\n", " ", ""]


class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    batch_size: int = 100


class LLMConfig(BaseModel):
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = (
        "You are a helpful assistant. Answer the question based on the provided context. "
        "If you cannot answer based on the context, say so clearly."
    )


class RAGConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    max_retrieval_docs: int = 5
    similarity_threshold: float = 0.7