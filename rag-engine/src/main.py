import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import tempfile
import os

from .core.models import (
    QueryRequest, 
    RAGResponse, 
    HealthResponse, 
    UploadResponse,
    ErrorResponse
)
from .core.config import get_settings
from .core.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG engine instance
rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_engine
    logger.info("Starting RAG Engine...")
    
    try:
        rag_engine = RAGEngine()
        logger.info("RAG Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Engine...")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Engine API",
    description="A Retrieval-Augmented Generation API built with LangChain",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings dependency
def get_app_settings():
    return get_settings()


@app.get("/", response_model=dict)
async def root():
    return {
        "message": "RAG Engine API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        health_status = rag_engine.health_check()
        
        return HealthResponse(
            chroma_connected=health_status["vector_db"],
            openai_configured=health_status["embeddings"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/stats")
async def get_stats():
    try:
        return rag_engine.get_stats()
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")


@app.get("/debug/documents")
async def debug_documents():
    try:
        # Get all documents from ChromaDB
        collection = rag_engine.vector_db._collection
        results = collection.get(include=["documents", "metadatas"])
        
        return {
            "total_documents": len(results["documents"]) if results["documents"] else 0,
            "documents": [
                {
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,
                    "metadata": metadata
                }
                for doc, metadata in zip(results["documents"], results["metadatas"])
            ] if results["documents"] else []
        }
        
    except Exception as e:
        logger.error(f"Debug documents failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug documents failed: {str(e)}")


@app.get("/debug/search")
async def debug_search(query: str = "artificial intelligence", threshold: float = 0.0):
    try:
        # Generate embedding for the query
        query_embedding = rag_engine.embedding_service.embed_text(query)
        
        # Search without threshold filtering
        retrieved_docs = rag_engine.vector_db.similarity_search(
            query_embedding=query_embedding,
            k=10,
            similarity_threshold=threshold
        )
        
        return {
            "query": query,
            "threshold": threshold,
            "results_count": len(retrieved_docs),
            "results": [
                {
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "similarity_score": doc.similarity_score,
                    "source": doc.metadata.source
                }
                for doc in retrieved_docs
            ]
        }
        
    except Exception as e:
        logger.error(f"Debug search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug search failed: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_extensions = ['.txt', '.pdf', '.docx', '.doc']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process the document
        document_ids = await rag_engine.ingest_document(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return UploadResponse(
            message=f"Document '{file.filename}' uploaded and processed successfully",
            document_id=document_ids[0] if document_ids else "unknown",
            chunks_created=len(document_ids)
        )
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


class TextUploadRequest(BaseModel):
    text: str
    source: str = "api_upload"

@app.post("/upload-text", response_model=UploadResponse)
async def upload_text(request: TextUploadRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        document_ids = await rag_engine.ingest_text(request.text, request.source)
        
        return UploadResponse(
            message="Text uploaded and processed successfully",
            document_id=document_ids[0] if document_ids else "unknown",
            chunks_created=len(document_ids)
        )
        
    except Exception as e:
        logger.error(f"Failed to upload text: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


@app.post("/query", response_model=RAGResponse)
async def query_documents(query_request: QueryRequest):
    if not query_request.query.strip():
        raise HTTPException(status_code=400, detail="Empty query provided")
    
    try:
        response = await rag_engine.query(query_request)
        return response
        
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.delete("/documents")
async def clear_documents():
    try:
        success = rag_engine.vector_db.clear_collection()
        if success:
            return {"message": "All documents cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear documents")
            
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    error_response = ErrorResponse(
        error=exc.detail,
        detail=f"HTTP {exc.status_code}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    error_response = ErrorResponse(
        error="Internal server error",
        detail=str(exc)
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )