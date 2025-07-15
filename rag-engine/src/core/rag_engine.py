import time
import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from .models import (
    QueryRequest, 
    RAGResponse, 
    Document, 
    RetrievedDocument
)
from .config import get_settings
from ..embeddings.openai.service import OpenAIEmbeddingService
from ..vectordb.chromadb.client import ChromaDBClient
from ..retrieval.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = OpenAIEmbeddingService()
        self.vector_db = ChromaDBClient()
        self.document_processor = DocumentProcessor()
        self.llm = ChatOpenAI(
            api_key=self.settings.openai_api_key,
            model=self.settings.llm_model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens
        )
        
        # System prompt for RAG
        self.system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Guidelines:
1. Use only the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise and accurate in your responses
4. If you're uncertain about something, express that uncertainty
5. Cite relevant parts of the context when possible

Context:
{context}

Question: {question}

Answer:"""

    async def ingest_document(self, file_path: str) -> List[str]:
        try:
            # Process the document
            documents = self.document_processor.load_document(file_path)
            
            # Generate embeddings for each chunk
            texts = [doc.content for doc in documents]
            embeddings = self.embedding_service.embed_texts(texts)
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            # Store in vector database
            document_ids = self.vector_db.add_documents(documents)
            
            logger.info(f"Successfully ingested document: {file_path} ({len(documents)} chunks)")
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {e}")
            raise

    async def ingest_text(self, text: str, source: str = "direct_input") -> List[str]:
        try:
            # Process the text
            documents = self.document_processor.load_text(text, source)
            
            # Generate embeddings for each chunk
            texts = [doc.content for doc in documents]
            embeddings = self.embedding_service.embed_texts(texts)
            
            # Add embeddings to documents
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            # Store in vector database
            document_ids = self.vector_db.add_documents(documents)
            
            logger.info(f"Successfully ingested text: ({len(documents)} chunks)")
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to ingest text: {e}")
            raise

    async def query(self, query_request: QueryRequest) -> RAGResponse:
        start_time = time.time()
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_service.embed_text(query_request.query)
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_db.similarity_search(
                query_embedding=query_embedding,
                k=query_request.max_results,
                similarity_threshold=query_request.similarity_threshold
            )
            
            if not retrieved_docs:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    query=query_request.query,
                    response_time=time.time() - start_time,
                    model_used=self.settings.llm_model
                )
            
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Source: {doc.metadata.source}\n{doc.content}" 
                for doc in retrieved_docs
            ])
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_template(self.system_prompt)
            formatted_prompt = prompt.format(
                context=context,
                question=query_request.query
            )
            
            # Generate response using LLM
            messages = [HumanMessage(content=formatted_prompt)]
            response = await self.llm.ainvoke(messages)
            
            response_time = time.time() - start_time
            
            # Create RAG response
            rag_response = RAGResponse(
                answer=response.content,
                sources=retrieved_docs if query_request.include_metadata else [],
                query=query_request.query,
                response_time=response_time,
                model_used=self.settings.llm_model,
                total_tokens=response.response_metadata.get('token_usage', {}).get('total_tokens')
            )
            
            logger.info(f"Query processed in {response_time:.2f}s, {len(retrieved_docs)} sources used")
            return rag_response
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise

    def health_check(self) -> dict:
        health_status = {
            "rag_engine": True,
            "vector_db": False,
            "embeddings": False,
            "llm": False
        }
        
        try:
            # Check vector database
            health_status["vector_db"] = self.vector_db.health_check()
            
            # Check embeddings service
            health_status["embeddings"] = self.embedding_service.test_connection()
            
            # Check LLM (basic test)
            test_response = self.llm.invoke([HumanMessage(content="test")])
            health_status["llm"] = bool(test_response.content)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return health_status

    def get_stats(self) -> dict:
        collection_info = self.vector_db.get_collection_info()
        embedding_info = self.embedding_service.get_embedding_info()
        
        return {
            "collection": collection_info,
            "embeddings": embedding_info,
            "llm_model": self.settings.llm_model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap
        }