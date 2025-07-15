import uuid
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
import logging

from ...core.models import Document, RetrievedDocument, DocumentMetadata
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class ChromaDBClient:
    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._collection = None
        self._setup_client()

    def _setup_client(self):
        try:
            # Configure ChromaDB client for HTTP API
            self._client = chromadb.HttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port
            )
            
            # Get or create collection with cosine distance
            self._collection = self._client.get_or_create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"description": "RAG document embeddings", "hnsw:space": "cosine"}
            )
            
            logger.info(f"Connected to ChromaDB collection: {self.settings.chroma_collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB client: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        document_ids = []
        embeddings = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = doc.id or str(uuid.uuid4())
            document_ids.append(doc_id)
            embeddings.append(doc.embedding)
            texts.append(doc.content)
            metadatas.append(doc.metadata.model_dump())
            
        try:
            self._collection.add(
                ids=document_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise

    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[RetrievedDocument]:
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_docs = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity score (now using cosine distance)
                    similarity_score = 1 - distance
                    
                    logger.info(f"Document {i}: similarity_score={similarity_score:.3f}, distance={distance:.3f}, threshold={similarity_threshold}")
                    
                    if similarity_score >= similarity_threshold:
                        retrieved_doc = RetrievedDocument(
                            content=doc,
                            metadata=DocumentMetadata(**metadata),
                            similarity_score=similarity_score
                        )
                        retrieved_docs.append(retrieved_doc)
                    else:
                        logger.info(f"Document {i} below threshold: {similarity_score:.3f} < {similarity_threshold}")
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents above threshold {similarity_threshold}")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            raise

    def delete_documents(self, document_ids: List[str]) -> bool:
        try:
            self._collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            count = self._collection.count()
            return {
                "name": self.settings.chroma_collection_name,
                "count": count,
                "host": self.settings.chroma_host,
                "port": self.settings.chroma_port
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def health_check(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False

    def clear_collection(self) -> bool:
        try:
            # Delete the collection and recreate it
            self._client.delete_collection(name=self.settings.chroma_collection_name)
            self._collection = self._client.create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"description": "RAG document embeddings"}
            )
            logger.info(f"Cleared collection: {self.settings.chroma_collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False