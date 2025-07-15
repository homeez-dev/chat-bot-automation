import asyncio
from typing import List, Union
import openai
import logging
from openai import OpenAI

from ...core.config import get_settings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService:
    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.embedding_model
        self.dimensions = self.settings.embedding_dimensions

    def embed_text(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        if not texts:
            return []
        
        embeddings = []
        
        try:
            # Process in batches to avoid API limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
            
            logger.info(f"Generated {len(embeddings)} embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    async def embed_text_async(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

    async def embed_texts_async(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts, batch_size)

    def test_connection(self) -> bool:
        try:
            # Test with a simple embedding
            test_embedding = self.embed_text("test connection")
            return len(test_embedding) == self.dimensions
            
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False

    def get_embedding_info(self) -> dict:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "max_input_tokens": 8191,  # For text-embedding-3-large
            "pricing_per_1k_tokens": 0.00013  # As of 2024
        }