ChromaDB -> VectorDB
LangChain -> Orchestration 
OpenAPI Embedder -> embedding

User asks: "What's our company's vacation policy?"
↓
Call OpenAI Embeddings API (text-embedding-3-large) to convert question to 3072-dimensional vector
↓
Search ChromaDB/Pinecone using the OpenAI-generated embedding vector for cosine similarity
↓
Retrieve top 5 relevant chunks from HR handbook with highest similarity scores
↓
Send to OpenAI GPT-4: "Based on this context: [retrieved HR policy text], answer: What's our company's vacation policy?"
↓
OpenAI GPT-4 responds with accurate, grounded answer citing the specific vacation policy details