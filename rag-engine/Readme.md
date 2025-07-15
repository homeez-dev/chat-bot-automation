# RAG Engine

A Retrieval-Augmented Generation (RAG) system built with LangChain, designed for scalable document retrieval and question answering.

## Architecture

**Core Components:**
- ChromaDB -> VectorDB
- LangChain -> Orchestration 
- OpenAI Embedder -> Embedding

## RAG Flow Example

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

## Project Structure

```
rag-engine/
├── src/
│   ├── core/              # Core RAG orchestration with Langchain
│   ├── retrieval/         # Document retrieval and chunking logic
│   ├── embeddings/        # Embedding providers
│   │   ├── openai/        # OpenAI text-embedding-3-large
│   │   └── local/         # Local embedding models
│   ├── vectordb/          # Vector database integrations
│   │   ├── chromadb/      # ChromaDB implementation
│   │   └── pinecone/      # Pinecone alternative
│   ├── agents/            # Future agent capabilities
│   │   ├── tools/         # Agent tools and functions
│   │   └── workflows/     # Multi-step agent workflows
│   ├── mcp/               # Model Context Protocol integration
│   │   ├── servers/       # MCP server implementations
│   │   └── clients/       # MCP client integrations
│   └── utils/             # Shared utilities and helpers
├── config/                # Configuration files
├── data/                  # Sample data and documents
├── scripts/               # Setup and utility scripts
├── tests/                 # Test files
└── docs/                  # Documentation
```

## Future Roadmap

### Agent Integration
- Multi-step reasoning workflows
- Tool-calling capabilities
- Dynamic query planning
- Context-aware retrieval strategies

### MCP (Model Context Protocol) Support
- Enhanced context management
- Cross-model communication
- Standardized context sharing
- Protocol-based integrations

## Technology Stack
- **LangChain**: RAG orchestration and document processing
- **ChromaDB**: Primary vector database for similarity search
- **OpenAI Embeddings**: text-embedding-3-large for vector generation
- **Pinecone**: Alternative vector database option