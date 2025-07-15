#!/usr/bin/env python3
"""
Development server runner for RAG Engine
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import get_settings


def main():
    # Load settings
    settings = get_settings()
    
    print(f"Starting RAG Engine server on {settings.api_host}:{settings.api_port}")
    print(f"Debug mode: {settings.debug}")
    print(f"ChromaDB: {settings.chroma_host}:{settings.chroma_port}")
    print(f"Collection: {settings.chroma_collection_name}")
    print("-" * 50)
    
    # Run the server
    os.chdir(project_root)
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()