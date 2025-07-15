import uuid
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)

from ..core.models import Document, DocumentMetadata
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Document]:
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path_obj.suffix.lower()
        
        try:
            # Choose appropriate loader based on file type
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Load the document
            langchain_docs = loader.load()
            
            # Convert to our Document model
            documents = []
            for doc in langchain_docs:
                # Split the document into chunks
                chunks = self.text_splitter.split_text(doc.page_content)
                
                for i, chunk in enumerate(chunks):
                    document = Document(
                        id=str(uuid.uuid4()),
                        content=chunk,
                        metadata=DocumentMetadata(
                            source=file_path,
                            page=doc.metadata.get('page'),
                            chunk_index=i,
                            total_chunks=len(chunks),
                            created_at=datetime.now(),
                            file_type=file_extension,
                            size=len(chunk)
                        )
                    )
                    documents.append(document)
            
            logger.info(f"Processed {file_path}: {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise

    def load_text(self, text: str, source: str = "direct_input") -> List[Document]:
        try:
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                document = Document(
                    id=str(uuid.uuid4()),
                    content=chunk,
                    metadata=DocumentMetadata(
                        source=source,
                        chunk_index=i,
                        total_chunks=len(chunks),
                        created_at=datetime.now(),
                        file_type="text",
                        size=len(chunk)
                    )
                )
                documents.append(document)
            
            logger.info(f"Processed text input: {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process text: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        # Basic text preprocessing
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        return text

    def validate_document(self, document: Document) -> bool:
        if not document.content or len(document.content.strip()) == 0:
            return False
        
        if len(document.content) > 10000:  # Arbitrary max chunk size
            logger.warning(f"Document chunk is very large: {len(document.content)} characters")
        
        return True

    def get_supported_formats(self) -> List[str]:
        return ['.txt', '.pdf', '.docx', '.doc']

    def estimate_tokens(self, text: str) -> int:
        # Rough estimation: 1 token ≈ 4 characters for English
        return len(text) // 4