"""
Utilities Module for Industry Reporter 2
Document loading and other utility functions
"""

from .document_loader import DocumentLoader, load_documents, load_single_document

__all__ = [
    "DocumentLoader",
    "load_documents", 
    "load_single_document"
]