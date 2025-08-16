"""
Enhanced Retrievers Module for Industry Reporter 2
Based on GPT-Researcher's retrievers with modern enhancements
"""

from .base_retriever import BaseRetriever
from .tavily.tavily_retriever import TavilyRetriever
from .local_docs.local_docs_retriever import LocalDocsRetriever
from .faiss_retriever.faiss_retriever import FAISSRetriever
from .redis_cache.redis_cache_retriever import RedisCacheRetriever
from .utils import get_retrievers, RetrieverFactory

# Export the main retriever classes
__all__ = [
    "BaseRetriever",
    "TavilyRetriever",
    "LocalDocsRetriever", 
    "FAISSRetriever",
    "RedisCacheRetriever",
    "get_retrievers",
    "RetrieverFactory"
]

# Version info
__version__ = "2.0.0"
__description__ = "Enhanced multi-retriever system with caching and vector search"