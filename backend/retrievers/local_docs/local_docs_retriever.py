"""
Enhanced Local Documents Retriever for Industry Reporter 2
Searches through local documents with vector similarity and caching
"""
import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from core.config import config
from ..base_retriever import CachedRetriever, SearchResult, RetrieverError
from services.redis_service import RedisService
from services.faiss_service import FAISSService
from utils.document_loader import DocumentLoader


class LocalDocsRetriever(CachedRetriever):
    """
    Enhanced local documents retriever with FAISS integration
    """
    
    def __init__(self, query: str, **kwargs):
        super().__init__(query, **kwargs)
        
        # Local docs configuration
        self.doc_path = kwargs.get('doc_path', config.settings.doc_path)
        self.supported_formats = kwargs.get(
            'supported_formats', 
            config.settings.supported_formats
        )
        self.max_doc_size_mb = kwargs.get(
            'max_doc_size_mb', 
            config.settings.max_doc_size_mb
        )
        
        # Search configuration
        self.use_vector_search = kwargs.get('use_vector_search', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.6)
        self.chunk_size = kwargs.get('chunk_size', 1000)
        self.chunk_overlap = kwargs.get('chunk_overlap', 200)
        
        # Services
        self.redis_service = RedisService()
        self.faiss_service = FAISSService() if self.use_vector_search else None
        self.document_loader = DocumentLoader(self.doc_path)
        
        # Document cache
        self._document_cache = {}
        self._cache_timestamp = None
        
        # Validate document path
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "Local Documents"
    
    @property
    def source_type(self) -> str:
        return "local_files"
    
    async def _search_impl(self, max_results: int) -> List[SearchResult]:
        """Implementation of local document search"""
        try:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📁 Searching local documents in: {self.doc_path}",
                    "doc_path": self.doc_path,
                    "supported_formats": self.supported_formats
                })
            
            # Load documents
            documents = await self._load_documents()
            
            if not documents:
                if self.logger:
                    await self.logger.send_json({
                        "type": "logs",
                        "content": "📭 No documents found in local directory",
                    })
                return []
            
            # Search documents
            if self.use_vector_search and self.faiss_service:
                search_results = await self._vector_search(documents, max_results)
            else:
                search_results = await self._text_search(documents, max_results)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🔍 Found {len(search_results)} relevant local documents",
                    "documents_searched": len(documents),
                    "results_found": len(search_results),
                    "search_method": "vector" if self.use_vector_search else "text"
                })
            
            return search_results
            
        except Exception as e:
            error_msg = f"Local documents search failed: {str(e)}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    async def _load_documents(self) -> List[Dict[str, Any]]:
        """Load and cache documents from local directory"""
        # Check if we need to reload documents
        current_time = time.time()
        
        if (self._cache_timestamp and 
            current_time - self._cache_timestamp < 300 and  # 5 minute cache
            self._document_cache):
            return list(self._document_cache.values())
        
        try:
            # Use document loader to load all supported files
            documents = await self.document_loader.load()
            
            # Update cache
            self._document_cache = {doc.get('file_path', str(i)): doc for i, doc in enumerate(documents)}
            self._cache_timestamp = current_time
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📄 Loaded {len(documents)} documents from local storage",
                    "document_count": len(documents),
                    "cache_updated": True
                })
            
            return documents
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Document loading failed: {e}")
            return []
    
    async def _vector_search(
        self, 
        documents: List[Dict[str, Any]], 
        max_results: int
    ) -> List[SearchResult]:
        """Perform vector-based similarity search"""
        try:
            # Ensure documents are indexed in FAISS
            await self._ensure_documents_indexed(documents)
            
            # Perform similarity search
            similar_docs = await self.faiss_service.similarity_search(
                query=self.query,
                k=max_results * 2,  # Get more results to filter
                filter_dict={"source": "local_documents"}
            )
            
            # Convert to SearchResult objects
            search_results = []
            for doc in similar_docs:
                try:
                    # Extract content and metadata
                    content = doc.get("content", "")
                    file_path = doc.get("file_path", "")
                    file_name = os.path.basename(file_path) if file_path else "unknown"
                    
                    # Calculate relevance score
                    relevance_score = doc.get("score", 0.0)
                    
                    # Apply similarity threshold
                    if relevance_score < self.similarity_threshold:
                        continue
                    
                    # Create search result
                    search_result = SearchResult(
                        title=file_name,
                        content=self._sanitize_content(content),
                        url=f"file://{file_path}",
                        source="local_document",
                        relevance_score=relevance_score,
                        metadata={
                            "file_path": file_path,
                            "file_name": file_name,
                            "file_size": doc.get("file_size", 0),
                            "file_type": doc.get("file_type", ""),
                            "last_modified": doc.get("last_modified", ""),
                            "search_method": "vector",
                            "faiss_score": relevance_score
                        }
                    )
                    
                    search_results.append(search_result)
                    
                    if len(search_results) >= max_results:
                        break
                        
                except Exception as e:
                    if self.logger:
                        await self.logger.log_error(f"Error processing vector result: {e}")
                    continue
            
            return search_results
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Vector search failed, falling back to text search: {e}")
            return await self._text_search(documents, max_results)
    
    async def _text_search(
        self, 
        documents: List[Dict[str, Any]], 
        max_results: int
    ) -> List[SearchResult]:
        """Perform text-based search as fallback"""
        query_words = set(self.query.lower().split())
        scored_docs = []
        
        for doc in documents:
            try:
                content = doc.get("content", "")
                file_path = doc.get("file_path", "")
                file_name = os.path.basename(file_path) if file_path else "unknown"
                
                if not content:
                    continue
                
                # Calculate text-based relevance score
                content_words = set(content.lower().split())
                title_words = set(file_name.lower().split())
                
                # Score based on word matches
                content_matches = len(query_words.intersection(content_words))
                title_matches = len(query_words.intersection(title_words))
                
                # Weight title matches more heavily
                score = (title_matches * 3 + content_matches) / (len(query_words) * 4)
                
                if score > 0:
                    scored_docs.append((score, doc))
                    
            except Exception as e:
                if self.logger:
                    await self.logger.log_error(f"Error processing document in text search: {e}")
                continue
        
        # Sort by score and create SearchResult objects
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        search_results = []
        for score, doc in scored_docs[:max_results]:
            try:
                file_path = doc.get("file_path", "")
                file_name = os.path.basename(file_path) if file_path else "unknown"
                
                search_result = SearchResult(
                    title=file_name,
                    content=self._sanitize_content(doc.get("content", "")),
                    url=f"file://{file_path}",
                    source="local_document",
                    relevance_score=score,
                    metadata={
                        "file_path": file_path,
                        "file_name": file_name,
                        "file_size": doc.get("file_size", 0),
                        "file_type": doc.get("file_type", ""),
                        "last_modified": doc.get("last_modified", ""),
                        "search_method": "text",
                        "text_score": score
                    }
                )
                
                search_results.append(search_result)
                
            except Exception as e:
                if self.logger:
                    await self.logger.log_error(f"Error creating search result: {e}")
                continue
        
        return search_results
    
    async def _ensure_documents_indexed(self, documents: List[Dict[str, Any]]):
        """Ensure all documents are indexed in FAISS"""
        if not self.faiss_service:
            return
        
        try:
            # Check which documents need to be indexed
            documents_to_index = []
            
            for doc in documents:
                file_path = doc.get("file_path", "")
                last_modified = doc.get("last_modified", "")
                
                # Create a unique document ID
                doc_id = hashlib.md5(f"{file_path}:{last_modified}".encode()).hexdigest()
                
                # Check if document is already indexed (simplified check)
                doc["id"] = doc_id
                doc["source"] = "local_documents"
                documents_to_index.append(doc)
            
            if documents_to_index:
                # Add documents to FAISS index
                await self.faiss_service.add_documents(documents_to_index)
                
                if self.logger:
                    await self.logger.send_json({
                        "type": "logs",
                        "content": f"📇 Indexed {len(documents_to_index)} documents in FAISS",
                        "indexed_count": len(documents_to_index)
                    })
                    
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Document indexing failed: {e}")
    
    # Caching implementation
    async def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached results from Redis"""
        try:
            cached_data = await self.redis_service.get_cached_context(cache_key)
            
            if cached_data and isinstance(cached_data, list):
                search_results = []
                for item in cached_data:
                    if isinstance(item, dict):
                        search_result = SearchResult(
                            title=item.get("title", ""),
                            content=item.get("content", ""),
                            url=item.get("url", ""),
                            source=item.get("source", "local_document"),
                            relevance_score=item.get("relevance_score", 0.0),
                            timestamp=item.get("timestamp", ""),
                            metadata=item.get("metadata", {})
                        )
                        search_results.append(search_result)
                
                return search_results
            
            return None
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Cache retrieval error: {e}")
            return None
    
    async def _save_to_cache(self, cache_key: str, results: List[SearchResult]):
        """Save results to Redis cache"""
        try:
            cache_data = [result.to_dict() for result in results]
            
            await self.redis_service.cache_context(
                cache_key, 
                cache_data, 
                ttl=self._cache_ttl
            )
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"💾 Cached {len(results)} local document results",
                    "cache_ttl": self._cache_ttl
                })
                
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Cache save error: {e}")
    
    # Enhanced methods
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about local documents"""
        try:
            documents = await self._load_documents()
            
            # Count by file type
            file_types = {}
            total_size = 0
            
            for doc in documents:
                file_type = doc.get("file_type", "unknown")
                file_size = doc.get("file_size", 0)
                
                file_types[file_type] = file_types.get(file_type, 0) + 1
                total_size += file_size
            
            return {
                "total_documents": len(documents),
                "file_types": file_types,
                "total_size_mb": total_size / (1024 * 1024),
                "doc_path": self.doc_path,
                "supported_formats": self.supported_formats,
                "vector_search_enabled": self.use_vector_search,
                "cache_status": {
                    "cached_documents": len(self._document_cache),
                    "cache_timestamp": self._cache_timestamp
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def refresh_document_cache(self):
        """Force refresh of document cache"""
        self._document_cache.clear()
        self._cache_timestamp = None
        
        documents = await self._load_documents()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔄 Document cache refreshed: {len(documents)} documents",
                "document_count": len(documents)
            })
    
    async def search_specific_file_types(
        self, 
        file_types: List[str], 
        max_results: int = None
    ) -> List[SearchResult]:
        """Search only specific file types"""
        if max_results is None:
            max_results = self._max_results
        
        # Load all documents
        all_documents = await self._load_documents()
        
        # Filter by file type
        filtered_documents = []
        for doc in all_documents:
            doc_file_type = doc.get("file_type", "").lower()
            if any(ft.lower() in doc_file_type for ft in file_types):
                filtered_documents.append(doc)
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🎯 Searching {len(filtered_documents)} documents of types: {file_types}",
                "filtered_count": len(filtered_documents),
                "file_types": file_types
            })
        
        # Search filtered documents
        if self.use_vector_search and self.faiss_service:
            return await self._vector_search(filtered_documents, max_results)
        else:
            return await self._text_search(filtered_documents, max_results)