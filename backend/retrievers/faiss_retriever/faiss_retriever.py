"""
FAISS Vector Retriever for Industry Reporter 2
Pure vector similarity search using FAISS index
"""
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from core.config import config
from ..base_retriever import CachedRetriever, SearchResult, RetrieverError
from services.redis_service import RedisService
from services.faiss_service import FAISSService


class FAISSRetriever(CachedRetriever):
    """
    Pure FAISS vector similarity retriever
    Searches through pre-indexed vector embeddings
    """
    
    def __init__(self, query: str, **kwargs):
        super().__init__(query, **kwargs)
        
        # FAISS configuration
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.6)
        self.include_scores = kwargs.get('include_scores', True)
        self.filter_dict = kwargs.get('filter_dict', None)
        self.search_type = kwargs.get('search_type', 'similarity')  # similarity, mmr, etc.
        
        # Advanced search parameters
        self.fetch_k = kwargs.get('fetch_k', None)  # Number of docs to fetch before filtering
        self.lambda_mult = kwargs.get('lambda_mult', 0.5)  # MMR diversity parameter
        
        # Services
        self.redis_service = RedisService()
        self.faiss_service = FAISSService()
        
        # Performance tracking
        self._vector_search_stats = {}
    
    @property
    def name(self) -> str:
        return "FAISS Vector Search"
    
    @property
    def source_type(self) -> str:
        return "vector_index"
    
    async def _search_impl(self, max_results: int) -> List[SearchResult]:
        """Implementation of FAISS vector search"""
        try:
            # Ensure FAISS service is initialized
            await self._ensure_faiss_initialized()
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🔍 Performing FAISS vector search",
                    "search_type": self.search_type,
                    "similarity_threshold": self.similarity_threshold,
                    "filter_dict": self.filter_dict
                })
            
            # Perform vector search based on search type
            if self.search_type == 'mmr':
                vector_results = await self._mmr_search(max_results)
            else:
                vector_results = await self._similarity_search(max_results)
            
            # Convert to SearchResult objects
            search_results = await self._process_vector_results(vector_results)
            
            # Track performance
            self._vector_search_stats = {
                "results_found": len(search_results),
                "search_type": self.search_type,
                "similarity_threshold": self.similarity_threshold,
                "index_size": await self._get_index_size()
            }
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🎯 FAISS found {len(search_results)} vector matches",
                    "results_count": len(search_results),
                    "avg_score": sum(r.relevance_score for r in search_results) / len(search_results) if search_results else 0
                })
            
            return search_results
            
        except Exception as e:
            error_msg = f"FAISS vector search failed: {str(e)}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    async def _ensure_faiss_initialized(self):
        """Ensure FAISS service is properly initialized"""
        try:
            if not await self.faiss_service.is_initialized():
                await self.faiss_service.initialize()
                
                if self.logger:
                    await self.logger.send_json({
                        "type": "logs",
                        "content": "🔧 FAISS service initialized",
                    })
        except Exception as e:
            raise RetrieverError(f"FAISS initialization failed: {e}") from e
    
    async def _similarity_search(self, max_results: int) -> List[Dict[str, Any]]:
        """Perform standard similarity search"""
        return await self.faiss_service.similarity_search(
            query=self.query,
            k=max_results,
            filter_dict=self.filter_dict,
            fetch_k=self.fetch_k
        )
    
    async def _mmr_search(self, max_results: int) -> List[Dict[str, Any]]:
        """Perform Maximal Marginal Relevance search for diversity"""
        try:
            return await self.faiss_service.max_marginal_relevance_search(
                query=self.query,
                k=max_results,
                fetch_k=self.fetch_k or max_results * 3,
                lambda_mult=self.lambda_mult,
                filter_dict=self.filter_dict
            )
        except AttributeError:
            # Fallback to similarity search if MMR not implemented
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": "⚠️ MMR not available, falling back to similarity search",
                })
            return await self._similarity_search(max_results)
    
    async def _process_vector_results(self, vector_results: List[Dict[str, Any]]) -> List[SearchResult]:
        """Process FAISS results into SearchResult objects"""
        search_results = []
        
        for result in vector_results:
            try:
                # Extract content and metadata
                content = result.get("content", "")
                if not content:
                    continue
                
                # Extract metadata
                metadata = result.get("metadata", {})
                title = metadata.get("title", "") or result.get("title", "")
                url = metadata.get("url", "") or result.get("url", "")
                source = metadata.get("source", "") or result.get("source", "faiss_index")
                
                # Get similarity score
                similarity_score = result.get("score", 0.0)
                
                # Apply similarity threshold
                if similarity_score < self.similarity_threshold:
                    continue
                
                # Generate title if not available
                if not title:
                    title = self._generate_title_from_content(content)
                
                # Generate URL if not available
                if not url:
                    doc_id = metadata.get("doc_id", "") or result.get("id", "")
                    url = f"faiss://doc/{doc_id}" if doc_id else "faiss://unknown"
                
                # Create search result
                search_result = SearchResult(
                    title=title,
                    content=self._sanitize_content(content),
                    url=url,
                    source=f"faiss_{source}" if source != "faiss_index" else "faiss_index",
                    relevance_score=similarity_score,
                    metadata={
                        **metadata,
                        "search_method": "vector_similarity",
                        "faiss_score": similarity_score,
                        "vector_search_type": self.search_type,
                        "similarity_threshold": self.similarity_threshold,
                        "original_source": source
                    }
                )
                
                search_results.append(search_result)
                
            except Exception as e:
                if self.logger:
                    await self.logger.log_error(f"Error processing FAISS result: {e}")
                continue
        
        return search_results
    
    def _generate_title_from_content(self, content: str) -> str:
        """Generate a title from content if not available"""
        # Take first sentence or first 50 characters
        sentences = content.split('. ')
        if sentences:
            title = sentences[0]
            if len(title) > 100:
                title = title[:97] + "..."
            return title
        
        # Fallback to first 50 characters
        return content[:47] + "..." if len(content) > 50 else content
    
    async def _get_index_size(self) -> int:
        """Get the current size of the FAISS index"""
        try:
            index_info = await self.faiss_service.get_index_info()
            return index_info.get("total_vectors", 0)
        except:
            return 0
    
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
                            source=item.get("source", "faiss_index"),
                            relevance_score=item.get("relevance_score", 0.0),
                            timestamp=item.get("timestamp", ""),
                            metadata=item.get("metadata", {})
                        )
                        search_results.append(search_result)
                
                return search_results
            
            return None
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"FAISS cache retrieval error: {e}")
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
                    "content": f"💾 Cached {len(results)} FAISS vector results",
                    "cache_ttl": self._cache_ttl
                })
                
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"FAISS cache save error: {e}")
    
    # Enhanced search methods
    async def search_with_embedding(
        self, 
        query_embedding: np.ndarray, 
        max_results: int = None
    ) -> List[SearchResult]:
        """
        Search using pre-computed embedding vector
        """
        if max_results is None:
            max_results = self._max_results
        
        try:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": "🔢 Searching with pre-computed embedding",
                    "embedding_dim": len(query_embedding)
                })
            
            # Perform search with embedding
            vector_results = await self.faiss_service.similarity_search_by_vector(
                embedding=query_embedding,
                k=max_results,
                filter_dict=self.filter_dict
            )
            
            # Process results
            search_results = await self._process_vector_results(vector_results)
            
            return search_results
            
        except Exception as e:
            error_msg = f"Embedding search failed: {str(e)}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    async def search_similar_documents(
        self, 
        document_text: str, 
        max_results: int = None,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document
        """
        if max_results is None:
            max_results = self._max_results
        
        try:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": "📄 Finding similar documents",
                    "doc_length": len(document_text)
                })
            
            # Get embedding for the document
            doc_embedding = await self.faiss_service.get_embedding(document_text)
            
            # Search for similar documents
            results = await self.search_with_embedding(doc_embedding, max_results * 2)
            
            # Filter out the document itself if requested
            if exclude_self:
                filtered_results = []
                for result in results:
                    # Simple content similarity check to exclude self
                    if not self._is_same_document(document_text, result.content):
                        filtered_results.append(result)
                    
                    if len(filtered_results) >= max_results:
                        break
                
                results = filtered_results
            
            return results[:max_results]
            
        except Exception as e:
            error_msg = f"Similar document search failed: {str(e)}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    def _is_same_document(self, doc1: str, doc2: str, threshold: float = 0.9) -> bool:
        """Check if two documents are the same based on content similarity"""
        # Simple similarity check
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    async def get_embedding_for_query(self) -> np.ndarray:
        """Get the embedding vector for the current query"""
        try:
            return await self.faiss_service.get_embedding(self.query)
        except Exception as e:
            raise RetrieverError(f"Failed to get query embedding: {e}") from e
    
    async def search_by_filters(
        self, 
        filter_dict: Dict[str, Any], 
        max_results: int = None,
        combine_with_query: bool = True
    ) -> List[SearchResult]:
        """
        Search using metadata filters with optional query combination
        """
        if max_results is None:
            max_results = self._max_results
        
        try:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🎯 Searching with filters: {filter_dict}",
                    "filters": filter_dict,
                    "combine_with_query": combine_with_query
                })
            
            if combine_with_query:
                # Combine filter with query search
                vector_results = await self.faiss_service.similarity_search(
                    query=self.query,
                    k=max_results,
                    filter_dict=filter_dict
                )
            else:
                # Pure filter search (if supported by FAISS service)
                vector_results = await self.faiss_service.filter_search(
                    filter_dict=filter_dict,
                    k=max_results
                )
            
            # Process results
            search_results = await self._process_vector_results(vector_results)
            
            return search_results
            
        except Exception as e:
            error_msg = f"Filter search failed: {str(e)}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    # Configuration methods
    def configure_similarity_threshold(self, threshold: float):
        """Configure similarity threshold for results"""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
        else:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
    
    def configure_search_type(self, search_type: str):
        """Configure search type (similarity, mmr, etc.)"""
        valid_types = ['similarity', 'mmr']
        if search_type in valid_types:
            self.search_type = search_type
        else:
            raise ValueError(f"Search type must be one of: {valid_types}")
    
    def configure_mmr_parameters(self, lambda_mult: float, fetch_k: int = None):
        """Configure MMR search parameters"""
        if 0.0 <= lambda_mult <= 1.0:
            self.lambda_mult = lambda_mult
        else:
            raise ValueError("Lambda multiplier must be between 0.0 and 1.0")
        
        if fetch_k is not None and fetch_k > 0:
            self.fetch_k = fetch_k
    
    async def get_faiss_stats(self) -> Dict[str, Any]:
        """Get FAISS-specific statistics"""
        base_stats = await self.get_search_stats()
        
        try:
            index_info = await self.faiss_service.get_index_info()
            
            faiss_stats = {
                "faiss_config": {
                    "similarity_threshold": self.similarity_threshold,
                    "search_type": self.search_type,
                    "lambda_mult": self.lambda_mult,
                    "fetch_k": self.fetch_k
                },
                "index_info": index_info,
                "vector_search_stats": self._vector_search_stats,
                "cache_enabled": self._cache_enabled
            }
            
            base_stats.update(faiss_stats)
            return base_stats
            
        except Exception as e:
            base_stats["faiss_error"] = str(e)
            return base_stats