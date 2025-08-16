"""
Enhanced Context Manager for Industry Reporter 2
Based on GPT-Researcher's context_manager.py with FAISS and Redis integration
"""
import asyncio
import hashlib
import json
import time
from typing import List, Dict, Optional, Set, Any, Tuple
from datetime import datetime, timedelta

from core.config import config
from core.logging import CustomLogsHandler
from services.faiss_service import FAISSService
from services.redis_service import RedisService


class ContextCompressor:
    """Enhanced context compressor with FAISS similarity search"""
    
    def __init__(self, documents: List[Dict], embeddings_service, **kwargs):
        self.documents = documents
        self.embeddings_service = embeddings_service
        self.kwargs = kwargs
    
    async def async_get_context(self, query: str, max_results: int = 10, cost_callback=None) -> str:
        """Get relevant context using FAISS similarity search"""
        if not self.documents:
            return ""
        
        try:
            # Extract text content from documents
            texts = []
            for doc in self.documents:
                if isinstance(doc, dict):
                    text = doc.get('content', '') or doc.get('body', '') or str(doc)
                else:
                    text = str(doc)
                if text.strip():
                    texts.append(text)
            
            if not texts:
                return ""
            
            # Use FAISS for similarity search
            similar_indices = await self.embeddings_service.find_similar_documents(
                query, texts, top_k=max_results
            )
            
            # Get the most relevant documents
            relevant_texts = [texts[i] for i in similar_indices if i < len(texts)]
            
            # Join and return
            return "\n\n".join(relevant_texts[:max_results])
            
        except Exception as e:
            # Fallback to simple text matching
            return self._fallback_text_search(query, max_results)
    
    def _fallback_text_search(self, query: str, max_results: int) -> str:
        """Fallback method using simple text matching"""
        query_words = query.lower().split()
        scored_docs = []
        
        for doc in self.documents:
            if isinstance(doc, dict):
                text = doc.get('content', '') or doc.get('body', '') or str(doc)
            else:
                text = str(doc)
            
            text_lower = text.lower()
            score = sum(1 for word in query_words if word in text_lower)
            if score > 0:
                scored_docs.append((score, text))
        
        # Sort by relevance score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results
        relevant_texts = [text for _, text in scored_docs[:max_results]]
        return "\n\n".join(relevant_texts)


class WrittenContentCompressor:
    """Compressor for previously written content with similarity threshold"""
    
    def __init__(self, documents: List[Dict], embeddings_service, similarity_threshold: float = 0.5, **kwargs):
        self.documents = documents
        self.embeddings_service = embeddings_service
        self.similarity_threshold = similarity_threshold
        self.kwargs = kwargs
    
    async def async_get_context(self, query: str, max_results: int = 10, cost_callback=None) -> List[str]:
        """Get similar written content using embeddings"""
        if not self.documents:
            return []
        
        try:
            # Extract content from written documents
            texts = []
            for doc in self.documents:
                if isinstance(doc, dict):
                    content = doc.get('content', '') or doc.get('text', '') or str(doc)
                else:
                    content = str(doc)
                
                if content.strip():
                    texts.append(content)
            
            if not texts:
                return []
            
            # Use FAISS for similarity search with threshold
            similar_indices = await self.embeddings_service.find_similar_documents(
                query, texts, top_k=max_results, similarity_threshold=self.similarity_threshold
            )
            
            # Return relevant content
            return [texts[i] for i in similar_indices if i < len(texts)]
            
        except Exception as e:
            # Fallback to simple matching
            return self._fallback_content_search(query, max_results)
    
    def _fallback_content_search(self, query: str, max_results: int) -> List[str]:
        """Fallback content search"""
        query_words = set(query.lower().split())
        relevant_content = []
        
        for doc in self.documents:
            if isinstance(doc, dict):
                content = doc.get('content', '') or doc.get('text', '') or str(doc)
            else:
                content = str(doc)
            
            content_words = set(content.lower().split())
            overlap = len(query_words.intersection(content_words))
            
            if overlap > 0:
                relevant_content.append((overlap, content))
        
        # Sort by overlap score and return top results
        relevant_content.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in relevant_content[:max_results]]


class ContextManager:
    """Enhanced context manager with FAISS and Redis integration"""
    
    def __init__(self, researcher):
        self.researcher = researcher
        self.faiss_service = FAISSService()
        self.redis_service = RedisService()
        self.logger = researcher.logger if hasattr(researcher, 'logger') else None
        
        # Cache for context operations
        self._context_cache = {}
        self._cache_ttl = config.settings.cache_ttl_seconds
    
    async def get_similar_content_by_query(self, query: str, pages: List[Dict], use_cache: bool = True) -> str:
        """
        Enhanced version with FAISS and Redis caching
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key("content_query", query, pages)
        
        # Check Redis cache first
        if use_cache:
            cached_result = await self.redis_service.get_cached_context(cache_key)
            if cached_result:
                if self.logger:
                    await self.logger.log_performance_metric(
                        "context_retrieval_time", 
                        time.time() - start_time, 
                        "seconds"
                    )
                    await self.logger.send_json({
                        "type": "logs",
                        "content": f"📚 Retrieved cached content for query: {query}",
                        "cache_hit": True
                    })
                return cached_result
        
        # Log the operation
        if self.logger:
            await self.logger.send_json({
                "type": "logs", 
                "content": f"📚 Getting relevant content based on query: {query}...",
                "cache_hit": False
            })
        
        # Use enhanced context compressor with FAISS
        context_compressor = ContextCompressor(
            documents=pages,
            embeddings_service=self.faiss_service,
            **getattr(self.researcher, 'kwargs', {})
        )
        
        result = await context_compressor.async_get_context(
            query=query, 
            max_results=10, 
            cost_callback=getattr(self.researcher, 'add_costs', None)
        )
        
        # Cache the result
        if use_cache and result:
            await self.redis_service.cache_context(cache_key, result, ttl=self._cache_ttl)
        
        # Log performance metrics
        if self.logger:
            processing_time = time.time() - start_time
            await self.logger.log_performance_metric(
                "context_retrieval_time", 
                processing_time, 
                "seconds"
            )
            await self.logger.send_json({
                "type": "logs",
                "content": f"✅ Context retrieved in {processing_time:.2f}s",
                "processing_time": processing_time,
                "result_length": len(result)
            })
        
        return result
    
    async def get_similar_content_by_query_with_vectorstore(self, query: str, filter_dict: Optional[Dict] = None) -> str:
        """
        Enhanced vectorstore search with FAISS integration
        """
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔍 Getting relevant content from vector store for query: {query}...",
            })
        
        try:
            # Use FAISS service for vector search
            results = await self.faiss_service.similarity_search(
                query, 
                k=8, 
                filter_dict=filter_dict
            )
            
            # Extract content from results
            content_pieces = []
            for result in results:
                if isinstance(result, dict):
                    content = result.get('content', '') or result.get('text', '')
                else:
                    content = str(result)
                
                if content.strip():
                    content_pieces.append(content)
            
            result = "\n\n".join(content_pieces)
            
            # Log performance
            if self.logger:
                processing_time = time.time() - start_time
                await self.logger.log_performance_metric(
                    "vectorstore_search_time", 
                    processing_time, 
                    "seconds"
                )
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"✅ Vector search completed in {processing_time:.2f}s",
                    "results_count": len(results)
                })
            
            return result
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Vector store search failed: {str(e)}")
            return ""
    
    async def get_similar_written_contents_by_draft_section_titles(
        self,
        current_subtopic: str,
        draft_section_titles: List[str],
        written_contents: List[Dict],
        max_results: int = 10
    ) -> List[str]:
        """
        Enhanced written content similarity search
        """
        start_time = time.time()
        
        # Combine all queries
        all_queries = [current_subtopic] + draft_section_titles
        
        async def process_query(query: str) -> Set[str]:
            """Process a single query and return relevant content"""
            cache_key = self._generate_cache_key("written_content", query, written_contents)
            
            # Check cache
            cached_result = await self.redis_service.get_cached_context(cache_key)
            if cached_result:
                return set(cached_result) if isinstance(cached_result, list) else {cached_result}
            
            # Search for similar content
            compressor = WrittenContentCompressor(
                documents=written_contents,
                embeddings_service=self.faiss_service,
                similarity_threshold=0.5,
                **getattr(self.researcher, 'kwargs', {})
            )
            
            results = await compressor.async_get_context(
                query=query,
                max_results=max_results,
                cost_callback=getattr(self.researcher, 'add_costs', None)
            )
            
            # Cache results
            if results:
                await self.redis_service.cache_context(cache_key, results, ttl=self._cache_ttl)
            
            return set(results)
        
        # Process all queries concurrently
        results = await asyncio.gather(*[process_query(query) for query in all_queries])
        
        # Combine and deduplicate results
        relevant_contents = set().union(*results)
        relevant_contents_list = list(relevant_contents)[:max_results]
        
        # Log performance
        if self.logger:
            processing_time = time.time() - start_time
            await self.logger.log_performance_metric(
                "written_content_search_time", 
                processing_time, 
                "seconds"
            )
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔎 Found {len(relevant_contents_list)} relevant written contents",
                "processing_time": processing_time,
                "queries_processed": len(all_queries)
            })
        
        return relevant_contents_list
    
    def _generate_cache_key(self, operation: str, query: str, data: Any) -> str:
        """Generate a cache key for the operation"""
        # Create a hash of the query and data structure
        data_hash = hashlib.md5(str(data).encode()).hexdigest()[:8]
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        return f"context:{operation}:{query_hash}:{data_hash}"
    
    async def clear_cache(self):
        """Clear the context cache"""
        await self.redis_service.clear_cache_pattern("context:*")
        self._context_cache.clear()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": "🧹 Context cache cleared"
            })
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return await self.redis_service.get_cache_stats("context:*")
    
    # Compatibility methods for original GPT-Researcher interface
    async def __get_similar_written_contents_by_query(
        self,
        query: str,
        written_contents: List[Dict],
        similarity_threshold: float = 0.5,
        max_results: int = 10
    ) -> List[str]:
        """Private method for backward compatibility"""
        compressor = WrittenContentCompressor(
            documents=written_contents,
            embeddings_service=self.faiss_service,
            similarity_threshold=similarity_threshold,
            **getattr(self.researcher, 'kwargs', {})
        )
        
        return await compressor.async_get_context(
            query=query,
            max_results=max_results,
            cost_callback=getattr(self.researcher, 'add_costs', None)
        )