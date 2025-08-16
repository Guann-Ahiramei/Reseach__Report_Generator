"""
Enhanced Tavily Retriever for Industry Reporter 2
Based on GPT-Researcher's tavily_search.py with modern enhancements
"""
import os
import json
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional, Literal, Sequence
from datetime import datetime

from core.config import config
from ..base_retriever import CachedRetriever, SearchResult, RetrieverError
from services.redis_service import RedisService


class TavilyRetriever(CachedRetriever):
    """
    Enhanced Tavily API Retriever with caching and modern async support
    """
    
    def __init__(self, query: str, **kwargs):
        super().__init__(query, **kwargs)
        
        # Tavily-specific configuration
        self.topic = kwargs.get('topic', 'general')
        self.search_depth = kwargs.get('search_depth', 'basic')
        self.include_images = kwargs.get('include_images', False)
        self.include_answer = kwargs.get('include_answer', False)
        self.include_raw_content = kwargs.get('include_raw_content', True)
        self.days = kwargs.get('days', 7)  # Recent results within N days
        
        # API configuration
        self.base_url = "https://api.tavily.com/search"
        self.api_key = self._get_api_key()
        
        # Redis service for caching
        self.redis_service = RedisService()
        
        # Validate API key
        if not self.api_key:
            raise RetrieverError("Tavily API key not found. Please set TAVILY_API_KEY environment variable.")
    
    @property
    def name(self) -> str:
        return "Tavily Web Search"
    
    @property
    def source_type(self) -> str:
        return "web"
    
    def _get_api_key(self) -> str:
        """Get Tavily API key from headers or environment"""
        # Try headers first (for custom configurations)
        api_key = self.headers.get("tavily_api_key")
        
        if not api_key:
            # Try environment variable
            api_key = os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            # Try config
            api_key = getattr(config, 'tavily_api_key', '')
        
        return api_key or ""
    
    async def _search_impl(self, max_results: int) -> List[SearchResult]:
        """Implementation of Tavily search with async support"""
        if not self.api_key:
            if self.logger:
                await self.logger.log_error("Tavily API key not available")
            return []
        
        try:
            # Prepare search parameters
            search_params = self._prepare_search_params(max_results)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🌐 Searching Tavily with depth: {self.search_depth}",
                    "search_depth": self.search_depth,
                    "topic": self.topic,
                    "include_domains": self.query_domains
                })
            
            # Perform async search
            response_data = await self._async_search(search_params)
            
            # Process results
            search_results = await self._process_tavily_response(response_data)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"✅ Tavily returned {len(search_results)} results",
                    "results_count": len(search_results),
                    "has_answer": bool(response_data.get("answer"))
                })
            
            return search_results
            
        except Exception as e:
            error_msg = f"Tavily search failed: {str(e)}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    def _prepare_search_params(self, max_results: int) -> Dict[str, Any]:
        """Prepare search parameters for Tavily API"""
        params = {
            "query": self.query,
            "search_depth": self.search_depth,
            "topic": self.topic,
            "days": self.days,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
            "max_results": max_results,
            "include_images": self.include_images,
            "api_key": self.api_key,
            "use_cache": True,
        }
        
        # Add domain filtering if specified
        if self.query_domains:
            params["include_domains"] = self.query_domains
        
        return params
    
    async def _async_search(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform async HTTP request to Tavily API"""
        headers = {
            "Content-Type": "application/json",
        }
        
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.base_url,
                data=json.dumps(search_params),
                headers=headers
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise RetrieverError(
                        f"Tavily API error {response.status}: {error_text}"
                    )
    
    async def _process_tavily_response(self, response_data: Dict[str, Any]) -> List[SearchResult]:
        """Process Tavily API response into SearchResult objects"""
        results = []
        
        # Extract search results
        tavily_results = response_data.get("results", [])
        
        for item in tavily_results:
            try:
                # Extract basic information
                title = item.get("title", "")
                content = item.get("content", "")
                url = item.get("url", "")
                
                # Skip empty results
                if not content or not url:
                    continue
                
                # Clean and process content
                content = self._sanitize_content(content)
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(title, content)
                
                # Add score boost for exact domain matches
                if self.query_domains:
                    for domain in self.query_domains:
                        if domain.lower() in url.lower():
                            relevance_score = min(relevance_score + 0.2, 1.0)
                            break
                
                # Create search result
                search_result = SearchResult(
                    title=title,
                    content=content,
                    url=url,
                    source="tavily",
                    relevance_score=relevance_score,
                    metadata={
                        "search_depth": self.search_depth,
                        "topic": self.topic,
                        "published_date": item.get("published_date", ""),
                        "raw_content_available": self.include_raw_content,
                        "tavily_score": item.get("score", 0.0)
                    }
                )
                
                results.append(search_result)
                
            except Exception as e:
                if self.logger:
                    await self.logger.log_error(f"Error processing Tavily result: {e}")
                continue
        
        # Add Tavily answer as a special result if available
        if response_data.get("answer") and self.include_answer:
            answer_result = SearchResult(
                title="Tavily AI Answer",
                content=response_data["answer"],
                url="https://tavily.com/answer",
                source="tavily_answer",
                relevance_score=0.95,  # High relevance for AI-generated answers
                metadata={
                    "type": "ai_answer",
                    "search_depth": self.search_depth,
                    "topic": self.topic
                }
            )
            results.insert(0, answer_result)  # Put answer first
        
        return results
    
    # Caching implementation
    async def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached results from Redis"""
        try:
            cached_data = await self.redis_service.get_cached_context(cache_key)
            
            if cached_data and isinstance(cached_data, list):
                # Convert cached dictionaries back to SearchResult objects
                search_results = []
                for item in cached_data:
                    if isinstance(item, dict):
                        search_result = SearchResult(
                            title=item.get("title", ""),
                            content=item.get("content", ""),
                            url=item.get("url", ""),
                            source=item.get("source", "tavily"),
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
            # Convert SearchResult objects to dictionaries for JSON serialization
            cache_data = [result.to_dict() for result in results]
            
            await self.redis_service.cache_context(
                cache_key, 
                cache_data, 
                ttl=self._cache_ttl
            )
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"💾 Cached {len(results)} Tavily results",
                    "cache_key": cache_key[:50] + "..." if len(cache_key) > 50 else cache_key,
                    "ttl": self._cache_ttl
                })
                
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Cache save error: {e}")
    
    # Enhanced search methods
    async def search_with_follow_up(
        self, 
        max_results: int = None,
        follow_up_queries: List[str] = None
    ) -> List[SearchResult]:
        """
        Enhanced search with automatic follow-up queries for better coverage
        """
        if max_results is None:
            max_results = self._max_results
        
        # Perform initial search
        initial_results = await self.search(max_results)
        
        # If we don't have enough high-quality results, try follow-up queries
        high_quality_results = [r for r in initial_results if r.relevance_score >= 0.7]
        
        if len(high_quality_results) < max_results // 2 and follow_up_queries:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": "🔄 Performing follow-up searches for better coverage",
                    "initial_results": len(initial_results),
                    "high_quality_results": len(high_quality_results)
                })
            
            all_results = initial_results.copy()
            
            for follow_up_query in follow_up_queries[:2]:  # Limit follow-ups
                # Create a new retriever instance for the follow-up query
                follow_up_retriever = TavilyRetriever(
                    query=follow_up_query,
                    headers=self.headers,
                    query_domains=self.query_domains,
                    logger=self.logger,
                    timeout=self._timeout // 2,  # Shorter timeout for follow-ups
                    search_depth="basic"  # Use basic depth for follow-ups
                )
                
                try:
                    follow_up_results = await follow_up_retriever.search(max_results // 3)
                    
                    # Add follow-up results with slightly lower scores
                    for result in follow_up_results:
                        result.relevance_score *= 0.9  # Slight penalty for follow-up
                        result.metadata["follow_up_query"] = follow_up_query
                    
                    all_results.extend(follow_up_results)
                    
                except Exception as e:
                    if self.logger:
                        await self.logger.log_error(f"Follow-up search failed: {e}")
            
            # Deduplicate and re-rank
            all_results = await self._post_process_results(all_results)
            return all_results[:max_results]
        
        return initial_results
    
    async def get_search_suggestions(self) -> List[str]:
        """Get search suggestions based on the current query"""
        # This could be enhanced to use Tavily's suggestion API if available
        # For now, generate simple variations
        suggestions = []
        
        query_words = self.query.split()
        if len(query_words) > 1:
            # Create variations by removing/adding words
            suggestions.append(f"latest {self.query}")
            suggestions.append(f"{self.query} trends")
            suggestions.append(f"{self.query} analysis")
            suggestions.append(f"future of {self.query}")
            
            # Industry-specific variations
            if any(word in self.query.lower() for word in ["industry", "market", "business"]):
                suggestions.append(f"{self.query} opportunities")
                suggestions.append(f"{self.query} challenges")
        
        return suggestions[:4]  # Limit suggestions
    
    def configure_search_depth(self, depth: Literal["basic", "advanced"]):
        """Configure search depth for subsequent searches"""
        self.search_depth = depth
    
    def configure_topic(self, topic: str):
        """Configure search topic for subsequent searches"""
        self.topic = topic
    
    async def get_tavily_stats(self) -> Dict[str, Any]:
        """Get Tavily-specific statistics"""
        stats = await self.get_search_stats()
        stats.update({
            "tavily_config": {
                "search_depth": self.search_depth,
                "topic": self.topic,
                "include_images": self.include_images,
                "include_answer": self.include_answer,
                "days_filter": self.days
            },
            "api_key_configured": bool(self.api_key),
            "cache_enabled": self._cache_enabled
        })
        return stats