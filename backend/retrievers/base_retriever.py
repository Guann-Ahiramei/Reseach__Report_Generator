"""
Base Retriever Interface for Industry Reporter 2
Defines the common interface for all retrievers
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from core.config import config
from core.logging import CustomLogsHandler


@dataclass
class SearchResult:
    """Standardized search result structure"""
    title: str
    content: str
    url: str
    source: str = "unknown"
    relevance_score: float = 0.0
    timestamp: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            # Legacy format compatibility
            "href": self.url,
            "body": self.content
        }


class RetrieverError(Exception):
    """Base exception for retriever errors"""
    pass


class RetrieverTimeoutError(RetrieverError):
    """Raised when retriever operation times out"""
    pass


class RetrieverConfigError(RetrieverError):
    """Raised when retriever configuration is invalid"""
    pass


class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers
    Provides common functionality and interface definition
    """
    
    def __init__(
        self,
        query: str,
        headers: Dict[str, str] = None,
        query_domains: List[str] = None,
        websocket=None,
        logger: CustomLogsHandler = None,
        **kwargs
    ):
        self.query = query
        self.headers = headers or {}
        self.query_domains = query_domains or []
        self.websocket = websocket
        self.logger = logger
        self.kwargs = kwargs
        
        # Performance tracking
        self._search_start_time = None
        self._search_stats = {}
        
        # Configuration
        self._timeout = kwargs.get('timeout', config.settings.search_timeout_seconds)
        self._max_results = kwargs.get('max_results', config.settings.max_search_results_per_query)
        
        # Validate configuration
        self._validate_config()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the retriever"""
        pass
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the type of source this retriever searches"""
        pass
    
    @abstractmethod
    async def _search_impl(self, max_results: int) -> List[SearchResult]:
        """
        Implementation-specific search logic
        Must be implemented by each retriever
        """
        pass
    
    async def search(self, max_results: int = None) -> List[SearchResult]:
        """
        Main search method with timeout, error handling, and logging
        """
        if max_results is None:
            max_results = self._max_results
        
        self._search_start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔍 Starting {self.name} search for: '{self.query}'",
                "retriever": self.name,
                "max_results": max_results
            })
        
        try:
            # Execute search with timeout
            results = await asyncio.wait_for(
                self._search_impl(max_results),
                timeout=self._timeout
            )
            
            # Post-process results
            processed_results = await self._post_process_results(results)
            
            # Track performance
            search_time = time.time() - self._search_start_time
            self._search_stats = {
                "search_time": search_time,
                "results_count": len(processed_results),
                "success": True
            }
            
            if self.logger:
                await self.logger.log_performance_metric(
                    f"{self.name.lower()}_search_time", 
                    search_time, 
                    "seconds"
                )
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"✅ {self.name}: {len(processed_results)} results in {search_time:.2f}s",
                    "retriever": self.name,
                    "results_count": len(processed_results),
                    "search_time": search_time
                })
            
            return processed_results
            
        except asyncio.TimeoutError:
            error_msg = f"{self.name} search timed out after {self._timeout}s"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverTimeoutError(error_msg)
            
        except Exception as e:
            error_msg = f"{self.name} search failed: {str(e)}"
            if self.logger:
                await self.logger.log_error(error_msg)
            
            self._search_stats = {
                "search_time": time.time() - self._search_start_time if self._search_start_time else 0,
                "results_count": 0,
                "success": False,
                "error": str(e)
            }
            
            raise RetrieverError(error_msg) from e
    
    async def _post_process_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Post-process search results (filtering, scoring, etc.)
        Can be overridden by specific retrievers
        """
        if not results:
            return []
        
        # Apply domain filtering if specified
        if self.query_domains:
            filtered_results = []
            for result in results:
                if any(domain.lower() in result.url.lower() for domain in self.query_domains):
                    filtered_results.append(result)
            results = filtered_results
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Sort by relevance score (descending)
        unique_results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        return unique_results
    
    def _validate_config(self):
        """Validate retriever configuration"""
        if not self.query:
            raise RetrieverConfigError("Query cannot be empty")
        
        if self._timeout <= 0:
            raise RetrieverConfigError("Timeout must be positive")
        
        if self._max_results <= 0:
            raise RetrieverConfigError("Max results must be positive")
    
    def _sanitize_content(self, content: str) -> str:
        """Sanitize and clean content"""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = " ".join(content.split())
        
        # Limit content length
        max_content_length = 5000  # Configurable limit
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        return content
    
    def _calculate_relevance_score(self, title: str, content: str) -> float:
        """
        Calculate relevance score based on query match
        Simple implementation - can be enhanced with ML models
        """
        query_words = set(self.query.lower().split())
        title_words = set(title.lower().split())
        content_words = set(content.lower().split())
        
        # Title matches are weighted more heavily
        title_matches = len(query_words.intersection(title_words))
        content_matches = len(query_words.intersection(content_words))
        
        total_query_words = len(query_words)
        if total_query_words == 0:
            return 0.0
        
        # Weighted scoring: title worth 3x content
        score = (title_matches * 3 + content_matches) / (total_query_words * 4)
        
        # Ensure score is between 0 and 1
        return min(max(score, 0.0), 1.0)
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        return {
            "retriever_name": self.name,
            "source_type": self.source_type,
            "last_search_stats": self._search_stats,
            "configuration": {
                "timeout": self._timeout,
                "max_results": self._max_results,
                "query_domains": self.query_domains
            }
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(query='{self.query}', max_results={self._max_results})"


class CachedRetriever(BaseRetriever):
    """
    Base class for retrievers that support caching
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_enabled = kwargs.get('cache_enabled', True)
        self._cache_ttl = kwargs.get('cache_ttl', config.settings.cache_ttl_seconds)
    
    async def search(self, max_results: int = None) -> List[SearchResult]:
        """Enhanced search with caching support"""
        if not self._cache_enabled:
            return await super().search(max_results)
        
        # Generate cache key
        cache_key = self._generate_cache_key(max_results)
        
        # Try to get from cache first
        cached_results = await self._get_from_cache(cache_key)
        if cached_results:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📋 {self.name}: Using cached results",
                    "retriever": self.name,
                    "cache_hit": True,
                    "results_count": len(cached_results)
                })
            return cached_results
        
        # Perform actual search
        results = await super().search(max_results)
        
        # Cache the results
        if results:
            await self._save_to_cache(cache_key, results)
        
        return results
    
    @abstractmethod
    async def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get results from cache"""
        pass
    
    @abstractmethod
    async def _save_to_cache(self, cache_key: str, results: List[SearchResult]):
        """Save results to cache"""
        pass
    
    def _generate_cache_key(self, max_results: int = None) -> str:
        """Generate cache key for the search"""
        if max_results is None:
            max_results = self._max_results
        
        # Include relevant parameters in cache key
        key_components = [
            self.name,
            self.query,
            str(max_results),
            str(sorted(self.query_domains)) if self.query_domains else "no_domains"
        ]
        
        cache_key = "|".join(key_components)
        return f"retriever:{hash(cache_key)}"