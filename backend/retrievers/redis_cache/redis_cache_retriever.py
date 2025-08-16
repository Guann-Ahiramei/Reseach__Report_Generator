"""
Redis Cache Retriever for Industry Reporter 2
Pure Redis-based caching retriever for fast context access
"""
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Set, Union
from datetime import datetime, timedelta

from core.config import config
from ..base_retriever import CachedRetriever, SearchResult, RetrieverError
from services.redis_service import RedisService


class RedisCacheRetriever(CachedRetriever):
    """
    Pure Redis cache retriever for ultra-fast cached context access
    """
    
    def __init__(self, query: str, **kwargs):
        super().__init__(query, **kwargs)
        
        # Redis cache configuration
        self.cache_namespace = kwargs.get('cache_namespace', 'research_cache')
        self.cache_patterns = kwargs.get('cache_patterns', ['*'])
        self.min_cache_age = kwargs.get('min_cache_age', 0)  # Minimum age in seconds
        self.max_cache_age = kwargs.get('max_cache_age', 86400)  # Maximum age in seconds (24h)
        self.include_metadata = kwargs.get('include_metadata', True)
        
        # Search configuration
        self.search_mode = kwargs.get('search_mode', 'fuzzy')  # fuzzy, exact, pattern
        self.boost_recent = kwargs.get('boost_recent', True)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.3)
        
        # Redis service
        self.redis_service = RedisService()
        
        # Performance tracking
        self._cache_stats = {}
    
    @property
    def name(self) -> str:
        return "Redis Cache"
    
    @property
    def source_type(self) -> str:
        return "cache"
    
    async def _search_impl(self, max_results: int) -> List[SearchResult]:
        """Implementation of Redis cache search"""
        try:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🗄️ Searching Redis cache",
                    "cache_namespace": self.cache_namespace,
                    "search_mode": self.search_mode,
                    "cache_patterns": self.cache_patterns
                })
            
            # Get cache keys matching patterns
            cache_keys = await self._get_matching_cache_keys()
            
            if not cache_keys:
                if self.logger:
                    await self.logger.send_json({
                        "type": "logs",
                        "content": "📭 No cache entries found matching criteria",
                    })
                return []
            
            # Search through cached entries
            search_results = await self._search_cache_entries(cache_keys, max_results)
            
            # Track performance
            self._cache_stats = {
                "total_cache_keys": len(cache_keys),
                "results_found": len(search_results),
                "search_mode": self.search_mode,
                "cache_hit_rate": len(search_results) / len(cache_keys) if cache_keys else 0
            }
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"💾 Found {len(search_results)} cached results",
                    "cache_keys_searched": len(cache_keys),
                    "results_found": len(search_results),
                    "cache_hit_rate": f"{self._cache_stats['cache_hit_rate']:.2%}"
                })
            
            return search_results
            
        except Exception as e:
            error_msg = f"Redis cache search failed: {str(e)}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    async def _get_matching_cache_keys(self) -> List[str]:
        """Get cache keys matching the specified patterns"""
        try:
            all_keys = []
            
            for pattern in self.cache_patterns:
                # Build full pattern with namespace
                full_pattern = f"{self.cache_namespace}:{pattern}"
                
                # Get keys matching pattern
                pattern_keys = await self.redis_service.get_keys_by_pattern(full_pattern)
                all_keys.extend(pattern_keys)
            
            # Remove duplicates while preserving order
            unique_keys = list(dict.fromkeys(all_keys))
            
            # Filter by age if specified
            if self.min_cache_age > 0 or self.max_cache_age < float('inf'):
                filtered_keys = await self._filter_keys_by_age(unique_keys)
                return filtered_keys
            
            return unique_keys
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Error getting cache keys: {e}")
            return []
    
    async def _filter_keys_by_age(self, keys: List[str]) -> List[str]:
        """Filter cache keys by age"""
        filtered_keys = []
        current_time = time.time()
        
        for key in keys:
            try:
                # Get key TTL or creation time
                ttl = await self.redis_service.get_ttl(key)
                
                if ttl == -1:  # No expiration
                    continue
                elif ttl == -2:  # Key doesn't exist
                    continue
                
                # Estimate age (this is approximate)
                # For better age tracking, consider storing creation timestamps
                estimated_age = self.max_cache_age - ttl if ttl > 0 else 0
                
                if self.min_cache_age <= estimated_age <= self.max_cache_age:
                    filtered_keys.append(key)
                    
            except Exception as e:
                # If we can't determine age, include the key
                filtered_keys.append(key)
                continue
        
        return filtered_keys
    
    async def _search_cache_entries(
        self, 
        cache_keys: List[str], 
        max_results: int
    ) -> List[SearchResult]:
        """Search through cached entries"""
        search_results = []
        query_words = set(self.query.lower().split())
        
        # Process cache entries in batches for better performance
        batch_size = 50
        for i in range(0, len(cache_keys), batch_size):
            batch_keys = cache_keys[i:i + batch_size]
            
            try:
                # Get cached data for this batch
                batch_data = await self.redis_service.get_multiple(batch_keys)
                
                for key, cached_data in zip(batch_keys, batch_data):
                    if cached_data is None:
                        continue
                    
                    try:
                        # Process cached entry
                        results = await self._process_cache_entry(
                            key, cached_data, query_words
                        )
                        search_results.extend(results)
                        
                        # Early exit if we have enough results
                        if len(search_results) >= max_results * 2:
                            break
                            
                    except Exception as e:
                        if self.logger:
                            await self.logger.log_error(f"Error processing cache entry {key}: {e}")
                        continue
                
                # Early exit if we have enough results
                if len(search_results) >= max_results * 2:
                    break
                    
            except Exception as e:
                if self.logger:
                    await self.logger.log_error(f"Error processing batch: {e}")
                continue
        
        # Sort by relevance and return top results
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return search_results[:max_results]
    
    async def _process_cache_entry(
        self, 
        cache_key: str, 
        cached_data: Any, 
        query_words: Set[str]
    ) -> List[SearchResult]:
        """Process a single cache entry"""
        results = []
        
        try:
            # Handle different data formats
            if isinstance(cached_data, str):
                try:
                    data = json.loads(cached_data)
                except json.JSONDecodeError:
                    # Treat as plain text
                    data = {"content": cached_data}
            elif isinstance(cached_data, dict):
                data = cached_data
            elif isinstance(cached_data, list):
                # Handle list of cached search results
                for item in cached_data:
                    if isinstance(item, dict):
                        sub_results = await self._process_cache_entry(
                            cache_key, item, query_words
                        )
                        results.extend(sub_results)
                return results
            else:
                # Unknown format
                return results
            
            # Extract content for search
            searchable_content = self._extract_searchable_content(data)
            
            if not searchable_content:
                return results
            
            # Calculate relevance score
            relevance_score = self._calculate_cache_relevance(
                searchable_content, query_words, cache_key
            )
            
            # Apply similarity threshold
            if relevance_score < self.similarity_threshold:
                return results
            
            # Create SearchResult
            search_result = await self._create_search_result_from_cache(
                data, cache_key, relevance_score
            )
            
            if search_result:
                results.append(search_result)
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Error processing cache data: {e}")
        
        return results
    
    def _extract_searchable_content(self, data: Dict[str, Any]) -> str:
        """Extract searchable text content from cached data"""
        content_parts = []
        
        # Common content fields
        content_fields = ["content", "text", "description", "summary", "title"]
        
        for field in content_fields:
            if field in data and data[field]:
                content_parts.append(str(data[field]))
        
        # Handle nested structures
        if "metadata" in data and isinstance(data["metadata"], dict):
            for value in data["metadata"].values():
                if isinstance(value, str) and value:
                    content_parts.append(value)
        
        # Handle search results format
        if "results" in data and isinstance(data["results"], list):
            for result in data["results"]:
                if isinstance(result, dict):
                    sub_content = self._extract_searchable_content(result)
                    if sub_content:
                        content_parts.append(sub_content)
        
        return " ".join(content_parts)
    
    def _calculate_cache_relevance(
        self, 
        content: str, 
        query_words: Set[str], 
        cache_key: str
    ) -> float:
        """Calculate relevance score for cached content"""
        if not content or not query_words:
            return 0.0
        
        content_lower = content.lower()
        content_words = set(content_lower.split())
        
        # Basic word matching score
        word_matches = len(query_words.intersection(content_words))
        base_score = word_matches / len(query_words) if query_words else 0.0
        
        # Boost for exact phrase matches
        if self.search_mode == 'exact':
            if self.query.lower() in content_lower:
                base_score += 0.3
        
        # Boost for cache key relevance
        key_lower = cache_key.lower()
        key_matches = sum(1 for word in query_words if word in key_lower)
        key_score = (key_matches / len(query_words)) * 0.2 if query_words else 0.0
        
        # Boost for recent cache entries
        if self.boost_recent:
            # This is a simplified recency boost
            # In practice, you'd want to store actual timestamps
            if "recent" in cache_key or "latest" in cache_key:
                base_score += 0.1
        
        total_score = min(base_score + key_score, 1.0)
        return total_score
    
    async def _create_search_result_from_cache(
        self, 
        data: Dict[str, Any], 
        cache_key: str, 
        relevance_score: float
    ) -> Optional[SearchResult]:
        """Create SearchResult from cached data"""
        try:
            # Extract or generate title
            title = data.get("title", "")
            if not title:
                # Generate title from cache key
                title = cache_key.split(":")[-1].replace("_", " ").title()
                if len(title) > 100:
                    title = title[:97] + "..."
            
            # Extract content
            content = self._extract_searchable_content(data)
            if not content:
                return None
            
            # Sanitize content
            content = self._sanitize_content(content)
            
            # Extract URL
            url = data.get("url", f"cache://{cache_key}")
            
            # Extract source
            source = data.get("source", "redis_cache")
            
            # Build metadata
            metadata = {
                "cache_key": cache_key,
                "cache_namespace": self.cache_namespace,
                "search_method": "cache_lookup",
                "cache_score": relevance_score,
                "search_mode": self.search_mode
            }
            
            # Include original metadata if available
            if self.include_metadata and "metadata" in data:
                original_metadata = data["metadata"]
                if isinstance(original_metadata, dict):
                    metadata.update(original_metadata)
            
            # Add cache timing info
            try:
                ttl = await self.redis_service.get_ttl(cache_key)
                if ttl > 0:
                    metadata["cache_ttl"] = ttl
                    metadata["estimated_expiry"] = (
                        datetime.now() + timedelta(seconds=ttl)
                    ).isoformat()
            except:
                pass
            
            return SearchResult(
                title=title,
                content=content,
                url=url,
                source=f"cache_{source}" if source != "redis_cache" else "redis_cache",
                relevance_score=relevance_score,
                metadata=metadata
            )
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Error creating search result from cache: {e}")
            return None
    
    # Cache management methods
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            # Get basic cache info
            cache_info = await self.redis_service.get_cache_info()
            
            # Get namespace-specific stats
            namespace_keys = await self.redis_service.get_keys_by_pattern(
                f"{self.cache_namespace}:*"
            )
            
            # Analyze cache patterns
            pattern_stats = {}
            for pattern in self.cache_patterns:
                pattern_keys = await self.redis_service.get_keys_by_pattern(
                    f"{self.cache_namespace}:{pattern}"
                )
                pattern_stats[pattern] = len(pattern_keys)
            
            return {
                "cache_config": {
                    "namespace": self.cache_namespace,
                    "patterns": self.cache_patterns,
                    "search_mode": self.search_mode,
                    "min_age": self.min_cache_age,
                    "max_age": self.max_cache_age
                },
                "cache_statistics": {
                    "total_namespace_keys": len(namespace_keys),
                    "pattern_breakdown": pattern_stats,
                    "redis_info": cache_info
                },
                "last_search_stats": self._cache_stats
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def clear_cache_namespace(self, confirm: bool = False):
        """Clear all cache entries in the current namespace"""
        if not confirm:
            raise ValueError("Must explicitly confirm cache clearing")
        
        try:
            pattern = f"{self.cache_namespace}:*"
            deleted_count = await self.redis_service.delete_by_pattern(pattern)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🗑️ Cleared {deleted_count} cache entries",
                    "namespace": self.cache_namespace,
                    "deleted_count": deleted_count
                })
            
            return deleted_count
            
        except Exception as e:
            error_msg = f"Cache clearing failed: {e}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    async def refresh_cache_entry(self, cache_key: str, new_data: Any):
        """Refresh a specific cache entry"""
        try:
            full_key = f"{self.cache_namespace}:{cache_key}"
            
            # Update cache entry
            await self.redis_service.set_cache(
                full_key, 
                new_data, 
                ttl=self._cache_ttl
            )
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🔄 Refreshed cache entry: {cache_key}",
                    "cache_key": cache_key
                })
                
        except Exception as e:
            error_msg = f"Cache refresh failed: {e}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    # Search configuration methods
    def configure_search_mode(self, mode: str):
        """Configure search mode (fuzzy, exact, pattern)"""
        valid_modes = ['fuzzy', 'exact', 'pattern']
        if mode in valid_modes:
            self.search_mode = mode
        else:
            raise ValueError(f"Search mode must be one of: {valid_modes}")
    
    def configure_cache_patterns(self, patterns: List[str]):
        """Configure cache search patterns"""
        self.cache_patterns = patterns if patterns else ['*']
    
    def configure_age_filter(self, min_age: int = 0, max_age: int = 86400):
        """Configure cache age filtering"""
        self.min_cache_age = max(0, min_age)
        self.max_cache_age = max(min_age, max_age)
    
    # Integration with other retrievers
    async def seed_from_retriever_results(
        self, 
        results: List[SearchResult], 
        cache_prefix: str = "seeded"
    ):
        """Seed cache with results from other retrievers"""
        try:
            seeded_count = 0
            
            for i, result in enumerate(results):
                cache_key = f"{self.cache_namespace}:{cache_prefix}:{i}:{int(time.time())}"
                
                # Convert SearchResult to cacheable format
                cache_data = result.to_dict()
                
                await self.redis_service.set_cache(
                    cache_key,
                    cache_data,
                    ttl=self._cache_ttl
                )
                
                seeded_count += 1
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🌱 Seeded {seeded_count} results into cache",
                    "seeded_count": seeded_count,
                    "cache_prefix": cache_prefix
                })
            
            return seeded_count
            
        except Exception as e:
            error_msg = f"Cache seeding failed: {e}"
            if self.logger:
                await self.logger.log_error(error_msg)
            raise RetrieverError(error_msg) from e
    
    # Disable base class caching for this retriever
    async def _get_from_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Disabled - this retriever IS the cache"""
        return None
    
    async def _save_to_cache(self, cache_key: str, results: List[SearchResult]):
        """Disabled - this retriever IS the cache"""
        pass