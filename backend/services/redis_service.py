"""
Redis Service for Industry Reporter 2
Enhanced Redis service with async support and advanced caching features
"""
import json
import pickle
import hashlib
import asyncio
from typing import Any, Dict, List, Optional, Union, Set
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from core.config import config
from core.logging import get_logger

logger = get_logger(__name__)


class RedisService:
    """
    Enhanced Redis service with async support and advanced caching
    """
    
    def __init__(self, redis_url: str = None, **kwargs):
        self.redis_url = redis_url or config.settings.redis_url
        self.redis_client: Optional[Redis] = None
        self.connection_pool = None
        
        # Configuration
        self.default_ttl = kwargs.get('default_ttl', 3600)  # 1 hour
        self.max_connections = kwargs.get('max_connections', 20)
        self.retry_attempts = kwargs.get('retry_attempts', 3)
        self.retry_delay = kwargs.get('retry_delay', 1)
        
        # Cache namespaces
        self.context_namespace = "context"
        self.search_namespace = "search"
        self.session_namespace = "session"
        self.index_namespace = "index"
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            # Create connection pool
            self.connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Create Redis client
            self.redis_client = Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info(f"Redis service initialized successfully: {self.redis_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis service: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        
        if self.connection_pool:
            await self.connection_pool.disconnect()
            self.connection_pool = None
        
        logger.info("Redis service closed")
    
    async def is_connected(self) -> bool:
        """Check if Redis is connected"""
        try:
            if not self.redis_client:
                return False
            await self.redis_client.ping()
            return True
        except:
            return False
    
    async def ensure_connected(self):
        """Ensure Redis is connected, reconnect if necessary"""
        if not await self.is_connected():
            await self.initialize()
    
    # Basic cache operations
    async def set_cache(
        self, 
        key: str, 
        value: Any, 
        ttl: int = None,
        serialize: bool = True
    ) -> bool:
        """Set a cache entry"""
        try:
            await self.ensure_connected()
            
            if ttl is None:
                ttl = self.default_ttl
            
            # Serialize value if needed
            if serialize:
                if isinstance(value, (dict, list)):
                    serialized_value = json.dumps(value, ensure_ascii=False)
                else:
                    serialized_value = pickle.dumps(value)
            else:
                serialized_value = value
            
            # Set with TTL
            result = await self.redis_client.setex(key, ttl, serialized_value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    async def get_cache(
        self, 
        key: str, 
        deserialize: bool = True,
        default: Any = None
    ) -> Any:
        """Get a cache entry"""
        try:
            await self.ensure_connected()
            
            value = await self.redis_client.get(key)
            
            if value is None:
                return default
            
            # Deserialize value if needed
            if deserialize:
                try:
                    # Try JSON first
                    return json.loads(value)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    try:
                        # Fall back to pickle
                        return pickle.loads(value)
                    except:
                        # Return raw value
                        return value.decode('utf-8') if isinstance(value, bytes) else value
            else:
                return value.decode('utf-8') if isinstance(value, bytes) else value
                
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return default
    
    async def delete_cache(self, key: str) -> bool:
        """Delete a cache entry"""
        try:
            await self.ensure_connected()
            result = await self.redis_client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a cache key exists"""
        try:
            await self.ensure_connected()
            result = await self.redis_client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get TTL for a key"""
        try:
            await self.ensure_connected()
            return await self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Failed to get TTL for key {key}: {e}")
            return -2  # Key doesn't exist
    
    # Advanced operations
    async def get_multiple(self, keys: List[str]) -> List[Any]:
        """Get multiple cache entries at once"""
        try:
            await self.ensure_connected()
            
            if not keys:
                return []
            
            # Use pipeline for efficiency
            pipe = self.redis_client.pipeline()
            for key in keys:
                pipe.get(key)
            
            values = await pipe.execute()
            
            # Deserialize values
            results = []
            for value in values:
                if value is None:
                    results.append(None)
                else:
                    try:
                        # Try JSON first
                        results.append(json.loads(value))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        try:
                            # Fall back to pickle
                            results.append(pickle.loads(value))
                        except:
                            # Return raw value
                            results.append(value.decode('utf-8') if isinstance(value, bytes) else value)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get multiple keys: {e}")
            return [None] * len(keys)
    
    async def set_multiple(
        self, 
        key_value_pairs: Dict[str, Any], 
        ttl: int = None
    ) -> bool:
        """Set multiple cache entries at once"""
        try:
            await self.ensure_connected()
            
            if not key_value_pairs:
                return True
            
            if ttl is None:
                ttl = self.default_ttl
            
            # Use pipeline for efficiency
            pipe = self.redis_client.pipeline()
            
            for key, value in key_value_pairs.items():
                # Serialize value
                if isinstance(value, (dict, list)):
                    serialized_value = json.dumps(value, ensure_ascii=False)
                else:
                    serialized_value = pickle.dumps(value)
                
                pipe.setex(key, ttl, serialized_value)
            
            results = await pipe.execute()
            return all(results)
            
        except Exception as e:
            logger.error(f"Failed to set multiple keys: {e}")
            return False
    
    async def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get keys matching a pattern"""
        try:
            await self.ensure_connected()
            
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                if isinstance(key, bytes):
                    keys.append(key.decode('utf-8'))
                else:
                    keys.append(key)
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to get keys by pattern {pattern}: {e}")
            return []
    
    async def delete_by_pattern(self, pattern: str) -> int:
        """Delete keys matching a pattern"""
        try:
            await self.ensure_connected()
            
            keys = await self.get_keys_by_pattern(pattern)
            
            if not keys:
                return 0
            
            # Delete in batches
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                result = await self.redis_client.delete(*batch_keys)
                deleted_count += result
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete keys by pattern {pattern}: {e}")
            return 0
    
    # Context-specific operations
    async def cache_context(
        self, 
        context_key: str, 
        context_data: Any, 
        ttl: int = None
    ) -> bool:
        """Cache research context"""
        full_key = f"{self.context_namespace}:{context_key}"
        return await self.set_cache(full_key, context_data, ttl)
    
    async def get_cached_context(self, context_key: str) -> Any:
        """Get cached research context"""
        full_key = f"{self.context_namespace}:{context_key}"
        return await self.get_cache(full_key)
    
    async def cache_search_results(
        self, 
        query_hash: str, 
        results: List[Any], 
        ttl: int = None
    ) -> bool:
        """Cache search results"""
        full_key = f"{self.search_namespace}:{query_hash}"
        return await self.set_cache(full_key, results, ttl)
    
    async def get_cached_search_results(self, query_hash: str) -> Optional[List[Any]]:
        """Get cached search results"""
        full_key = f"{self.search_namespace}:{query_hash}"
        return await self.get_cache(full_key)
    
    async def cache_session_data(
        self, 
        session_id: str, 
        session_data: Dict[str, Any], 
        ttl: int = None
    ) -> bool:
        """Cache session data"""
        full_key = f"{self.session_namespace}:{session_id}"
        return await self.set_cache(full_key, session_data, ttl)
    
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        full_key = f"{self.session_namespace}:{session_id}"
        return await self.get_cache(full_key)
    
    # Utility methods
    async def generate_cache_key(self, *components: str) -> str:
        """Generate a consistent cache key from components"""
        key_string = ":".join(str(c) for c in components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get Redis memory usage statistics"""
        try:
            await self.ensure_connected()
            
            info = await self.redis_client.info('memory')
            
            return {
                "used_memory": info.get('used_memory', 0),
                "used_memory_human": info.get('used_memory_human', '0B'),
                "used_memory_peak": info.get('used_memory_peak', 0),
                "used_memory_peak_human": info.get('used_memory_peak_human', '0B'),
                "total_system_memory": info.get('total_system_memory', 0),
                "maxmemory": info.get('maxmemory', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        try:
            await self.ensure_connected()
            
            # Get basic info
            info = await self.redis_client.info()
            
            # Count keys by namespace
            namespace_counts = {}
            for namespace in [self.context_namespace, self.search_namespace, 
                            self.session_namespace, self.index_namespace]:
                pattern = f"{namespace}:*"
                keys = await self.get_keys_by_pattern(pattern)
                namespace_counts[namespace] = len(keys)
            
            # Get memory usage
            memory_info = await self.get_memory_usage()
            
            return {
                "redis_version": info.get('redis_version', 'unknown'),
                "uptime_in_seconds": info.get('uptime_in_seconds', 0),
                "connected_clients": info.get('connected_clients', 0),
                "total_commands_processed": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "namespace_counts": namespace_counts,
                "memory_usage": memory_info,
                "connection_url": self.redis_url
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {"error": str(e)}
    
    async def flush_namespace(self, namespace: str) -> int:
        """Flush all keys in a specific namespace"""
        pattern = f"{namespace}:*"
        return await self.delete_by_pattern(pattern)
    
    async def cleanup_expired_keys(self) -> Dict[str, int]:
        """Clean up expired keys (Redis handles this automatically, but useful for monitoring)"""
        try:
            await self.ensure_connected()
            
            cleanup_stats = {}
            
            for namespace in [self.context_namespace, self.search_namespace, 
                            self.session_namespace, self.index_namespace]:
                pattern = f"{namespace}:*"
                keys = await self.get_keys_by_pattern(pattern)
                
                expired_count = 0
                for key in keys:
                    ttl = await self.get_ttl(key)
                    if ttl == -2:  # Key doesn't exist (expired)
                        expired_count += 1
                
                cleanup_stats[namespace] = {
                    "total_keys": len(keys),
                    "expired_keys": expired_count
                }
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            return {}
    
    # Health check
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            start_time = datetime.now()
            
            # Test basic operations
            test_key = "health_check_test"
            test_value = {"timestamp": start_time.isoformat()}
            
            # Test set
            set_success = await self.set_cache(test_key, test_value, ttl=60)
            
            # Test get
            retrieved_value = await self.get_cache(test_key)
            get_success = retrieved_value is not None
            
            # Test delete
            delete_success = await self.delete_cache(test_key)
            
            # Calculate latency
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Get cache info
            cache_info = await self.get_cache_info()
            
            return {
                "status": "healthy" if all([set_success, get_success, delete_success]) else "unhealthy",
                "latency_ms": latency_ms,
                "operations": {
                    "set": set_success,
                    "get": get_success,
                    "delete": delete_success
                },
                "cache_info": cache_info,
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global Redis service instance
redis_service = RedisService()


# Convenience functions
async def get_redis_service() -> RedisService:
    """Get the global Redis service instance"""
    if not await redis_service.is_connected():
        await redis_service.initialize()
    return redis_service


async def cache_data(key: str, data: Any, ttl: int = None) -> bool:
    """Convenience function to cache data"""
    service = await get_redis_service()
    return await service.set_cache(key, data, ttl)


async def get_cached_data(key: str, default: Any = None) -> Any:
    """Convenience function to get cached data"""
    service = await get_redis_service()
    return await service.get_cache(key, default=default)