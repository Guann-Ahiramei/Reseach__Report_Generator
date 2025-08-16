"""
Redis Service End-to-End Tests
Comprehensive testing for Redis service functionality
"""
import asyncio
import pytest
import json
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.redis_service import RedisService


class TestRedisService:
    """Test Redis service end-to-end functionality"""
    
    @pytest.fixture
    async def redis_service(self):
        """Create Redis service instance for testing"""
        service = RedisService(redis_url="redis://localhost:6379/15")  # Use test database
        try:
            await service.initialize()
            yield service
        finally:
            # Cleanup test data
            try:
                await service.redis_client.flushdb()  # Clear test database
                await service.close()
            except:
                pass
    
    async def test_redis_connection(self, redis_service):
        """Test Redis connection and basic operations"""
        # Test connection
        assert await redis_service.is_connected() == True
        
        # Test ping
        result = await redis_service.redis_client.ping()
        assert result == True
        
        print("✅ Redis connection test passed")
    
    async def test_basic_cache_operations(self, redis_service):
        """Test basic cache set/get/delete operations"""
        # Test string data
        test_key = "test_string"
        test_value = "Hello, Redis!"
        
        # Set cache
        result = await redis_service.set_cache(test_key, test_value, ttl=60)
        assert result == True
        
        # Get cache
        retrieved_value = await redis_service.get_cache(test_key)
        assert retrieved_value == test_value
        
        # Test TTL
        ttl = await redis_service.get_ttl(test_key)
        assert 0 < ttl <= 60
        
        # Delete cache
        result = await redis_service.delete_cache(test_key)
        assert result == True
        
        # Verify deletion
        retrieved_value = await redis_service.get_cache(test_key)
        assert retrieved_value is None
        
        print("✅ Basic cache operations test passed")
    
    async def test_json_serialization(self, redis_service):
        """Test JSON data serialization/deserialization"""
        test_key = "test_json"
        test_data = {
            "name": "Industry Reporter Test",
            "version": "2.0",
            "features": ["FAISS", "Redis", "Multi-Retriever"],
            "config": {
                "max_results": 10,
                "timeout": 30
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Set JSON data
        result = await redis_service.set_cache(test_key, test_data, ttl=60)
        assert result == True
        
        # Get JSON data
        retrieved_data = await redis_service.get_cache(test_key)
        assert isinstance(retrieved_data, dict)
        assert retrieved_data["name"] == test_data["name"]
        assert retrieved_data["features"] == test_data["features"]
        assert retrieved_data["config"]["max_results"] == test_data["config"]["max_results"]
        
        print("✅ JSON serialization test passed")
    
    async def test_batch_operations(self, redis_service):
        """Test batch set/get operations"""
        test_data = {
            "batch_key_1": {"type": "search_result", "score": 0.95},
            "batch_key_2": {"type": "document", "content": "Test content"},
            "batch_key_3": {"type": "metadata", "source": "test"}
        }
        
        # Batch set
        result = await redis_service.set_multiple(test_data, ttl=60)
        assert result == True
        
        # Batch get
        keys = list(test_data.keys())
        retrieved_values = await redis_service.get_multiple(keys)
        
        assert len(retrieved_values) == len(keys)
        for i, key in enumerate(keys):
            assert retrieved_values[i] is not None
            assert retrieved_values[i]["type"] == test_data[key]["type"]
        
        print("✅ Batch operations test passed")
    
    async def test_pattern_operations(self, redis_service):
        """Test pattern-based key operations"""
        # Set test data with patterns
        test_patterns = {
            "search:query1:result1": {"content": "Result 1"},
            "search:query1:result2": {"content": "Result 2"},
            "search:query2:result1": {"content": "Result 3"},
            "context:session1:data": {"session": "Session 1"},
            "context:session2:data": {"session": "Session 2"}
        }
        
        # Set all test data
        for key, value in test_patterns.items():
            await redis_service.set_cache(key, value, ttl=60)
        
        # Test pattern matching
        search_keys = await redis_service.get_keys_by_pattern("search:*")
        assert len(search_keys) >= 3
        assert any("query1" in key for key in search_keys)
        assert any("query2" in key for key in search_keys)
        
        context_keys = await redis_service.get_keys_by_pattern("context:*")
        assert len(context_keys) >= 2
        assert any("session1" in key for key in context_keys)
        assert any("session2" in key for key in context_keys)
        
        # Test pattern deletion
        deleted_count = await redis_service.delete_by_pattern("search:query1:*")
        assert deleted_count >= 2
        
        # Verify deletion
        remaining_search_keys = await redis_service.get_keys_by_pattern("search:*")
        assert len(remaining_search_keys) < len(search_keys)
        
        print("✅ Pattern operations test passed")
    
    async def test_namespace_operations(self, redis_service):
        """Test namespace-specific operations"""
        # Test context caching
        context_key = "test_research_session"
        context_data = {
            "query": "AI industry trends",
            "sources": ["source1", "source2"],
            "findings": ["finding1", "finding2"],
            "timestamp": datetime.now().isoformat()
        }
        
        result = await redis_service.cache_context(context_key, context_data, ttl=120)
        assert result == True
        
        retrieved_context = await redis_service.get_cached_context(context_key)
        assert retrieved_context is not None
        assert retrieved_context["query"] == context_data["query"]
        assert len(retrieved_context["sources"]) == len(context_data["sources"])
        
        # Test search results caching
        query_hash = "query_123456"
        search_results = [
            {"title": "Result 1", "content": "Content 1", "score": 0.9},
            {"title": "Result 2", "content": "Content 2", "score": 0.8}
        ]
        
        result = await redis_service.cache_search_results(query_hash, search_results, ttl=300)
        assert result == True
        
        retrieved_results = await redis_service.get_cached_search_results(query_hash)
        assert retrieved_results is not None
        assert len(retrieved_results) == len(search_results)
        assert retrieved_results[0]["title"] == search_results[0]["title"]
        
        # Test session data caching
        session_id = "session_789"
        session_data = {
            "user_id": "user123",
            "research_topic": "Industry Analysis",
            "start_time": datetime.now().isoformat(),
            "status": "active"
        }
        
        result = await redis_service.cache_session_data(session_id, session_data, ttl=1800)
        assert result == True
        
        retrieved_session = await redis_service.get_session_data(session_id)
        assert retrieved_session is not None
        assert retrieved_session["user_id"] == session_data["user_id"]
        assert retrieved_session["status"] == session_data["status"]
        
        print("✅ Namespace operations test passed")
    
    async def test_memory_and_info(self, redis_service):
        """Test memory usage and cache info retrieval"""
        # Get memory usage
        memory_info = await redis_service.get_memory_usage()
        assert isinstance(memory_info, dict)
        assert "used_memory" in memory_info
        assert "used_memory_human" in memory_info
        
        # Get cache info
        cache_info = await redis_service.get_cache_info()
        assert isinstance(cache_info, dict)
        assert "redis_version" in cache_info
        assert "namespace_counts" in cache_info
        assert "memory_usage" in cache_info
        
        print("✅ Memory and info test passed")
    
    async def test_health_check(self, redis_service):
        """Test Redis service health check"""
        health_result = await redis_service.health_check()
        
        assert isinstance(health_result, dict)
        assert "status" in health_result
        assert health_result["status"] in ["healthy", "unhealthy"]
        assert "latency_ms" in health_result
        assert "operations" in health_result
        assert "timestamp" in health_result
        
        # If healthy, check operation results
        if health_result["status"] == "healthy":
            operations = health_result["operations"]
            assert operations["set"] == True
            assert operations["get"] == True
            assert operations["delete"] == True
        
        print("✅ Health check test passed")
    
    async def test_error_handling(self, redis_service):
        """Test error handling and edge cases"""
        # Test non-existent key
        result = await redis_service.get_cache("non_existent_key")
        assert result is None
        
        # Test deletion of non-existent key
        result = await redis_service.delete_cache("non_existent_key")
        assert result == False
        
        # Test TTL of non-existent key
        ttl = await redis_service.get_ttl("non_existent_key")
        assert ttl == -2  # Key doesn't exist
        
        # Test empty batch operations
        result = await redis_service.set_multiple({})
        assert result == True
        
        results = await redis_service.get_multiple([])
        assert results == []
        
        print("✅ Error handling test passed")


async def run_redis_tests():
    """Run all Redis service tests"""
    print("🚀 Starting Redis Service End-to-End Tests\n")
    
    try:
        # Create test instance
        test_instance = TestRedisService()
        
        # Initialize Redis service
        redis_service = RedisService(redis_url="redis://localhost:6379/15")
        await redis_service.initialize()
        
        print("📋 Running Redis service tests...")
        
        # Run all tests
        await test_instance.test_redis_connection(redis_service)
        await test_instance.test_basic_cache_operations(redis_service)
        await test_instance.test_json_serialization(redis_service)
        await test_instance.test_batch_operations(redis_service)
        await test_instance.test_pattern_operations(redis_service)
        await test_instance.test_namespace_operations(redis_service)
        await test_instance.test_memory_and_info(redis_service)
        await test_instance.test_health_check(redis_service)
        await test_instance.test_error_handling(redis_service)
        
        # Cleanup
        await redis_service.redis_client.flushdb()
        await redis_service.close()
        
        print("\n🎉 All Redis service tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Redis service tests failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    result = asyncio.run(run_redis_tests())
    exit(0 if result else 1)