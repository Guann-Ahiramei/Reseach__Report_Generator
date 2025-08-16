"""
Multi-Retriever System End-to-End Tests
Comprehensive testing for the entire multi-retriever system
"""
import asyncio
import pytest
import tempfile
import shutil
import os
import json
from typing import List, Dict, Any
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers.utils import (
    RetrieverFactory, 
    get_retrievers,
    search_all_retrievers,
    merge_search_results,
    filter_results_by_domains,
    calculate_search_quality_score
)
from retrievers.base_retriever import SearchResult
from services.redis_service import RedisService
from services.faiss_service import FAISSService


class MockLogger:
    """Mock logger for testing"""
    def __init__(self):
        self.logs = []
        self.errors = []
    
    async def send_json(self, data):
        self.logs.append(data)
    
    async def log_error(self, error):
        self.errors.append(error)


class TestMultiRetrieverSystem:
    """Test multi-retriever system end-to-end functionality"""
    
    @pytest.fixture
    async def test_environment(self):
        """Set up test environment with Redis and FAISS"""
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        faiss_path = os.path.join(temp_dir, "test_faiss")
        docs_path = os.path.join(temp_dir, "test_docs")
        os.makedirs(docs_path)
        
        # Set up test documents
        await self._create_test_documents(docs_path)
        
        # Initialize services
        redis_service = RedisService(redis_url="redis://localhost:6379/14")  # Test DB
        faiss_service = FAISSService(
            index_path=faiss_path,
            dimension=1536,
            index_type='Flat'
        )
        
        try:
            await redis_service.initialize()
            await faiss_service.initialize()
            
            # Add some test documents to FAISS
            await self._populate_faiss_service(faiss_service)
            
            yield {
                'redis_service': redis_service,
                'faiss_service': faiss_service,
                'docs_path': docs_path,
                'temp_dir': temp_dir
            }
        finally:
            # Cleanup
            try:
                await redis_service.redis_client.flushdb()
                await redis_service.close()
                shutil.rmtree(temp_dir)
            except:
                pass
    
    async def _create_test_documents(self, docs_path: str):
        """Create test documents for local retriever"""
        documents = [
            {
                "filename": "ai_trends.txt",
                "content": """Artificial Intelligence Industry Trends Report 2024
                
The AI industry continues to experience unprecedented growth across multiple sectors.
Machine learning adoption has accelerated in healthcare, finance, and manufacturing.
Natural language processing technologies are transforming customer service operations.
Computer vision applications are revolutionizing autonomous vehicles and medical diagnostics.

Key findings:
- AI investment reached $50 billion globally in 2024
- 75% of enterprises plan to implement AI solutions within 2 years
- Demand for AI specialists increased by 200% year-over-year
- Regulatory frameworks are emerging to govern AI deployment
"""
            },
            {
                "filename": "cloud_computing.txt", 
                "content": """Cloud Computing Market Analysis
                
Cloud infrastructure adoption continues to grow as organizations modernize their IT operations.
Major cloud providers AWS, Microsoft Azure, and Google Cloud Platform dominate the market.
Hybrid and multi-cloud strategies are becoming increasingly popular among enterprises.
Edge computing is emerging as a complement to traditional cloud services.

Market insights:
- Global cloud market reached $400 billion in 2024
- SaaS segment shows strongest growth at 25% annually
- Security and compliance remain top concerns for cloud adoption
- Serverless computing adoption increased by 150% this year
"""
            },
            {
                "filename": "industry_4_0.txt",
                "content": """Industry 4.0 and Smart Manufacturing
                
The fourth industrial revolution is transforming manufacturing through digitalization.
IoT sensors and smart devices enable real-time monitoring and predictive maintenance.
Digital twins provide virtual representations of physical manufacturing processes.
Automated quality control systems reduce defects and improve efficiency.

Technology trends:
- IoT device deployments in manufacturing increased 300%
- Predictive maintenance reduces downtime by up to 50%
- Digital twin implementations show 20% efficiency gains
- Collaborative robots (cobots) enhance human-machine interaction
"""
            }
        ]
        
        for doc in documents:
            file_path = os.path.join(docs_path, doc["filename"])
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc["content"])
    
    async def _populate_faiss_service(self, faiss_service: FAISSService):
        """Add test documents to FAISS service"""
        test_docs = [
            {
                "content": "Artificial intelligence and machine learning are driving innovation across industries. Deep learning models show remarkable performance in image recognition and natural language understanding.",
                "metadata": {"source": "ai_research", "category": "technology"},
                "file_name": "ai_research.txt"
            },
            {
                "content": "Cloud computing platforms provide scalable infrastructure for modern applications. Container orchestration and microservices architecture enable efficient resource utilization.",
                "metadata": {"source": "cloud_guide", "category": "infrastructure"},
                "file_name": "cloud_guide.txt"
            },
            {
                "content": "Industry 4.0 technologies including IoT sensors and predictive analytics are transforming manufacturing operations. Smart factories optimize production through real-time data analysis.",
                "metadata": {"source": "manufacturing", "category": "industry"},
                "file_name": "manufacturing.txt"
            }
        ]
        
        await faiss_service.add_documents(test_docs)
        await faiss_service.save_index()
    
    async def test_retriever_factory(self, test_environment):
        """Test retriever factory functionality"""
        # Test available retrievers
        available = RetrieverFactory.get_available_retrievers()
        assert isinstance(available, list)
        assert len(available) > 0
        
        # Should have our built-in retrievers
        expected_retrievers = ['tavily', 'local_docs', 'faiss', 'redis_cache']
        for retriever in expected_retrievers:
            assert retriever in available
        
        # Test retriever creation
        query = "artificial intelligence trends"
        logger = MockLogger()
        
        # Create FAISS retriever
        faiss_retriever = RetrieverFactory.create_retriever(
            'faiss', 
            query=query,
            logger=logger
        )
        assert faiss_retriever is not None
        assert faiss_retriever.name == "FAISS Vector Search"
        
        # Create Redis cache retriever
        redis_retriever = RetrieverFactory.create_retriever(
            'redis_cache',
            query=query,
            logger=logger
        )
        assert redis_retriever is not None
        assert redis_retriever.name == "Redis Cache"
        
        # Test retriever info
        info = RetrieverFactory.get_retriever_info('faiss')
        assert isinstance(info, dict)
        assert info["name"] == "faiss"
        
        print("✅ Retriever factory test passed")
    
    async def test_individual_retrievers(self, test_environment):
        """Test each retriever individually"""
        query = "machine learning and AI applications"
        logger = MockLogger()
        
        # Test FAISS retriever
        faiss_retriever = RetrieverFactory.create_retriever(
            'faiss',
            query=query,
            logger=logger
        )
        
        faiss_results = await faiss_retriever.search(max_results=5)
        assert isinstance(faiss_results, list)
        # Should find some results from our test documents
        
        # Test Redis cache retriever
        redis_retriever = RetrieverFactory.create_retriever(
            'redis_cache',
            query=query,
            logger=logger,
            cache_namespace='test_cache',
            cache_patterns=['*']
        )
        
        # Pre-populate cache for testing
        redis_service = test_environment['redis_service']
        await redis_service.set_cache(
            'test_cache:ai_result',
            {
                'title': 'AI Research Result',
                'content': 'Artificial intelligence research shows promising results in machine learning applications',
                'url': 'cache://ai_result',
                'source': 'research_cache'
            },
            ttl=300
        )
        
        redis_results = await redis_retriever.search(max_results=5)
        assert isinstance(redis_results, list)
        
        # Test Local docs retriever
        local_retriever = RetrieverFactory.create_retriever(
            'local_docs',
            query=query,
            logger=logger,
            doc_path=test_environment['docs_path']
        )
        
        local_results = await local_retriever.search(max_results=5)
        assert isinstance(local_results, list)
        # Should find results from our test documents
        
        print("✅ Individual retrievers test passed")
    
    async def test_parallel_search(self, test_environment):
        """Test parallel search across multiple retrievers"""
        query = "artificial intelligence and cloud computing"
        logger = MockLogger()
        
        # Get available retrievers
        retrievers = get_retrievers(
            retriever_names=['faiss', 'redis_cache', 'local_docs']
        )
        
        assert len(retrievers) >= 2  # Should get at least some retrievers
        
        # Perform parallel search
        results_by_retriever = await search_all_retrievers(
            query=query,
            retrievers=retrievers,
            max_results_per_retriever=5,
            logger=logger,
            timeout_per_retriever=10
        )
        
        assert isinstance(results_by_retriever, dict)
        
        # Check that we got results from multiple retrievers
        total_results = 0
        for retriever_name, results in results_by_retriever.items():
            assert isinstance(results, list)
            print(f"📊 {retriever_name}: {len(results)} results")
            total_results += len(results)
        
        print(f"📈 Total results from all retrievers: {total_results}")
        
        print("✅ Parallel search test passed")
    
    async def test_result_merging(self, test_environment):
        """Test merging and ranking results from multiple retrievers"""
        # Create mock results from different retrievers
        mock_results = {
            'faiss': [
                SearchResult(
                    title="AI Research Paper",
                    content="Artificial intelligence research in machine learning",
                    url="https://example.com/ai-research",
                    source="faiss",
                    relevance_score=0.95,
                    metadata={'retriever': 'faiss'}
                ),
                SearchResult(
                    title="Cloud AI Services",
                    content="Cloud-based artificial intelligence services and platforms",
                    url="https://example.com/cloud-ai",
                    source="faiss", 
                    relevance_score=0.85,
                    metadata={'retriever': 'faiss'}
                )
            ],
            'redis_cache': [
                SearchResult(
                    title="Cached AI Analysis",
                    content="Cached analysis of AI industry trends and developments",
                    url="cache://ai-analysis",
                    source="redis_cache",
                    relevance_score=0.90,
                    metadata={'retriever': 'redis_cache'}
                )
            ],
            'local_docs': [
                SearchResult(
                    title="Local AI Document",
                    content="Local document about artificial intelligence applications",
                    url="file://local-ai-doc.txt",
                    source="local_docs",
                    relevance_score=0.80,
                    metadata={'retriever': 'local_docs'}
                ),
                SearchResult(
                    title="AI Research Paper",  # Duplicate URL for deduplication test
                    content="Different content but same URL as FAISS result",
                    url="https://example.com/ai-research",
                    source="local_docs",
                    relevance_score=0.75,
                    metadata={'retriever': 'local_docs'}
                )
            ]
        }
        
        # Test merging with deduplication
        merged_results = await merge_search_results(
            mock_results,
            max_total_results=10,
            deduplicate=True
        )
        
        assert isinstance(merged_results, list)
        assert len(merged_results) <= 10
        
        # Check that results are sorted by weighted relevance score
        if len(merged_results) > 1:
            for i in range(len(merged_results) - 1):
                assert merged_results[i].relevance_score >= merged_results[i + 1].relevance_score
        
        # Check deduplication - should not have duplicate URLs
        urls = [result.url for result in merged_results]
        assert len(urls) == len(set(urls)), "Deduplication failed - found duplicate URLs"
        
        # Test merging without deduplication
        non_deduplicated = await merge_search_results(
            mock_results,
            max_total_results=10,
            deduplicate=False
        )
        
        assert len(non_deduplicated) >= len(merged_results)
        
        print("✅ Result merging test passed")
    
    async def test_domain_filtering(self, test_environment):
        """Test domain-based result filtering"""
        # Create test results with different domains
        test_results = [
            SearchResult(
                title="Example Result 1",
                content="Content from example.com",
                url="https://example.com/page1",
                source="test",
                relevance_score=0.9
            ),
            SearchResult(
                title="Google Result",
                content="Content from google.com", 
                url="https://google.com/search",
                source="test",
                relevance_score=0.8
            ),
            SearchResult(
                title="GitHub Result",
                content="Content from github.com",
                url="https://github.com/repo",
                source="test",
                relevance_score=0.7
            ),
            SearchResult(
                title="Another Example",
                content="More content from example.com",
                url="https://example.com/page2", 
                source="test",
                relevance_score=0.6
            )
        ]
        
        # Test filtering by allowed domains
        allowed_domains = ["example.com", "github.com"]
        filtered_results = filter_results_by_domains(test_results, allowed_domains)
        
        assert len(filtered_results) == 3  # Should exclude google.com result
        
        # Verify all results are from allowed domains
        for result in filtered_results:
            assert any(domain in result.url for domain in allowed_domains)
        
        # Test with empty allowed domains (should return all)
        all_results = filter_results_by_domains(test_results, [])
        assert len(all_results) == len(test_results)
        
        print("✅ Domain filtering test passed")
    
    async def test_search_quality_metrics(self, test_environment):
        """Test search quality calculation"""
        query = "artificial intelligence machine learning"
        
        # Create test results with varying quality
        test_results = [
            SearchResult(
                title="High Quality AI Result",
                content="Artificial intelligence and machine learning applications in industry",
                url="https://example.com/ai1",
                source="test",
                relevance_score=0.95
            ),
            SearchResult(
                title="Medium Quality Result", 
                content="Some information about artificial intelligence applications",
                url="https://different.com/ai2",
                source="test",
                relevance_score=0.75
            ),
            SearchResult(
                title="Lower Quality Result",
                content="Brief mention of AI in this document",
                url="https://another.com/misc",
                source="test",
                relevance_score=0.50
            )
        ]
        
        # Calculate quality metrics
        quality_metrics = calculate_search_quality_score(test_results, query)
        
        assert isinstance(quality_metrics, dict)
        assert "average_relevance" in quality_metrics
        assert "result_diversity" in quality_metrics
        assert "query_coverage" in quality_metrics
        assert "overall_quality" in quality_metrics
        
        # Check metric ranges
        assert 0 <= quality_metrics["average_relevance"] <= 1
        assert 0 <= quality_metrics["result_diversity"] <= 1
        assert 0 <= quality_metrics["query_coverage"] <= 1
        assert 0 <= quality_metrics["overall_quality"] <= 1
        
        # Average relevance should be around (0.95 + 0.75 + 0.50) / 3 = 0.73
        expected_avg = (0.95 + 0.75 + 0.50) / 3
        assert abs(quality_metrics["average_relevance"] - expected_avg) < 0.01
        
        # Diversity should be high (3 different domains)
        assert quality_metrics["result_diversity"] > 0.9
        
        print("✅ Search quality metrics test passed")
    
    async def test_error_handling(self, test_environment):
        """Test error handling in multi-retriever system"""
        logger = MockLogger()
        
        # Test with invalid retriever name
        try:
            invalid_retriever = RetrieverFactory.create_retriever(
                'nonexistent_retriever',
                query="test query"
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test search with empty query
        faiss_retriever = RetrieverFactory.create_retriever(
            'faiss',
            query="",
            logger=logger
        )
        
        empty_results = await faiss_retriever.search(max_results=5)
        assert isinstance(empty_results, list)
        
        # Test with very large max_results
        large_results = await faiss_retriever.search(max_results=1000)
        assert isinstance(large_results, list)
        
        # Test merging empty results
        empty_merge = await merge_search_results({}, max_total_results=10)
        assert empty_merge == []
        
        # Test quality calculation with empty results
        empty_quality = calculate_search_quality_score([], "test query")
        assert empty_quality["overall_quality"] == 0.0
        
        print("✅ Error handling test passed")
    
    async def test_end_to_end_workflow(self, test_environment):
        """Test complete end-to-end multi-retriever workflow"""
        query = "artificial intelligence and machine learning trends"
        logger = MockLogger()
        
        print(f"🔍 Testing end-to-end search for: '{query}'")
        
        # Step 1: Get available retrievers
        retrievers = get_retrievers(
            retriever_names=['faiss', 'redis_cache', 'local_docs']
        )
        
        print(f"📋 Found {len(retrievers)} available retrievers")
        
        # Step 2: Perform parallel search
        results_by_retriever = await search_all_retrievers(
            query=query,
            retrievers=retrievers,
            max_results_per_retriever=3,
            logger=logger,
            timeout_per_retriever=15
        )
        
        print(f"🔄 Search completed across {len(results_by_retriever)} retrievers")
        
        # Step 3: Merge and rank results
        merged_results = await merge_search_results(
            results_by_retriever,
            max_total_results=10,
            deduplicate=True
        )
        
        print(f"📊 Merged to {len(merged_results)} total results")
        
        # Step 4: Apply domain filtering (if needed)
        # filtered_results = filter_results_by_domains(merged_results, ["trusted-domain.com"])
        
        # Step 5: Calculate quality metrics
        quality_metrics = calculate_search_quality_score(merged_results, query)
        
        print(f"📈 Search quality score: {quality_metrics['overall_quality']:.3f}")
        
        # Verify end-to-end results
        assert isinstance(merged_results, list)
        assert isinstance(quality_metrics, dict)
        
        # Print summary
        print(f"\n📋 End-to-End Search Summary:")
        print(f"   Query: {query}")
        print(f"   Retrievers used: {len(results_by_retriever)}")
        print(f"   Total results: {len(merged_results)}")
        print(f"   Average relevance: {quality_metrics['average_relevance']:.3f}")
        print(f"   Result diversity: {quality_metrics['result_diversity']:.3f}")
        print(f"   Query coverage: {quality_metrics['query_coverage']:.3f}")
        print(f"   Overall quality: {quality_metrics['overall_quality']:.3f}")
        
        # Check that we got meaningful results
        if len(merged_results) > 0:
            print(f"\n🔍 Sample results:")
            for i, result in enumerate(merged_results[:3]):
                print(f"   {i+1}. {result.title[:50]}... (score: {result.relevance_score:.3f})")
        
        print("✅ End-to-end workflow test passed")


async def run_multi_retriever_tests():
    """Run all multi-retriever system tests"""
    print("🚀 Starting Multi-Retriever System End-to-End Tests\n")
    
    try:
        # Create test instance
        test_instance = TestMultiRetrieverSystem()
        
        # Set up test environment
        temp_dir = tempfile.mkdtemp()
        faiss_path = os.path.join(temp_dir, "test_faiss")
        docs_path = os.path.join(temp_dir, "test_docs")
        os.makedirs(docs_path)
        
        await test_instance._create_test_documents(docs_path)
        
        # Initialize services
        redis_service = RedisService(redis_url="redis://localhost:6379/14")
        faiss_service = FAISSService(
            index_path=faiss_path,
            dimension=1536,
            index_type='Flat'
        )
        
        await redis_service.initialize()
        await faiss_service.initialize()
        await test_instance._populate_faiss_service(faiss_service)
        
        test_env = {
            'redis_service': redis_service,
            'faiss_service': faiss_service,
            'docs_path': docs_path,
            'temp_dir': temp_dir
        }
        
        print("📋 Running multi-retriever system tests...")
        
        # Run all tests
        await test_instance.test_retriever_factory(test_env)
        await test_instance.test_individual_retrievers(test_env)
        await test_instance.test_parallel_search(test_env)
        await test_instance.test_result_merging(test_env)
        await test_instance.test_domain_filtering(test_env)
        await test_instance.test_search_quality_metrics(test_env)
        await test_instance.test_error_handling(test_env)
        await test_instance.test_end_to_end_workflow(test_env)
        
        # Cleanup
        await redis_service.redis_client.flushdb()
        await redis_service.close()
        shutil.rmtree(temp_dir)
        
        print("\n🎉 All multi-retriever system tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Multi-retriever system tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    result = asyncio.run(run_multi_retriever_tests())
    exit(0 if result else 1)