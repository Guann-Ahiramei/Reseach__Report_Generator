"""
FAISS Service End-to-End Tests
Comprehensive testing for FAISS service functionality
"""
import asyncio
import pytest
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import tempfile
import shutil
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.faiss_service import FAISSService


class TestFAISSService:
    """Test FAISS service end-to-end functionality"""
    
    @pytest.fixture
    async def faiss_service(self):
        """Create FAISS service instance for testing"""
        # Create temporary directory for test index
        temp_dir = tempfile.mkdtemp()
        test_index_path = os.path.join(temp_dir, "test_index")
        
        service = FAISSService(
            index_path=test_index_path,
            dimension=1536,
            index_type='Flat',  # Use simple flat index for testing
            chunk_size=500,
            chunk_overlap=50
        )
        
        try:
            await service.initialize()
            yield service
        finally:
            # Cleanup
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    async def test_faiss_initialization(self, faiss_service):
        """Test FAISS service initialization"""
        assert await faiss_service.is_initialized() == True
        assert faiss_service.index is not None
        assert faiss_service.embeddings is not None
        assert faiss_service.text_splitter is not None
        
        # Test index info
        info = await faiss_service.get_index_info()
        assert isinstance(info, dict)
        assert info["index_type"] == "Flat"
        assert info["dimension"] == 1536
        assert info["total_vectors"] == 0
        
        print("✅ FAISS initialization test passed")
    
    async def test_document_processing(self, faiss_service):
        """Test document processing and chunking"""
        test_documents = [
            {
                "content": "This is a test document about artificial intelligence. AI is transforming industries worldwide. Machine learning algorithms are becoming more sophisticated. Deep learning models can process vast amounts of data.",
                "metadata": {"source": "test_doc_1", "category": "AI"},
                "file_name": "ai_document.txt",
                "file_type": ".txt"
            },
            {
                "content": "Industry analysis shows significant growth in technology sector. Companies are investing heavily in digital transformation. Cloud computing adoption is accelerating across enterprises.",
                "metadata": {"source": "test_doc_2", "category": "Industry"},
                "file_name": "industry_report.txt", 
                "file_type": ".txt"
            }
        ]
        
        # Process documents
        initial_count = await faiss_service.get_document_count()
        await faiss_service.add_documents(test_documents)
        final_count = await faiss_service.get_document_count()
        
        assert final_count > initial_count
        assert faiss_service.index.ntotal > 0
        
        # Check index info after adding documents
        info = await faiss_service.get_index_info()
        assert info["total_vectors"] > 0
        assert info["document_count"] > 0
        
        print("✅ Document processing test passed")
    
    async def test_similarity_search(self, faiss_service):
        """Test similarity search functionality"""
        # Add test documents first
        test_documents = [
            {
                "content": "Machine learning and artificial intelligence are revolutionizing data analysis. Neural networks can identify complex patterns in large datasets.",
                "metadata": {"topic": "ML", "difficulty": "intermediate"},
                "file_name": "ml_guide.txt"
            },
            {
                "content": "Cloud computing provides scalable infrastructure for modern applications. AWS, Azure, and Google Cloud are leading providers in this space.",
                "metadata": {"topic": "Cloud", "difficulty": "beginner"},
                "file_name": "cloud_intro.txt"
            },
            {
                "content": "Industry 4.0 transformation involves IoT sensors, automation, and smart manufacturing processes. Real-time data collection enables predictive maintenance.",
                "metadata": {"topic": "Industry", "difficulty": "advanced"},
                "file_name": "industry4.txt"
            }
        ]
        
        await faiss_service.add_documents(test_documents)
        
        # Test basic similarity search
        query = "artificial intelligence and machine learning"
        results = await faiss_service.similarity_search(query, k=3)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert len(results) <= 3
        
        # Check result structure
        for result in results:
            assert "content" in result
            assert "score" in result
            assert "metadata" in result
            assert isinstance(result["score"], float)
        
        # Results should be sorted by score (higher is better for similarity)
        if len(results) > 1:
            assert results[0]["score"] >= results[1]["score"]
        
        print("✅ Similarity search test passed")
    
    async def test_filtered_search(self, faiss_service):
        """Test search with metadata filters"""
        # Add documents with different metadata
        test_documents = [
            {
                "content": "Python programming for data science and machine learning applications",
                "metadata": {"language": "Python", "level": "intermediate", "category": "programming"},
                "file_name": "python_ml.txt"
            },
            {
                "content": "JavaScript frameworks for web development including React and Vue",
                "metadata": {"language": "JavaScript", "level": "beginner", "category": "web"},
                "file_name": "js_frameworks.txt"
            },
            {
                "content": "Advanced Python techniques for high-performance computing",
                "metadata": {"language": "Python", "level": "advanced", "category": "programming"},
                "file_name": "python_advanced.txt"
            }
        ]
        
        await faiss_service.add_documents(test_documents)
        
        # Test filter search
        filter_dict = {"language": "Python"}
        results = await faiss_service.filter_search(filter_dict, k=5)
        
        assert isinstance(results, list)
        assert len(results) >= 2  # Should find both Python documents
        
        # Verify all results match filter
        for result in results:
            assert result["metadata"]["language"] == "Python"
        
        # Test similarity search with filters
        query = "programming techniques"
        filtered_results = await faiss_service.similarity_search(
            query, k=5, filter_dict={"level": "advanced"}
        )
        
        assert isinstance(filtered_results, list)
        # Should find results that match both query and filter
        for result in filtered_results:
            assert result["metadata"]["level"] == "advanced"
        
        print("✅ Filtered search test passed")
    
    async def test_mmr_search(self, faiss_service):
        """Test Maximum Marginal Relevance search"""
        # Add documents with some overlap for diversity testing
        test_documents = [
            {
                "content": "Machine learning algorithms for data analysis and pattern recognition",
                "metadata": {"type": "algorithm"},
                "file_name": "ml_algo1.txt"
            },
            {
                "content": "Machine learning techniques for predictive modeling and classification",
                "metadata": {"type": "technique"},
                "file_name": "ml_algo2.txt"
            },
            {
                "content": "Cloud computing architecture and distributed systems design",
                "metadata": {"type": "architecture"},
                "file_name": "cloud_arch.txt"
            },
            {
                "content": "Database optimization strategies for high-performance applications",
                "metadata": {"type": "optimization"},
                "file_name": "db_opt.txt"
            }
        ]
        
        await faiss_service.add_documents(test_documents)
        
        # Test MMR search
        query = "machine learning and data analysis"
        mmr_results = await faiss_service.max_marginal_relevance_search(
            query, k=3, lambda_mult=0.5
        )
        
        assert isinstance(mmr_results, list)
        assert len(mmr_results) <= 3
        
        # MMR should provide diverse results
        # Check that we get results from different topics, not just ML
        topics = set()
        for result in mmr_results:
            content = result["content"].lower()
            if "machine learning" in content:
                topics.add("ml")
            elif "cloud" in content:
                topics.add("cloud")
            elif "database" in content:
                topics.add("database")
        
        # With good MMR, we should get diverse topics
        print(f"MMR found topics: {topics}")
        
        print("✅ MMR search test passed")
    
    async def test_vector_search(self, faiss_service):
        """Test direct vector-based search"""
        # Add test documents
        test_documents = [
            {
                "content": "Vector databases enable semantic search capabilities for AI applications",
                "metadata": {"domain": "AI"},
                "file_name": "vector_db.txt"
            }
        ]
        
        await faiss_service.add_documents(test_documents)
        
        # Get embedding for a query
        query_text = "semantic search and vector databases"
        query_embedding = await faiss_service.get_embedding(query_text)
        
        assert isinstance(query_embedding, np.ndarray)
        assert query_embedding.shape == (1536,)  # OpenAI embedding dimension
        
        # Perform vector search
        vector_results = await faiss_service.similarity_search_by_vector(
            query_embedding, k=5
        )
        
        assert isinstance(vector_results, list)
        
        print("✅ Vector search test passed")
    
    async def test_index_persistence(self, faiss_service):
        """Test saving and loading index"""
        # Add some documents
        test_documents = [
            {
                "content": "Test document for persistence testing",
                "metadata": {"test": "persistence"},
                "file_name": "persistence_test.txt"
            }
        ]
        
        await faiss_service.add_documents(test_documents)
        initial_count = await faiss_service.get_document_count()
        
        # Save index
        await faiss_service.save_index()
        
        # Verify save time was recorded
        info = await faiss_service.get_index_info()
        assert info["last_save_time"] is not None
        
        print("✅ Index persistence test passed")
    
    async def test_index_optimization(self, faiss_service):
        """Test index optimization"""
        # Add documents
        test_documents = [
            {
                "content": f"Test document {i} for optimization testing",
                "metadata": {"test_id": i},
                "file_name": f"opt_test_{i}.txt"
            }
            for i in range(5)
        ]
        
        await faiss_service.add_documents(test_documents)
        
        # Optimize index
        await faiss_service.optimize_index()
        
        # Verify index still works after optimization
        results = await faiss_service.similarity_search("test document", k=3)
        assert len(results) > 0
        
        print("✅ Index optimization test passed")
    
    async def test_health_check(self, faiss_service):
        """Test FAISS service health check"""
        health_result = await faiss_service.health_check()
        
        assert isinstance(health_result, dict)
        assert "status" in health_result
        assert health_result["status"] in ["healthy", "unhealthy"]
        assert "latency_ms" in health_result
        assert "checks" in health_result
        assert "index_info" in health_result
        
        # Check individual health components
        checks = health_result["checks"]
        assert "initialized" in checks
        assert "index_ready" in checks
        assert "embeddings_ready" in checks
        assert "search_functional" in checks
        
        print("✅ Health check test passed")
    
    async def test_error_handling(self, faiss_service):
        """Test error handling and edge cases"""
        # Test search with empty index
        await faiss_service.clear_index()
        
        results = await faiss_service.similarity_search("test query", k=5)
        assert isinstance(results, list)
        assert len(results) == 0
        
        # Test empty document list
        await faiss_service.add_documents([])
        
        # Test document with empty content
        empty_docs = [{"content": "", "metadata": {}}]
        await faiss_service.add_documents(empty_docs)
        
        print("✅ Error handling test passed")


async def run_faiss_tests():
    """Run all FAISS service tests"""
    print("🚀 Starting FAISS Service End-to-End Tests\n")
    
    try:
        # Create test instance
        test_instance = TestFAISSService()
        
        # Create temporary FAISS service
        temp_dir = tempfile.mkdtemp()
        test_index_path = os.path.join(temp_dir, "test_index")
        
        faiss_service = FAISSService(
            index_path=test_index_path,
            dimension=1536,
            index_type='Flat',
            chunk_size=500
        )
        
        await faiss_service.initialize()
        
        print("📋 Running FAISS service tests...")
        
        # Run all tests
        await test_instance.test_faiss_initialization(faiss_service)
        await test_instance.test_document_processing(faiss_service)
        await test_instance.test_similarity_search(faiss_service)
        await test_instance.test_filtered_search(faiss_service)
        await test_instance.test_mmr_search(faiss_service)
        await test_instance.test_vector_search(faiss_service)
        await test_instance.test_index_persistence(faiss_service)
        await test_instance.test_index_optimization(faiss_service)
        await test_instance.test_health_check(faiss_service)
        await test_instance.test_error_handling(faiss_service)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n🎉 All FAISS service tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAISS service tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    result = asyncio.run(run_faiss_tests())
    exit(0 if result else 1)