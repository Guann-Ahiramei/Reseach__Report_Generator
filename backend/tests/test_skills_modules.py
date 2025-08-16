"""
Skills Modules End-to-End Tests
Comprehensive testing for all Skills modules (ContextManager, ResearchConductor, ReportGenerator)
"""
import asyncio
import pytest
import tempfile
import shutil
import os
import json
from typing import Dict, Any, List
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.context_manager import ContextManager
from skills.researcher import ResearchConductor  
from skills.writer import ReportGenerator
from skills.faiss_manager import FAISSManager
from skills.hybrid_searcher import HybridSearcher
from services.redis_service import RedisService
from services.faiss_service import FAISSService
from core.config import config


class MockLogger:
    """Mock logger for testing"""
    def __init__(self):
        self.logs = []
        self.errors = []
    
    async def send_json(self, data):
        self.logs.append(data)
    
    async def log_error(self, error):
        self.errors.append(error)


class TestSkillsModules:
    """Test Skills modules end-to-end functionality"""
    
    @pytest.fixture
    async def test_environment(self):
        """Set up test environment with services"""
        temp_dir = tempfile.mkdtemp()
        faiss_path = os.path.join(temp_dir, "test_faiss")
        docs_path = os.path.join(temp_dir, "test_docs")
        os.makedirs(docs_path)
        
        # Create test documents
        await self._create_test_documents(docs_path)
        
        # Initialize services
        redis_service = RedisService(redis_url="redis://localhost:6379/13")
        faiss_service = FAISSService(
            index_path=faiss_path,
            dimension=1536,
            index_type='Flat'
        )
        
        try:
            await redis_service.initialize()
            await faiss_service.initialize()
            
            # Populate services with test data
            await self._populate_services(redis_service, faiss_service)
            
            yield {
                'redis_service': redis_service,
                'faiss_service': faiss_service,
                'docs_path': docs_path,
                'temp_dir': temp_dir
            }
        finally:
            try:
                await redis_service.redis_client.flushdb()
                await redis_service.close()
                shutil.rmtree(temp_dir)
            except:
                pass
    
    async def _create_test_documents(self, docs_path: str):
        """Create test documents"""
        documents = [
            {
                "filename": "ai_industry_report.txt",
                "content": """Artificial Intelligence Industry Report 2024

Executive Summary:
The AI industry has experienced unprecedented growth, with global investments reaching $50 billion.
Machine learning applications are transforming healthcare, finance, and manufacturing sectors.
Natural language processing technologies are revolutionizing customer service operations.

Key Market Trends:
1. Enterprise AI adoption increased by 200% year-over-year
2. Demand for AI specialists continues to outpace supply
3. Regulatory frameworks are emerging to govern AI deployment
4. Edge AI computing is gaining momentum for real-time applications

Industry Challenges:
- Data privacy and security concerns
- Lack of skilled AI professionals
- Integration complexity with legacy systems
- Ethical considerations in AI decision-making

Future Outlook:
AI will continue to drive innovation across industries, with particular growth expected in:
- Autonomous vehicles and transportation
- Personalized healthcare and drug discovery
- Smart manufacturing and Industry 4.0
- Financial services automation
"""
            },
            {
                "filename": "technology_trends.txt",
                "content": """Technology Trends Analysis

Cloud Computing Evolution:
Hybrid and multi-cloud strategies are becoming the norm for enterprises.
Serverless computing adoption has increased by 150% this year.
Edge computing complements traditional cloud services for latency-sensitive applications.

Cybersecurity Landscape:
Zero-trust security models are being widely adopted.
AI-powered threat detection systems show 80% improvement in accuracy.
Ransomware attacks have increased, driving cybersecurity investments.

Emerging Technologies:
- Quantum computing research shows promising breakthroughs
- 5G networks enable new IoT and AR/VR applications  
- Blockchain technology finds new use cases beyond cryptocurrency
- Extended reality (XR) transforms training and collaboration

Digital Transformation:
Organizations are accelerating digital initiatives post-pandemic.
Remote work technologies have become permanent fixtures.
Data analytics and business intelligence drive decision-making.
"""
            }
        ]
        
        for doc in documents:
            file_path = os.path.join(docs_path, doc["filename"])
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc["content"])
    
    async def _populate_services(self, redis_service: RedisService, faiss_service: FAISSService):
        """Populate services with test data"""
        # Add documents to FAISS
        test_docs = [
            {
                "content": "Artificial intelligence and machine learning technologies are driving digital transformation across industries. Deep learning models demonstrate exceptional performance in computer vision and natural language processing tasks.",
                "metadata": {"source": "ai_research", "category": "technology", "date": "2024-01-15"},
                "file_name": "ai_research.txt"
            },
            {
                "content": "Cloud computing platforms provide scalable infrastructure for modern applications. Container orchestration and microservices architecture enable efficient resource utilization and deployment.",
                "metadata": {"source": "cloud_guide", "category": "infrastructure", "date": "2024-01-20"},
                "file_name": "cloud_guide.txt"
            },
            {
                "content": "Industry 4.0 represents the fourth industrial revolution, characterized by smart manufacturing, IoT integration, and predictive analytics. Real-time data collection enables proactive maintenance and optimization.",
                "metadata": {"source": "industry40", "category": "manufacturing", "date": "2024-01-25"},
                "file_name": "industry40.txt"
            }
        ]
        
        await faiss_service.add_documents(test_docs)
        await faiss_service.save_index()
        
        # Add cache data to Redis
        cache_data = {
            'research_cache:ai_trends': {
                'query': 'artificial intelligence trends',
                'results': [
                    {
                        'title': 'AI Market Growth Report',
                        'content': 'The AI market is projected to reach $190 billion by 2025',
                        'source': 'market_research',
                        'score': 0.95
                    }
                ],
                'timestamp': datetime.now().isoformat()
            },
            'research_cache:technology_analysis': {
                'query': 'technology trends analysis',
                'results': [
                    {
                        'title': 'Tech Trends 2024',
                        'content': 'Key technology trends include AI, cloud computing, and cybersecurity',
                        'source': 'tech_analysis',
                        'score': 0.90
                    }
                ],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        for key, data in cache_data.items():
            await redis_service.set_cache(key, data, ttl=3600)
    
    async def test_context_manager(self, test_environment):
        """Test ContextManager functionality"""
        logger = MockLogger()
        
        # Initialize ContextManager
        context_manager = ContextManager(
            query="artificial intelligence market trends",
            faiss_service=test_environment['faiss_service'],
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        # Test context initialization
        await context_manager.initialize()
        assert context_manager.is_initialized
        
        # Test context building
        context_data = await context_manager.build_context()
        
        assert isinstance(context_data, dict)
        assert "query" in context_data
        assert "context_sources" in context_data
        assert "semantic_context" in context_data
        assert "cached_insights" in context_data
        assert "context_score" in context_data
        
        # Verify context quality
        assert context_data["context_score"] > 0
        assert len(context_data["context_sources"]) > 0
        
        # Test context caching
        cache_key = await context_manager.cache_context(context_data)
        assert cache_key is not None
        
        # Retrieve cached context
        cached_context = await context_manager.get_cached_context(cache_key)
        assert cached_context is not None
        assert cached_context["query"] == context_data["query"]
        
        # Test context enrichment
        enriched_context = await context_manager.enrich_context(
            context_data, 
            additional_sources=["additional source data"]
        )
        assert enriched_context["context_score"] >= context_data["context_score"]
        
        print("✅ ContextManager test passed")
    
    async def test_research_conductor(self, test_environment):
        """Test ResearchConductor functionality"""
        logger = MockLogger()
        
        # Initialize ResearchConductor
        researcher = ResearchConductor(
            query="AI industry analysis and market trends",
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        # Test research planning
        research_plan = await researcher.plan_research()
        
        assert isinstance(research_plan, dict)
        assert "research_objectives" in research_plan
        assert "search_strategies" in research_plan
        assert "data_sources" in research_plan
        assert "timeline" in research_plan
        
        # Test research execution
        research_results = await researcher.conduct_research()
        
        assert isinstance(research_results, dict)
        assert "findings" in research_results
        assert "sources" in research_results
        assert "confidence_score" in research_results
        assert "research_metadata" in research_results
        
        # Verify research quality
        assert research_results["confidence_score"] > 0
        assert len(research_results["findings"]) > 0
        assert len(research_results["sources"]) > 0
        
        # Test research caching
        cache_key = await researcher.cache_research_results(research_results)
        assert cache_key is not None
        
        # Test research analysis
        analysis = await researcher.analyze_findings(research_results["findings"])
        
        assert isinstance(analysis, dict)
        assert "key_insights" in analysis
        assert "trends" in analysis
        assert "recommendations" in analysis
        
        # Test performance metrics
        metrics = await researcher.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "research_count" in metrics
        assert "cache_hit_rate" in metrics
        assert "average_confidence" in metrics
        
        print("✅ ResearchConductor test passed")
    
    async def test_report_generator(self, test_environment):
        """Test ReportGenerator functionality"""
        logger = MockLogger()
        
        # Prepare test data
        research_data = {
            "query": "AI industry market analysis",
            "findings": [
                {
                    "title": "AI Market Growth",
                    "content": "The AI market is experiencing rapid growth with investments reaching $50 billion globally",
                    "source": "market_research",
                    "confidence": 0.95
                },
                {
                    "title": "Technology Adoption",
                    "content": "Enterprise AI adoption has increased by 200% year-over-year across industries",
                    "source": "industry_report",
                    "confidence": 0.90
                },
                {
                    "title": "Future Projections",
                    "content": "AI market is projected to reach $190 billion by 2025 driven by automation and data analytics",
                    "source": "analyst_forecast",
                    "confidence": 0.85
                }
            ],
            "context": "Industry analysis focusing on AI market trends, adoption rates, and future projections",
            "metadata": {
                "research_date": datetime.now().isoformat(),
                "data_sources": 3,
                "confidence_score": 0.90
            }
        }
        
        # Initialize ReportGenerator
        writer = ReportGenerator(
            query=research_data["query"],
            research_data=research_data,
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        # Test report planning
        report_plan = await writer.plan_report()
        
        assert isinstance(report_plan, dict)
        assert "report_structure" in report_plan
        assert "key_sections" in report_plan
        assert "writing_strategy" in report_plan
        
        # Test report generation
        generated_report = await writer.generate_report()
        
        assert isinstance(generated_report, dict)
        assert "title" in generated_report
        assert "executive_summary" in generated_report
        assert "main_content" in generated_report
        assert "conclusions" in generated_report
        assert "sources" in generated_report
        assert "metadata" in generated_report
        
        # Verify report quality
        assert len(generated_report["title"]) > 0
        assert len(generated_report["executive_summary"]) > 100  # Substantial summary
        assert len(generated_report["main_content"]) > 500  # Detailed content
        assert len(generated_report["sources"]) > 0
        
        # Test report caching
        cache_key = await writer.cache_report(generated_report)
        assert cache_key is not None
        
        # Test report formatting
        formatted_report = await writer.format_report(
            generated_report, 
            format_type="markdown"
        )
        
        assert isinstance(formatted_report, str)
        assert "# " in formatted_report  # Markdown headers
        assert len(formatted_report) > len(generated_report["main_content"])
        
        # Test report export
        export_data = await writer.export_report(
            generated_report,
            export_format="json"
        )
        
        assert isinstance(export_data, str)
        exported_dict = json.loads(export_data)
        assert exported_dict["title"] == generated_report["title"]
        
        print("✅ ReportGenerator test passed")
    
    async def test_faiss_manager(self, test_environment):
        """Test FAISSManager skill functionality"""
        logger = MockLogger()
        
        # Initialize FAISSManager
        faiss_manager = FAISSManager(
            faiss_service=test_environment['faiss_service'],
            logger=logger
        )
        
        # Test vector operations
        query_vector = "artificial intelligence and machine learning applications"
        
        # Test similarity search
        similar_docs = await faiss_manager.find_similar_documents(
            query_vector, 
            k=3
        )
        
        assert isinstance(similar_docs, list)
        assert len(similar_docs) <= 3
        
        for doc in similar_docs:
            assert "content" in doc
            assert "score" in doc
            assert "metadata" in doc
        
        # Test document clustering
        cluster_results = await faiss_manager.cluster_documents(
            num_clusters=2
        )
        
        assert isinstance(cluster_results, dict)
        assert "clusters" in cluster_results
        assert "cluster_centers" in cluster_results
        
        # Test semantic gap analysis
        gap_analysis = await faiss_manager.analyze_semantic_gaps(
            query_vector
        )
        
        assert isinstance(gap_analysis, dict)
        assert "coverage_score" in gap_analysis
        assert "gap_areas" in gap_analysis
        assert "recommendations" in gap_analysis
        
        # Test knowledge expansion
        expansion_suggestions = await faiss_manager.suggest_knowledge_expansion(
            current_topics=["AI", "machine learning"]
        )
        
        assert isinstance(expansion_suggestions, list)
        
        print("✅ FAISSManager test passed")
    
    async def test_hybrid_searcher(self, test_environment):
        """Test HybridSearcher skill functionality"""
        logger = MockLogger()
        
        # Initialize HybridSearcher
        hybrid_searcher = HybridSearcher(
            query="artificial intelligence market trends and analysis",
            faiss_service=test_environment['faiss_service'],
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        # Test search strategy planning
        search_strategy = await hybrid_searcher.plan_search_strategy()
        
        assert isinstance(search_strategy, dict)
        assert "search_methods" in search_strategy
        assert "query_expansions" in search_strategy
        assert "confidence_threshold" in search_strategy
        
        # Test adaptive search execution
        search_results = await hybrid_searcher.execute_adaptive_search()
        
        assert isinstance(search_results, dict)
        assert "primary_results" in search_results
        assert "supplementary_results" in search_results
        assert "search_confidence" in search_results
        assert "strategy_used" in search_results
        
        # Test query expansion
        expanded_queries = await hybrid_searcher.expand_query()
        
        assert isinstance(expanded_queries, list)
        assert len(expanded_queries) > 0
        
        # Test result synthesis
        synthesized_results = await hybrid_searcher.synthesize_results(
            search_results["primary_results"]
        )
        
        assert isinstance(synthesized_results, dict)
        assert "synthesized_findings" in synthesized_results
        assert "confidence_score" in synthesized_results
        assert "source_diversity" in synthesized_results
        
        print("✅ HybridSearcher test passed")
    
    async def test_skills_integration(self, test_environment):
        """Test integration between different Skills modules"""
        logger = MockLogger()
        query = "AI industry transformation and future outlook"
        
        # Step 1: Use ContextManager to build context
        context_manager = ContextManager(
            query=query,
            faiss_service=test_environment['faiss_service'],
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        await context_manager.initialize()
        context_data = await context_manager.build_context()
        
        # Step 2: Use ResearchConductor to conduct research
        researcher = ResearchConductor(
            query=query,
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        research_results = await researcher.conduct_research()
        
        # Step 3: Use HybridSearcher for enhanced search
        hybrid_searcher = HybridSearcher(
            query=query,
            faiss_service=test_environment['faiss_service'],
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        hybrid_results = await hybrid_searcher.execute_adaptive_search()
        
        # Step 4: Combine data for report generation
        combined_data = {
            "query": query,
            "context": context_data,
            "research_findings": research_results["findings"],
            "hybrid_search_results": hybrid_results["primary_results"],
            "metadata": {
                "context_score": context_data["context_score"],
                "research_confidence": research_results["confidence_score"], 
                "search_confidence": hybrid_results["search_confidence"]
            }
        }
        
        # Step 5: Use ReportGenerator to create final report
        writer = ReportGenerator(
            query=query,
            research_data=combined_data,
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        final_report = await writer.generate_report()
        
        # Verify integration results
        assert isinstance(final_report, dict)
        assert len(final_report["main_content"]) > 1000  # Substantial integrated content
        assert final_report["metadata"]["data_sources"] >= 3  # Multiple data sources
        
        # Test end-to-end workflow metrics
        workflow_metrics = {
            "context_quality": context_data["context_score"],
            "research_confidence": research_results["confidence_score"],
            "search_effectiveness": hybrid_results["search_confidence"],
            "report_completeness": len(final_report["main_content"]) / 1000,  # Normalize
            "integration_success": True
        }
        
        overall_score = sum(workflow_metrics.values()) / len(workflow_metrics)
        
        print(f"\n📊 Skills Integration Metrics:")
        print(f"   Context Quality: {workflow_metrics['context_quality']:.3f}")
        print(f"   Research Confidence: {workflow_metrics['research_confidence']:.3f}")
        print(f"   Search Effectiveness: {workflow_metrics['search_effectiveness']:.3f}")
        print(f"   Report Completeness: {workflow_metrics['report_completeness']:.3f}")
        print(f"   Overall Score: {overall_score:.3f}")
        
        assert overall_score > 0.5  # Reasonable integration quality
        
        print("✅ Skills integration test passed")
    
    async def test_error_handling_and_resilience(self, test_environment):
        """Test error handling across Skills modules"""
        logger = MockLogger()
        
        # Test ContextManager with invalid query
        context_manager = ContextManager(
            query="",  # Empty query
            faiss_service=test_environment['faiss_service'],
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        await context_manager.initialize()
        empty_context = await context_manager.build_context()
        assert isinstance(empty_context, dict)  # Should handle gracefully
        
        # Test ResearchConductor with no results
        researcher = ResearchConductor(
            query="extremely_specific_nonexistent_topic_12345",
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        no_results_research = await researcher.conduct_research()
        assert isinstance(no_results_research, dict)
        assert "findings" in no_results_research
        
        # Test ReportGenerator with minimal data
        minimal_data = {
            "query": "test query",
            "findings": [],
            "context": "minimal context",
            "metadata": {}
        }
        
        writer = ReportGenerator(
            query="test query",
            research_data=minimal_data,
            redis_service=test_environment['redis_service'],
            logger=logger
        )
        
        minimal_report = await writer.generate_report()
        assert isinstance(minimal_report, dict)
        assert len(minimal_report["main_content"]) > 0  # Should generate something
        
        print("✅ Error handling and resilience test passed")


async def run_skills_tests():
    """Run all Skills modules tests"""
    print("🚀 Starting Skills Modules End-to-End Tests\n")
    
    try:
        # Create test instance
        test_instance = TestSkillsModules()
        
        # Set up test environment
        temp_dir = tempfile.mkdtemp()
        faiss_path = os.path.join(temp_dir, "test_faiss")
        docs_path = os.path.join(temp_dir, "test_docs")
        os.makedirs(docs_path)
        
        await test_instance._create_test_documents(docs_path)
        
        # Initialize services
        redis_service = RedisService(redis_url="redis://localhost:6379/13")
        faiss_service = FAISSService(
            index_path=faiss_path,
            dimension=1536,
            index_type='Flat'
        )
        
        await redis_service.initialize()
        await faiss_service.initialize()
        await test_instance._populate_services(redis_service, faiss_service)
        
        test_env = {
            'redis_service': redis_service,
            'faiss_service': faiss_service,
            'docs_path': docs_path,
            'temp_dir': temp_dir
        }
        
        print("📋 Running Skills modules tests...")
        
        # Run all tests
        await test_instance.test_context_manager(test_env)
        await test_instance.test_research_conductor(test_env)
        await test_instance.test_report_generator(test_env)
        await test_instance.test_faiss_manager(test_env)
        await test_instance.test_hybrid_searcher(test_env)
        await test_instance.test_skills_integration(test_env)
        await test_instance.test_error_handling_and_resilience(test_env)
        
        # Cleanup
        await redis_service.redis_client.flushdb()
        await redis_service.close()
        shutil.rmtree(temp_dir)
        
        print("\n🎉 All Skills modules tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Skills modules tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    result = asyncio.run(run_skills_tests())
    exit(0 if result else 1)