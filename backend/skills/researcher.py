"""
Enhanced Research Conductor for Industry Reporter 2
Based on GPT-Researcher's researcher.py with modern enhancements
"""
import asyncio
import random
import logging
import time
from typing import List, Dict, Optional, Any, Set
from datetime import datetime

from core.config import config
from core.logging import CustomLogsHandler
from services.redis_service import RedisService
from services.faiss_service import FAISSService
from utils.document_loader import DocumentLoader
from .context_manager import ContextManager


class ResearchConductor:
    """Enhanced research conductor with Redis caching and FAISS integration"""
    
    def __init__(self, researcher):
        self.researcher = researcher
        self.redis_service = RedisService()
        self.faiss_service = FAISSService()
        self.logger = getattr(researcher, 'logger', None)
        
        # Performance tracking
        self._research_start_time = None
        self._step_times = {}
        
        # Cache for research results
        self._search_cache = {}
        self._mcp_results_cache = None
        self._mcp_query_count = 0
    
    async def plan_research(self, query: str, query_domains: List[str] = None) -> List[str]:
        """
        Enhanced research planning with caching
        """
        if query_domains is None:
            query_domains = []
        
        self._step_times["planning_start"] = time.time()
        
        # Check cache for research plan
        cache_key = f"research_plan:{hash(query)}:{hash(str(query_domains))}"
        cached_plan = await self.redis_service.get_cached_context(cache_key)
        
        if cached_plan and isinstance(cached_plan, list):
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📋 Using cached research plan for: {query}",
                    "cache_hit": True
                })
            return cached_plan
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🌐 Browsing the web to learn more about the task: {query}...",
            })
        
        # Get initial search results for planning
        search_results = await self._get_initial_search_results(query, query_domains)
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🤔 Planning the research strategy and subtasks...",
            })
            await self.logger.log_search_results("planning", len(search_results), query)
        
        # Generate research outline/sub-queries
        outline = await self._plan_research_outline(
            query=query,
            search_results=search_results,
            query_domains=query_domains
        )
        
        # Cache the research plan
        if outline:
            await self.redis_service.cache_context(
                cache_key, 
                outline, 
                ttl=config.settings.cache_ttl_seconds
            )
        
        self._step_times["planning_end"] = time.time()
        planning_time = self._step_times["planning_end"] - self._step_times["planning_start"]
        
        if self.logger:
            await self.logger.log_performance_metric("planning_time", planning_time, "seconds")
            await self.logger.send_json({
                "type": "logs",
                "content": f"📋 Research plan completed with {len(outline)} sub-queries",
                "sub_queries": outline,
                "planning_time": planning_time
            })
        
        return outline
    
    async def conduct_research(self) -> List[str]:
        """
        Enhanced research conductor with comprehensive caching and performance tracking
        """
        self._research_start_time = time.time()
        
        if self.logger:
            await self.logger.log_research_start(
                query=self.researcher.query,
                config_info={
                    "retrievers": [r.__name__ for r in self.researcher.retrievers] if hasattr(self.researcher, 'retrievers') else [],
                    "cache_enabled": True,
                    "faiss_enabled": True
                }
            )
        
        # Reset visited URLs and initialize research data
        if hasattr(self.researcher, 'visited_urls'):
            self.researcher.visited_urls.clear()
        research_data = []
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔍 Starting the research task for '{self.researcher.query}'...",
            })
            
            if hasattr(self.researcher, 'agent'):
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🤖 Using agent: {self.researcher.agent}",
                    "agent": self.researcher.agent
                })
        
        # Determine research approach based on configuration
        research_data = await self._execute_research_strategy()
        
        # Store context and apply curation if enabled
        self.researcher.context = research_data
        
        if config.settings.curate_sources and hasattr(self.researcher, 'source_curator'):
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": "🎯 Curating sources for quality and relevance...",
                })
            
            curated_context = await self.researcher.source_curator.curate_sources(research_data)
            self.researcher.context = curated_context
        
        # Calculate total research time and costs
        total_time = time.time() - self._research_start_time
        total_costs = getattr(self.researcher, 'research_costs', 0.0)
        
        if self.logger:
            await self.logger.log_performance_metric("total_research_time", total_time, "seconds")
            await self.logger.send_json({
                "type": "logs",
                "content": f"✅ Research completed in {total_time:.2f}s\n💸 Total Research Costs: ${total_costs:.4f}",
                "total_time": total_time,
                "total_costs": total_costs,
                "context_size": len(str(self.researcher.context))
            })
        
        return self.researcher.context
    
    async def _execute_research_strategy(self) -> Any:
        """Execute the appropriate research strategy based on configuration"""
        
        # Check if we have specific source URLs
        if hasattr(self.researcher, 'source_urls') and self.researcher.source_urls:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": "🔗 Using provided source URLs",
                })
            
            research_data = await self._get_context_by_urls(self.researcher.source_urls)
            
            # Complement with web search if requested
            if hasattr(self.researcher, 'complement_source_urls') and self.researcher.complement_source_urls:
                additional_research = await self._get_context_by_web_search(
                    self.researcher.query, 
                    [], 
                    getattr(self.researcher, 'query_domains', [])
                )
                research_data += ' '.join(additional_research) if additional_research else ''
            
            return research_data
        
        # Determine report source type
        report_source = getattr(self.researcher, 'report_source', 'web')
        
        if report_source == 'web':
            return await self._get_context_by_web_search(
                self.researcher.query, 
                [], 
                getattr(self.researcher, 'query_domains', [])
            )
        
        elif report_source == 'local':
            return await self._get_context_by_local_documents()
        
        elif report_source == 'hybrid':
            return await self._get_context_by_hybrid_search()
        
        elif report_source == 'vectorstore':
            return await self._get_context_by_vectorstore(
                self.researcher.query,
                getattr(self.researcher, 'vector_store_filter', None)
            )
        
        else:
            # Default to web search
            return await self._get_context_by_web_search(
                self.researcher.query, 
                [], 
                getattr(self.researcher, 'query_domains', [])
            )
    
    async def _get_context_by_urls(self, urls: List[str]) -> str:
        """Scrape and process content from specific URLs with caching"""
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔗 Processing {len(urls)} provided URLs...",
            })
        
        # Check for new URLs
        new_search_urls = await self._get_new_urls(urls)
        
        if not new_search_urls:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": "ℹ️ All URLs have been processed previously",
                })
            return ""
        
        # Scrape content (this would use the browser manager from original)
        scraped_content = await self._scrape_urls(new_search_urls)
        
        # Process with context manager
        if scraped_content and hasattr(self.researcher, 'context_manager'):
            context = await self.researcher.context_manager.get_similar_content_by_query(
                self.researcher.query, scraped_content
            )
            return context
        
        return ""
    
    async def _get_context_by_web_search(
        self, 
        query: str, 
        scraped_data: List[Dict] = None, 
        query_domains: List[str] = None
    ) -> str:
        """Enhanced web search with caching and FAISS integration"""
        
        if scraped_data is None:
            scraped_data = []
        if query_domains is None:
            query_domains = []
        
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🌐 Starting web search for query: {query}",
            })
        
        # Generate sub-queries for comprehensive research
        sub_queries = await self.plan_research(query, query_domains)
        
        # Add original query if not in subtopic mode
        report_type = getattr(self.researcher, 'report_type', '')
        if report_type != "subtopic_report":
            sub_queries.append(query)
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🗂️ Conducting research based on {len(sub_queries)} queries: {sub_queries}...",
                "sub_queries": sub_queries
            })
        
        # Process sub-queries concurrently
        try:
            context_results = await asyncio.gather(*[
                self._process_sub_query(sub_query, scraped_data, query_domains)
                for sub_query in sub_queries
            ])
            
            # Filter out empty results and combine
            valid_contexts = [ctx for ctx in context_results if ctx and ctx.strip()]
            
            if valid_contexts:
                combined_context = " ".join(valid_contexts)
                
                # Store in FAISS for future similarity searches
                await self.faiss_service.add_documents([{
                    "content": combined_context,
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "source": "web_search"
                }])
                
                search_time = time.time() - start_time
                if self.logger:
                    await self.logger.log_performance_metric("web_search_time", search_time, "seconds")
                    await self.logger.send_json({
                        "type": "logs",
                        "content": f"✅ Web search completed in {search_time:.2f}s",
                        "context_length": len(combined_context),
                        "queries_processed": len(sub_queries)
                    })
                
                return combined_context
            
            return ""
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Web search error: {str(e)}")
            return ""
    
    async def _get_context_by_local_documents(self) -> str:
        """Enhanced local document search with FAISS"""
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": "📁 Searching local documents...",
            })
        
        try:
            # Load documents
            doc_loader = DocumentLoader(config.settings.doc_path)
            document_data = await doc_loader.load()
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📄 Loaded {len(document_data)} local documents",
                })
            
            # Add to FAISS if not already indexed
            await self.faiss_service.add_documents(document_data)
            
            # Search using FAISS
            similar_docs = await self.faiss_service.similarity_search(
                self.researcher.query,
                k=config.settings.max_search_results_per_query
            )
            
            # Extract content
            content_pieces = []
            for doc in similar_docs:
                if isinstance(doc, dict):
                    content = doc.get('content', '') or doc.get('text', '')
                else:
                    content = str(doc)
                
                if content.strip():
                    content_pieces.append(content)
            
            result = "\n\n".join(content_pieces)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"✅ Found {len(content_pieces)} relevant local documents",
                })
            
            return result
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Local document search error: {str(e)}")
            return ""
    
    async def _get_context_by_hybrid_search(self) -> str:
        """Enhanced hybrid search combining local and web sources"""
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": "🔄 Performing hybrid search (local + web)...",
            })
        
        # Search local documents
        local_context = await self._get_context_by_local_documents()
        
        # Search web sources
        web_context = await self._get_context_by_web_search(
            self.researcher.query, 
            [], 
            getattr(self.researcher, 'query_domains', [])
        )
        
        # Combine results intelligently
        combined_context = self._combine_local_web_contexts(local_context, web_context)
        
        if self.logger:
            local_length = len(local_context) if local_context else 0
            web_length = len(web_context) if web_context else 0
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔄 Hybrid search completed",
                "local_content_length": local_length,
                "web_content_length": web_length,
                "combined_length": len(combined_context)
            })
        
        return combined_context
    
    async def _get_context_by_vectorstore(self, query: str, filter_dict: Optional[Dict] = None) -> str:
        """Enhanced vectorstore search using FAISS"""
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔍 Searching vector store for: {query}",
            })
        
        # Generate sub-queries for comprehensive search
        sub_queries = await self.plan_research(query)
        if getattr(self.researcher, 'report_type', '') != "subtopic_report":
            sub_queries.append(query)
        
        # Process sub-queries using vectorstore
        context_results = await asyncio.gather(*[
            self._process_sub_query_with_vectorstore(sub_query, filter_dict)
            for sub_query in sub_queries
        ])
        
        # Combine results
        valid_contexts = [ctx for ctx in context_results if ctx and ctx.strip()]
        return " ".join(valid_contexts) if valid_contexts else ""
    
    async def _process_sub_query(
        self, 
        sub_query: str, 
        scraped_data: List[Dict] = None, 
        query_domains: List[str] = None
    ) -> str:
        """Process a single sub-query with caching"""
        
        if scraped_data is None:
            scraped_data = []
        if query_domains is None:
            query_domains = []
        
        # Check cache first
        cache_key = f"sub_query:{hash(sub_query)}:{hash(str(query_domains))}"
        cached_result = await self.redis_service.get_cached_context(cache_key)
        
        if cached_result:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📋 Using cached result for: {sub_query}",
                    "cache_hit": True
                })
            return cached_result
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔍 Running research for '{sub_query}'...",
            })
        
        try:
            # Get search results if no scraped data provided
            if not scraped_data:
                scraped_data = await self._scrape_data_by_query(sub_query, query_domains)
            
            # Process with context manager
            if scraped_data and hasattr(self.researcher, 'context_manager'):
                context = await self.researcher.context_manager.get_similar_content_by_query(
                    sub_query, scraped_data
                )
                
                # Cache the result
                if context:
                    await self.redis_service.cache_context(
                        cache_key, 
                        context, 
                        ttl=config.settings.cache_ttl_seconds
                    )
                
                return context
            
            return ""
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Sub-query processing error for '{sub_query}': {str(e)}")
            return ""
    
    async def _process_sub_query_with_vectorstore(self, sub_query: str, filter_dict: Optional[Dict] = None) -> str:
        """Process sub-query using vectorstore (FAISS)"""
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔍 Vector search for '{sub_query}'...",
            })
        
        if hasattr(self.researcher, 'context_manager'):
            return await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(
                sub_query, filter_dict
            )
        else:
            # Direct FAISS search fallback
            results = await self.faiss_service.similarity_search(sub_query, k=8, filter_dict=filter_dict)
            content_pieces = []
            for result in results:
                content = result.get('content', '') if isinstance(result, dict) else str(result)
                if content.strip():
                    content_pieces.append(content)
            return "\n\n".join(content_pieces)
    
    # Helper methods
    async def _get_initial_search_results(self, query: str, query_domains: List[str]) -> List[Dict]:
        """Get initial search results for planning"""
        # This would integrate with the retrievers system
        # For now, return empty list as placeholder
        return []
    
    async def _plan_research_outline(self, query: str, search_results: List[Dict], query_domains: List[str]) -> List[str]:
        """Generate research outline/sub-queries"""
        # This would use LLM to generate sub-queries based on initial search
        # For now, return simple sub-queries
        return [
            f"What is {query}?",
            f"Current trends in {query}",
            f"Challenges and opportunities in {query}",
            f"Future outlook for {query}"
        ]
    
    async def _get_new_urls(self, urls: List[str]) -> List[str]:
        """Filter out already visited URLs"""
        if not hasattr(self.researcher, 'visited_urls'):
            self.researcher.visited_urls = set()
        
        new_urls = []
        for url in urls:
            if url not in self.researcher.visited_urls:
                self.researcher.visited_urls.add(url)
                new_urls.append(url)
        
        return new_urls
    
    async def _scrape_urls(self, urls: List[str]) -> List[Dict]:
        """Scrape content from URLs"""
        # Placeholder for URL scraping functionality
        # This would integrate with the browser manager
        return []
    
    async def _scrape_data_by_query(self, query: str, query_domains: List[str]) -> List[Dict]:
        """Scrape data based on query using retrievers"""
        # Placeholder for retriever integration
        return []
    
    def _combine_local_web_contexts(self, local_context: str, web_context: str) -> str:
        """Intelligently combine local and web contexts"""
        contexts = []
        
        if local_context and local_context.strip():
            contexts.append(f"=== LOCAL DOCUMENTS ===\n{local_context}")
        
        if web_context and web_context.strip():
            contexts.append(f"=== WEB SOURCES ===\n{web_context}")
        
        return "\n\n".join(contexts)
    
    # Performance and monitoring methods
    async def get_research_stats(self) -> Dict[str, Any]:
        """Get research performance statistics"""
        total_time = time.time() - self._research_start_time if self._research_start_time else 0
        
        return {
            "total_research_time": total_time,
            "step_times": self._step_times,
            "cache_stats": await self.redis_service.get_cache_stats("sub_query:*"),
            "context_size": len(str(getattr(self.researcher, 'context', ''))),
            "visited_urls_count": len(getattr(self.researcher, 'visited_urls', set())),
            "total_costs": getattr(self.researcher, 'research_costs', 0.0)
        }