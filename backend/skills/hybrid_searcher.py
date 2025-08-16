"""
Hybrid Search Skill for Industry Reporter 2
Combines multiple search strategies for optimal results
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from core.config import config
from core.logging import CustomLogsHandler
from services.redis_service import RedisService
from services.faiss_service import FAISSService


@dataclass
class SearchResult:
    """Structured search result with metadata"""
    content: str
    source: str
    url: Optional[str] = None
    title: Optional[str] = None
    relevance_score: float = 0.0
    search_method: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchStrategy:
    """Search strategy configuration"""
    name: str
    weight: float
    enabled: bool = True
    timeout: int = 30
    max_results: int = 10
    parameters: Dict[str, Any] = field(default_factory=dict)


class HybridSearcher:
    """Advanced hybrid search combining multiple search strategies"""
    
    def __init__(self, researcher=None):
        self.researcher = researcher
        self.faiss_service = FAISSService()
        self.redis_service = RedisService()
        self.logger = getattr(researcher, 'logger', None) if researcher else None
        
        # Search strategies configuration
        self.strategies = {
            "tavily_web": SearchStrategy(
                name="Tavily Web Search",
                weight=0.3,
                enabled=True,
                timeout=20,
                max_results=10
            ),
            "local_documents": SearchStrategy(
                name="Local Documents",
                weight=0.25,
                enabled=True,
                timeout=15,
                max_results=8
            ),
            "faiss_similarity": SearchStrategy(
                name="FAISS Similarity",
                weight=0.25,
                enabled=True,
                timeout=10,
                max_results=8
            ),
            "redis_cache": SearchStrategy(
                name="Redis Cache",
                weight=0.2,
                enabled=True,
                timeout=5,
                max_results=5
            )
        }
        
        # Performance tracking
        self._search_times = {}
        self._result_stats = {}
    
    async def hybrid_search(
        self,
        query: str,
        search_types: List[str] = None,
        max_total_results: int = None,
        enable_fusion: bool = True,
        similarity_threshold: float = 0.6
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining multiple strategies
        """
        if search_types is None:
            search_types = list(self.strategies.keys())
        if max_total_results is None:
            max_total_results = config.settings.max_total_search_results
        
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔍 Starting hybrid search for: '{query}'",
                "search_types": search_types,
                "max_results": max_total_results
            })
        
        # Execute search strategies concurrently
        search_tasks = []
        for search_type in search_types:
            if search_type in self.strategies and self.strategies[search_type].enabled:
                task = self._execute_search_strategy(query, search_type)
                search_tasks.append(task)
        
        if not search_tasks:
            return []
        
        # Wait for all searches to complete
        try:
            strategy_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Hybrid search failed: {str(e)}")
            return []
        
        # Combine and process results
        all_results = []
        strategy_stats = {}
        
        for i, result in enumerate(strategy_results):
            if isinstance(result, Exception):
                if self.logger:
                    await self.logger.log_error(f"Search strategy failed: {str(result)}")
                continue
            
            if isinstance(result, list):
                strategy_name = search_types[i] if i < len(search_types) else f"strategy_{i}"
                strategy_stats[strategy_name] = len(result)
                all_results.extend(result)
        
        # Apply result fusion if enabled
        if enable_fusion and len(all_results) > 1:
            fused_results = await self._fuse_search_results(
                all_results, 
                similarity_threshold=similarity_threshold
            )
        else:
            fused_results = all_results
        
        # Rank and limit results
        ranked_results = await self._rank_search_results(fused_results)
        final_results = ranked_results[:max_total_results]
        
        # Track performance
        total_time = time.time() - start_time
        self._search_times["hybrid_search"] = total_time
        self._result_stats = {
            "total_time": total_time,
            "strategy_stats": strategy_stats,
            "total_raw_results": len(all_results),
            "final_results": len(final_results),
            "fusion_enabled": enable_fusion
        }
        
        if self.logger:
            await self.logger.log_performance_metric("hybrid_search_time", total_time, "seconds")
            await self.logger.send_json({
                "type": "logs",
                "content": f"🎯 Hybrid search completed in {total_time:.2f}s",
                "final_results": len(final_results),
                "strategy_stats": strategy_stats,
                "fusion_applied": enable_fusion
            })
        
        return final_results
    
    async def intelligent_query_expansion(self, query: str) -> List[str]:
        """
        Expand query with related terms for better search coverage
        """
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🧠 Expanding query: '{query}'",
            })
        
        expanded_queries = [query]  # Always include original
        
        try:
            # Use FAISS to find similar queries from cache
            cached_queries = await self._find_similar_cached_queries(query)
            
            # Extract key terms and generate variations
            key_terms = await self._extract_key_terms(query)
            term_variations = await self._generate_term_variations(key_terms)
            
            # Generate related questions
            related_questions = await self._generate_related_questions(query)
            
            # Combine all expansions
            expanded_queries.extend(cached_queries[:2])  # Top 2 similar cached queries
            expanded_queries.extend(term_variations[:3])  # Top 3 term variations
            expanded_queries.extend(related_questions[:2])  # Top 2 related questions
            
            # Remove duplicates while preserving order
            unique_queries = []
            seen = set()
            for q in expanded_queries:
                if q.lower() not in seen:
                    unique_queries.append(q)
                    seen.add(q.lower())
            
            expansion_time = time.time() - start_time
            
            if self.logger:
                await self.logger.log_performance_metric("query_expansion_time", expansion_time, "seconds")
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📈 Query expanded to {len(unique_queries)} variations",
                    "expanded_queries": unique_queries,
                    "expansion_time": expansion_time
                })
            
            return unique_queries
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Query expansion failed: {str(e)}")
            return [query]
    
    async def adaptive_search(
        self,
        query: str,
        initial_results_threshold: int = 5,
        max_iterations: int = 3
    ) -> List[SearchResult]:
        """
        Adaptive search that adjusts strategy based on initial results
        """
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔄 Starting adaptive search for: '{query}'",
                "threshold": initial_results_threshold,
                "max_iterations": max_iterations
            })
        
        all_results = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Determine search strategy for this iteration
            if iteration == 1:
                # First iteration: fast strategies
                search_types = ["redis_cache", "faiss_similarity"]
            elif iteration == 2:
                # Second iteration: add local documents
                search_types = ["local_documents", "faiss_similarity"]
            else:
                # Final iteration: comprehensive search
                search_types = list(self.strategies.keys())
            
            # Perform search
            iteration_results = await self.hybrid_search(
                query=query,
                search_types=search_types,
                enable_fusion=False  # Fusion will be applied at the end
            )
            
            all_results.extend(iteration_results)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🔄 Iteration {iteration}: found {len(iteration_results)} results",
                    "iteration": iteration,
                    "results_count": len(iteration_results),
                    "total_so_far": len(all_results)
                })
            
            # Check if we have enough good results
            high_quality_results = [r for r in all_results if r.relevance_score >= 0.7]
            if len(high_quality_results) >= initial_results_threshold:
                break
            
            # Expand query for next iteration if needed
            if iteration < max_iterations:
                expanded_queries = await self.intelligent_query_expansion(query)
                if len(expanded_queries) > 1:
                    # Use first expanded query for next iteration
                    query = expanded_queries[1]
        
        # Apply final fusion and ranking
        final_results = await self._fuse_search_results(all_results)
        ranked_results = await self._rank_search_results(final_results)
        
        adaptive_time = time.time() - start_time
        
        if self.logger:
            await self.logger.log_performance_metric("adaptive_search_time", adaptive_time, "seconds")
            await self.logger.send_json({
                "type": "logs",
                "content": f"🎯 Adaptive search completed in {adaptive_time:.2f}s",
                "iterations_used": iteration,
                "final_results": len(ranked_results),
                "high_quality_results": len([r for r in ranked_results if r.relevance_score >= 0.7])
            })
        
        return ranked_results
    
    # Strategy execution methods
    async def _execute_search_strategy(self, query: str, strategy_name: str) -> List[SearchResult]:
        """Execute a specific search strategy"""
        strategy = self.strategies.get(strategy_name)
        if not strategy or not strategy.enabled:
            return []
        
        start_time = time.time()
        
        try:
            if strategy_name == "tavily_web":
                results = await self._tavily_web_search(query, strategy)
            elif strategy_name == "local_documents":
                results = await self._local_documents_search(query, strategy)
            elif strategy_name == "faiss_similarity":
                results = await self._faiss_similarity_search(query, strategy)
            elif strategy_name == "redis_cache":
                results = await self._redis_cache_search(query, strategy)
            else:
                results = []
            
            search_time = time.time() - start_time
            
            # Add search method to results
            for result in results:
                result.search_method = strategy_name
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📊 {strategy.name}: {len(results)} results in {search_time:.2f}s",
                    "strategy": strategy_name,
                    "results_count": len(results),
                    "search_time": search_time
                })
            
            return results
            
        except asyncio.TimeoutError:
            if self.logger:
                await self.logger.log_error(f"Search strategy '{strategy_name}' timed out")
            return []
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Search strategy '{strategy_name}' failed: {str(e)}")
            return []
    
    async def _tavily_web_search(self, query: str, strategy: SearchStrategy) -> List[SearchResult]:
        """Execute Tavily web search"""
        # Placeholder for Tavily integration
        return []
    
    async def _local_documents_search(self, query: str, strategy: SearchStrategy) -> List[SearchResult]:
        """Execute local documents search"""
        # Placeholder for local document search
        return []
    
    async def _faiss_similarity_search(self, query: str, strategy: SearchStrategy) -> List[SearchResult]:
        """Execute FAISS similarity search"""
        try:
            faiss_results = await self.faiss_service.similarity_search(
                query=query,
                k=strategy.max_results
            )
            
            search_results = []
            for result in faiss_results:
                search_result = SearchResult(
                    content=result.get("content", ""),
                    source="faiss_index",
                    title=result.get("title", ""),
                    relevance_score=result.get("score", 0.0),
                    search_method="faiss_similarity",
                    metadata=result.get("metadata", {})
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"FAISS search failed: {str(e)}")
            return []
    
    async def _redis_cache_search(self, query: str, strategy: SearchStrategy) -> List[SearchResult]:
        """Execute Redis cache search"""
        try:
            cache_key = f"search_results:{hash(query)}"
            cached_results = await self.redis_service.get_cached_context(cache_key)
            
            if cached_results and isinstance(cached_results, list):
                search_results = []
                for result in cached_results[:strategy.max_results]:
                    if isinstance(result, dict):
                        search_result = SearchResult(
                            content=result.get("content", ""),
                            source="redis_cache",
                            title=result.get("title", ""),
                            url=result.get("url", ""),
                            relevance_score=result.get("relevance_score", 0.8),  # Cached results get high score
                            search_method="redis_cache",
                            metadata=result.get("metadata", {})
                        )
                        search_results.append(search_result)
                
                return search_results
            
            return []
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Redis cache search failed: {str(e)}")
            return []
    
    # Result processing methods
    async def _fuse_search_results(
        self, 
        results: List[SearchResult],
        similarity_threshold: float = 0.6
    ) -> List[SearchResult]:
        """Fuse similar results from different sources"""
        if len(results) <= 1:
            return results
        
        fused_results = []
        processed_indices = set()
        
        for i, result in enumerate(results):
            if i in processed_indices:
                continue
            
            # Find similar results
            similar_results = [result]
            for j, other_result in enumerate(results[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                similarity = await self._calculate_result_similarity(result, other_result)
                if similarity >= similarity_threshold:
                    similar_results.append(other_result)
                    processed_indices.add(j)
            
            # Fuse similar results
            if len(similar_results) > 1:
                fused_result = await self._merge_similar_results(similar_results)
                fused_results.append(fused_result)
            else:
                fused_results.append(result)
            
            processed_indices.add(i)
        
        return fused_results
    
    async def _rank_search_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank search results by relevance and quality"""
        def calculate_final_score(result: SearchResult) -> float:
            # Base relevance score
            score = result.relevance_score
            
            # Apply strategy weight
            strategy_weight = self.strategies.get(result.search_method, SearchStrategy("", 0.1)).weight
            score *= (1.0 + strategy_weight)
            
            # Boost for multiple sources (fused results)
            if "fused_sources" in result.metadata:
                source_count = len(result.metadata["fused_sources"])
                score *= (1.0 + 0.1 * source_count)
            
            # Boost for recent content
            if result.timestamp:
                try:
                    timestamp = datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))
                    hours_old = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds() / 3600
                    if hours_old < 24:  # Boost recent content
                        score *= 1.1
                except:
                    pass
            
            return score
        
        # Calculate final scores and sort
        for result in results:
            result.relevance_score = calculate_final_score(result)
        
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)
    
    # Helper methods
    async def _calculate_result_similarity(self, result1: SearchResult, result2: SearchResult) -> float:
        """Calculate similarity between two search results"""
        try:
            # Use FAISS embeddings for semantic similarity
            embedding1 = await self.faiss_service.get_embedding(result1.content)
            embedding2 = await self.faiss_service.get_embedding(result2.content)
            
            import numpy as np
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception:
            # Fallback to simple text similarity
            words1 = set(result1.content.lower().split())
            words2 = set(result2.content.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
    
    async def _merge_similar_results(self, results: List[SearchResult]) -> SearchResult:
        """Merge similar results into a single comprehensive result"""
        # Use the result with highest score as base
        base_result = max(results, key=lambda r: r.relevance_score)
        
        # Combine content from all results
        combined_content = []
        sources = []
        
        for result in results:
            if result.content not in combined_content:
                combined_content.append(result.content)
            sources.append(result.search_method)
        
        # Create merged result
        merged_result = SearchResult(
            content="\n\n".join(combined_content),
            source="fused",
            url=base_result.url,
            title=base_result.title,
            relevance_score=sum(r.relevance_score for r in results) / len(results),
            search_method="fused",
            timestamp=base_result.timestamp,
            metadata={
                **base_result.metadata,
                "fused_sources": sources,
                "source_count": len(results)
            }
        )
        
        return merged_result
    
    async def _find_similar_cached_queries(self, query: str) -> List[str]:
        """Find similar queries from cache"""
        # Placeholder implementation
        return []
    
    async def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Simple implementation - would use NLP in practice
        words = query.lower().split()
        # Filter out common words
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms
    
    async def _generate_term_variations(self, terms: List[str]) -> List[str]:
        """Generate variations of key terms"""
        variations = []
        for term in terms:
            # Add plural/singular variations
            if term.endswith('s'):
                variations.append(term[:-1])
            else:
                variations.append(term + 's')
        
        return variations[:3]  # Limit variations
    
    async def _generate_related_questions(self, query: str) -> List[str]:
        """Generate related questions"""
        # Simple question generation
        question_templates = [
            f"What are the benefits of {query}?",
            f"How does {query} work?",
            f"What are the challenges with {query}?",
            f"What is the future of {query}?"
        ]
        
        return question_templates[:2]
    
    # Configuration and monitoring methods
    def configure_strategy(self, strategy_name: str, **kwargs):
        """Configure a search strategy"""
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            for key, value in kwargs.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        return {
            "search_times": self._search_times,
            "result_stats": self._result_stats,
            "strategy_config": {name: {
                "enabled": strategy.enabled,
                "weight": strategy.weight,
                "timeout": strategy.timeout,
                "max_results": strategy.max_results
            } for name, strategy in self.strategies.items()},
            "cache_stats": await self.redis_service.get_cache_stats("search_results:*")
        }