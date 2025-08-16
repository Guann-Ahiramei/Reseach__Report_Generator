"""
Retriever Utilities for Industry Reporter 2
Enhanced version based on GPT-Researcher's retriever utils
"""
import asyncio
import importlib
from typing import List, Dict, Type, Any, Optional
from datetime import datetime

from core.config import config
from .base_retriever import BaseRetriever, SearchResult


class RetrieverFactory:
    """Factory for creating and managing retrievers"""
    
    _retriever_registry: Dict[str, Type[BaseRetriever]] = {}
    
    @classmethod
    def register_retriever(cls, name: str, retriever_class: Type[BaseRetriever]):
        """Register a retriever class"""
        cls._retriever_registry[name] = retriever_class
    
    @classmethod
    def create_retriever(
        cls, 
        name: str, 
        query: str, 
        **kwargs
    ) -> BaseRetriever:
        """Create a retriever instance"""
        if name not in cls._retriever_registry:
            raise ValueError(f"Unknown retriever: {name}")
        
        retriever_class = cls._retriever_registry[name]
        return retriever_class(query=query, **kwargs)
    
    @classmethod
    def get_available_retrievers(cls) -> List[str]:
        """Get list of available retriever names"""
        return list(cls._retriever_registry.keys())
    
    @classmethod
    def get_retriever_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a retriever"""
        if name not in cls._retriever_registry:
            return {}
        
        retriever_class = cls._retriever_registry[name]
        return {
            "name": name,
            "class": retriever_class.__name__,
            "module": retriever_class.__module__,
            "doc": retriever_class.__doc__
        }


def get_retrievers(
    headers: Dict[str, str] = None, 
    cfg: Any = None,
    retriever_names: List[str] = None
) -> List[Type[BaseRetriever]]:
    """
    Get configured retriever classes
    Enhanced version of the original get_retrievers function
    """
    if headers is None:
        headers = {}
    if cfg is None:
        cfg = config
    
    # Determine which retrievers to use
    if retriever_names:
        enabled_retrievers = retriever_names
    else:
        # Get from config or use defaults
        enabled_retrievers = getattr(cfg, 'retrievers', 'tavily,local_docs,faiss').split(',')
    
    # Clean up retriever names
    enabled_retrievers = [r.strip() for r in enabled_retrievers if r.strip()]
    
    retriever_classes = []
    
    for retriever_name in enabled_retrievers:
        try:
            retriever_class = _get_retriever_class(retriever_name)
            if retriever_class:
                retriever_classes.append(retriever_class)
        except Exception as e:
            print(f"Warning: Could not load retriever '{retriever_name}': {e}")
    
    # Ensure we have at least one retriever
    if not retriever_classes:
        print("Warning: No retrievers loaded, falling back to Tavily")
        try:
            from .tavily.tavily_retriever import TavilyRetriever
            retriever_classes.append(TavilyRetriever)
        except ImportError:
            print("Error: Could not load fallback retriever")
    
    return retriever_classes


def _get_retriever_class(retriever_name: str) -> Optional[Type[BaseRetriever]]:
    """Get retriever class by name"""
    retriever_map = {
        'tavily': ('tavily.tavily_retriever', 'TavilyRetriever'),
        'local_docs': ('local_docs.local_docs_retriever', 'LocalDocsRetriever'),
        'faiss': ('faiss_retriever.faiss_retriever', 'FAISSRetriever'),
        'redis_cache': ('redis_cache.redis_cache_retriever', 'RedisCacheRetriever'),
        'google': ('google.google_retriever', 'GoogleRetriever'),
        'bing': ('bing.bing_retriever', 'BingRetriever'),
        'duckduckgo': ('duckduckgo.duckduckgo_retriever', 'DuckDuckGoRetriever'),
    }
    
    if retriever_name not in retriever_map:
        return None
    
    module_path, class_name = retriever_map[retriever_name]
    
    try:
        # Import the module dynamically
        module = importlib.import_module(f'retrievers.{module_path}')
        retriever_class = getattr(module, class_name)
        return retriever_class
    except (ImportError, AttributeError) as e:
        print(f"Could not import {class_name} from {module_path}: {e}")
        return None


async def search_all_retrievers(
    query: str,
    retrievers: List[Type[BaseRetriever]],
    max_results_per_retriever: int = None,
    headers: Dict[str, str] = None,
    query_domains: List[str] = None,
    websocket=None,
    logger=None,
    timeout_per_retriever: int = None
) -> Dict[str, List[SearchResult]]:
    """
    Search using all specified retrievers concurrently
    """
    if max_results_per_retriever is None:
        max_results_per_retriever = config.settings.max_search_results_per_query
    if timeout_per_retriever is None:
        timeout_per_retriever = config.settings.search_timeout_seconds
    
    # Create retriever instances
    retriever_instances = []
    for retriever_class in retrievers:
        try:
            instance = retriever_class(
                query=query,
                headers=headers or {},
                query_domains=query_domains or [],
                websocket=websocket,
                logger=logger,
                timeout=timeout_per_retriever
            )
            retriever_instances.append(instance)
        except Exception as e:
            if logger:
                await logger.log_error(f"Failed to create {retriever_class.__name__}: {e}")
    
    if not retriever_instances:
        return {}
    
    # Execute searches concurrently
    search_tasks = []
    for instance in retriever_instances:
        task = asyncio.create_task(
            _safe_search(instance, max_results_per_retriever, logger)
        )
        search_tasks.append((instance.name, task))
    
    # Collect results
    results = {}
    for retriever_name, task in search_tasks:
        try:
            search_results = await task
            results[retriever_name] = search_results
        except Exception as e:
            if logger:
                await logger.log_error(f"Search failed for {retriever_name}: {e}")
            results[retriever_name] = []
    
    return results


async def _safe_search(
    retriever: BaseRetriever, 
    max_results: int, 
    logger=None
) -> List[SearchResult]:
    """Safely execute a search with error handling"""
    try:
        return await retriever.search(max_results)
    except Exception as e:
        if logger:
            await logger.log_error(f"Safe search failed for {retriever.name}: {e}")
        return []


async def merge_search_results(
    results_by_retriever: Dict[str, List[SearchResult]],
    max_total_results: int = None,
    deduplicate: bool = True,
    score_weights: Dict[str, float] = None
) -> List[SearchResult]:
    """
    Merge and rank results from multiple retrievers
    """
    if max_total_results is None:
        max_total_results = config.settings.max_total_search_results
    
    # Default weights for different retriever types
    default_weights = {
        'tavily': 1.0,
        'faiss': 0.9,
        'local_docs': 0.8,
        'redis_cache': 0.7,
        'google': 0.6,
        'bing': 0.6,
        'duckduckgo': 0.5
    }
    
    if score_weights is None:
        score_weights = default_weights
    
    all_results = []
    
    # Collect all results and apply retriever-specific weights
    for retriever_name, results in results_by_retriever.items():
        weight = score_weights.get(retriever_name, 0.5)
        
        for result in results:
            # Apply weight to relevance score
            result.relevance_score = result.relevance_score * weight
            result.metadata['retriever_weight'] = weight
            result.metadata['original_score'] = result.relevance_score / weight
            
            all_results.append(result)
    
    # Deduplicate by URL if requested
    if deduplicate:
        seen_urls = set()
        deduplicated_results = []
        
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                deduplicated_results.append(result)
            else:
                # Find existing result and potentially merge
                for existing in deduplicated_results:
                    if existing.url == result.url:
                        # Keep the result with higher score
                        if result.relevance_score > existing.relevance_score:
                            deduplicated_results.remove(existing)
                            deduplicated_results.append(result)
                        break
        
        all_results = deduplicated_results
    
    # Sort by relevance score (descending)
    all_results.sort(key=lambda r: r.relevance_score, reverse=True)
    
    # Limit to max results
    return all_results[:max_total_results]


def filter_results_by_domains(
    results: List[SearchResult], 
    allowed_domains: List[str]
) -> List[SearchResult]:
    """Filter results to only include specified domains"""
    if not allowed_domains:
        return results
    
    filtered_results = []
    for result in results:
        if any(domain.lower() in result.url.lower() for domain in allowed_domains):
            filtered_results.append(result)
    
    return filtered_results


def calculate_search_quality_score(
    results: List[SearchResult],
    query: str
) -> Dict[str, float]:
    """Calculate quality metrics for search results"""
    if not results:
        return {
            "average_relevance": 0.0,
            "result_diversity": 0.0,
            "query_coverage": 0.0,
            "overall_quality": 0.0
        }
    
    # Average relevance score
    avg_relevance = sum(r.relevance_score for r in results) / len(results)
    
    # Result diversity (unique domains)
    unique_domains = set()
    for result in results:
        try:
            from urllib.parse import urlparse
            domain = urlparse(result.url).netloc
            unique_domains.add(domain)
        except:
            pass
    
    diversity = len(unique_domains) / len(results) if results else 0
    
    # Query coverage (how many query terms are covered)
    query_words = set(query.lower().split())
    covered_words = set()
    
    for result in results:
        result_words = set((result.title + " " + result.content).lower().split())
        covered_words.update(query_words.intersection(result_words))
    
    coverage = len(covered_words) / len(query_words) if query_words else 0
    
    # Overall quality (weighted combination)
    overall_quality = (avg_relevance * 0.5) + (diversity * 0.3) + (coverage * 0.2)
    
    return {
        "average_relevance": avg_relevance,
        "result_diversity": diversity,
        "query_coverage": coverage,
        "overall_quality": overall_quality
    }


# Register built-in retrievers when module is imported
def _register_builtin_retrievers():
    """Register built-in retrievers with the factory"""
    try:
        from .tavily.tavily_retriever import TavilyRetriever
        RetrieverFactory.register_retriever('tavily', TavilyRetriever)
    except ImportError:
        pass
    
    try:
        from .local_docs.local_docs_retriever import LocalDocsRetriever
        RetrieverFactory.register_retriever('local_docs', LocalDocsRetriever)
    except ImportError:
        pass
    
    try:
        from .faiss_retriever.faiss_retriever import FAISSRetriever
        RetrieverFactory.register_retriever('faiss', FAISSRetriever)
    except ImportError:
        pass
    
    try:
        from .redis_cache.redis_cache_retriever import RedisCacheRetriever
        RetrieverFactory.register_retriever('redis_cache', RedisCacheRetriever)
    except ImportError:
        pass


# Auto-register retrievers on import
_register_builtin_retrievers()