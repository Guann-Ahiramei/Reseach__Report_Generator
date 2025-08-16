"""
FAISS Manager Skill for Industry Reporter 2
Manages FAISS vector operations as a research skill
"""
import asyncio
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from core.config import config
from core.logging import CustomLogsHandler
from services.faiss_service import FAISSService
from services.redis_service import RedisService


class FAISSManager:
    """FAISS management skill for advanced vector operations"""
    
    def __init__(self, researcher=None):
        self.researcher = researcher
        self.faiss_service = FAISSService()
        self.redis_service = RedisService()
        self.logger = getattr(researcher, 'logger', None) if researcher else None
        
        # Performance tracking
        self._operation_times = {}
        self._index_stats = {}
    
    async def initialize_index(self, documents: List[Dict[str, Any]] = None) -> bool:
        """Initialize or update the FAISS index with documents"""
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": "🔧 Initializing FAISS vector index...",
            })
        
        try:
            # Initialize the service
            await self.faiss_service.initialize()
            
            # Add documents if provided
            if documents:
                await self.faiss_service.add_documents(documents)
                
                if self.logger:
                    await self.logger.send_json({
                        "type": "logs",
                        "content": f"📚 Added {len(documents)} documents to FAISS index",
                        "documents_count": len(documents)
                    })
            
            # Get index statistics
            self._index_stats = await self.get_index_statistics()
            
            init_time = time.time() - start_time
            self._operation_times["initialization"] = init_time
            
            if self.logger:
                await self.logger.log_performance_metric("faiss_init_time", init_time, "seconds")
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"✅ FAISS index initialized in {init_time:.2f}s",
                    "index_size": self._index_stats.get("total_vectors", 0),
                    "dimension": self._index_stats.get("dimension", 0)
                })
            
            return True
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"FAISS initialization failed: {str(e)}")
            return False
    
    async def smart_similarity_search(
        self, 
        query: str, 
        k: int = 10,
        similarity_threshold: float = 0.7,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Enhanced similarity search with intelligent filtering
        """
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔍 Smart similarity search for: '{query}'",
                "k": k,
                "threshold": similarity_threshold
            })
        
        try:
            # Perform similarity search
            results = await self.faiss_service.similarity_search(
                query=query,
                k=k * 2,  # Get more results to filter
                filter_dict=None
            )
            
            # Apply similarity threshold filtering
            filtered_results = []
            for result in results:
                # Calculate semantic similarity (placeholder - would use actual embeddings)
                similarity_score = await self._calculate_semantic_similarity(query, result)
                
                if similarity_score >= similarity_threshold:
                    if include_metadata:
                        result_with_metadata = {
                            "content": result.get("content", ""),
                            "similarity_score": similarity_score,
                            "source": result.get("source", "unknown"),
                            "timestamp": result.get("timestamp", datetime.now().isoformat()),
                            "metadata": result.get("metadata", {})
                        }
                        filtered_results.append(result_with_metadata)
                    else:
                        filtered_results.append(result)
                
                if len(filtered_results) >= k:
                    break
            
            search_time = time.time() - start_time
            self._operation_times["smart_search"] = search_time
            
            if self.logger:
                await self.logger.log_performance_metric("smart_search_time", search_time, "seconds")
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🎯 Found {len(filtered_results)} relevant results (threshold: {similarity_threshold})",
                    "results_count": len(filtered_results),
                    "search_time": search_time,
                    "total_candidates": len(results)
                })
            
            return filtered_results
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Smart similarity search failed: {str(e)}")
            return []
    
    async def cluster_similar_documents(
        self, 
        documents: List[Dict[str, Any]], 
        num_clusters: int = 5
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster documents by similarity for better organization
        """
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🎯 Clustering {len(documents)} documents into {num_clusters} groups...",
            })
        
        try:
            # Get embeddings for all documents
            embeddings = []
            for doc in documents:
                content = doc.get("content", "")
                if content:
                    embedding = await self.faiss_service.get_embedding(content)
                    embeddings.append(embedding)
                else:
                    embeddings.append(np.zeros(config.faiss_config["dimension"]))
            
            if not embeddings:
                return {}
            
            # Simple k-means clustering (placeholder for more sophisticated clustering)
            clusters = await self._perform_kmeans_clustering(embeddings, num_clusters)
            
            # Organize documents by cluster
            clustered_docs = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_docs:
                    clustered_docs[cluster_id] = []
                
                if i < len(documents):
                    doc_with_cluster = documents[i].copy()
                    doc_with_cluster["cluster_id"] = cluster_id
                    clustered_docs[cluster_id].append(doc_with_cluster)
            
            cluster_time = time.time() - start_time
            self._operation_times["clustering"] = cluster_time
            
            if self.logger:
                await self.logger.log_performance_metric("clustering_time", cluster_time, "seconds")
                cluster_sizes = {k: len(v) for k, v in clustered_docs.items()}
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📊 Documents clustered in {cluster_time:.2f}s",
                    "cluster_sizes": cluster_sizes,
                    "num_clusters": len(clustered_docs)
                })
            
            return clustered_docs
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Document clustering failed: {str(e)}")
            return {}
    
    async def find_document_gaps(
        self, 
        query: str, 
        existing_documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify knowledge gaps in the existing document set
        """
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🔍 Analyzing knowledge gaps for: '{query}'",
            })
        
        try:
            # Break down query into sub-topics
            sub_topics = await self._generate_query_subtopics(query)
            
            # Check coverage for each sub-topic
            coverage_gaps = []
            for topic in sub_topics:
                # Search for related documents
                related_docs = await self.smart_similarity_search(
                    query=topic,
                    k=5,
                    similarity_threshold=0.6
                )
                
                # If few or no related documents found, it's a gap
                if len(related_docs) < 2:
                    coverage_gaps.append(topic)
            
            gap_analysis_time = time.time() - start_time
            self._operation_times["gap_analysis"] = gap_analysis_time
            
            if self.logger:
                await self.logger.log_performance_metric("gap_analysis_time", gap_analysis_time, "seconds")
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📋 Found {len(coverage_gaps)} knowledge gaps",
                    "gaps": coverage_gaps,
                    "analysis_time": gap_analysis_time
                })
            
            return coverage_gaps
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Gap analysis failed: {str(e)}")
            return []
    
    async def optimize_index_performance(self) -> Dict[str, Any]:
        """
        Optimize FAISS index for better performance
        """
        start_time = time.time()
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": "⚡ Optimizing FAISS index performance...",
            })
        
        try:
            # Get current index statistics
            before_stats = await self.get_index_statistics()
            
            # Perform optimization
            optimization_results = await self.faiss_service.optimize_index()
            
            # Get updated statistics
            after_stats = await self.get_index_statistics()
            
            optimization_time = time.time() - start_time
            self._operation_times["optimization"] = optimization_time
            
            results = {
                "optimization_time": optimization_time,
                "before_stats": before_stats,
                "after_stats": after_stats,
                "improvements": {
                    "search_speed_improvement": optimization_results.get("speed_improvement", 0),
                    "memory_reduction": optimization_results.get("memory_reduction", 0)
                }
            }
            
            if self.logger:
                await self.logger.log_performance_metric("optimization_time", optimization_time, "seconds")
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"⚡ Index optimization completed in {optimization_time:.2f}s",
                    "improvements": results["improvements"]
                })
            
            return results
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Index optimization failed: {str(e)}")
            return {}
    
    async def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive FAISS index statistics"""
        try:
            stats = await self.faiss_service.get_index_info()
            
            # Add additional computed statistics
            stats.update({
                "last_updated": datetime.now().isoformat(),
                "operation_times": self._operation_times,
                "cache_stats": await self.redis_service.get_cache_stats("faiss:*")
            })
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    # Helper methods
    async def _calculate_semantic_similarity(self, query: str, document: Dict[str, Any]) -> float:
        """Calculate semantic similarity between query and document"""
        try:
            # Get embeddings
            query_embedding = await self.faiss_service.get_embedding(query)
            doc_content = document.get("content", "")
            doc_embedding = await self.faiss_service.get_embedding(doc_content)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            return float(similarity)
            
        except Exception:
            # Fallback to simple text similarity
            return self._simple_text_similarity(query, document.get("content", ""))
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity fallback"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _generate_query_subtopics(self, query: str) -> List[str]:
        """Generate sub-topics from a query for gap analysis"""
        # Placeholder implementation - would use LLM in practice
        return [
            f"Overview of {query}",
            f"Technical aspects of {query}",
            f"Current trends in {query}",
            f"Challenges in {query}",
            f"Future of {query}"
        ]
    
    async def _perform_kmeans_clustering(
        self, 
        embeddings: List[np.ndarray], 
        num_clusters: int
    ) -> List[int]:
        """Perform k-means clustering on embeddings"""
        # Simplified clustering - would use scikit-learn in practice
        import random
        
        # For now, assign random clusters as placeholder
        return [random.randint(0, num_clusters - 1) for _ in embeddings]
    
    async def clear_index(self) -> bool:
        """Clear the FAISS index"""
        try:
            await self.faiss_service.clear_index()
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": "🧹 FAISS index cleared",
                })
            
            return True
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Index clearing failed: {str(e)}")
            return False