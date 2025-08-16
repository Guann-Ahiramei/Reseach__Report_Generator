"""
FAISS Service for Industry Reporter 2
Enhanced FAISS service with async support and advanced vector operations
"""
import os
import json
import pickle
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from core.config import config
from core.logging import get_logger
from services.redis_service import RedisService

logger = get_logger(__name__)


class FAISSService:
    """
    Enhanced FAISS service with async support and advanced vector operations
    """
    
    def __init__(self, **kwargs):
        # Configuration
        self.index_path = kwargs.get('index_path', config.settings.faiss_index_path)
        self.dimension = kwargs.get('dimension', 1536)  # OpenAI embedding dimension
        self.index_type = kwargs.get('index_type', 'IVF')  # IVF, HNSW, Flat
        self.distance_metric = kwargs.get('distance_metric', 'cosine')  # cosine, l2, inner_product
        
        # Advanced configuration
        self.nlist = kwargs.get('nlist', 100)  # Number of clusters for IVF
        self.nprobe = kwargs.get('nprobe', 10)  # Number of clusters to search
        self.ef_search = kwargs.get('ef_search', 128)  # HNSW search parameter
        self.ef_construction = kwargs.get('ef_construction', 200)  # HNSW construction parameter
        
        # Text processing
        self.chunk_size = kwargs.get('chunk_size', 1000)
        self.chunk_overlap = kwargs.get('chunk_overlap', 200)
        self.max_tokens_per_chunk = kwargs.get('max_tokens_per_chunk', 8000)
        
        # Services and components
        self.embeddings = None
        self.text_splitter = None
        self.redis_service = RedisService()
        
        # FAISS components
        self.index = None
        self.docstore = {}
        self.index_to_docstore_id = {}
        
        # State tracking
        self._is_initialized = False
        self._last_save_time = None
        self._document_count = 0
        
        # Ensure index directory exists
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize FAISS service"""
        try:
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=config.settings.openai_api_key,
                model="text-embedding-ada-002"
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Load or create FAISS index
            await self._load_or_create_index()
            
            self._is_initialized = True
            logger.info(f"FAISS service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS service: {e}")
            raise
    
    async def is_initialized(self) -> bool:
        """Check if FAISS service is initialized"""
        return self._is_initialized
    
    async def _load_or_create_index(self):
        """Load existing index or create new one"""
        index_file = f"{self.index_path}.faiss"
        metadata_file = f"{self.index_path}.metadata"
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            await self._load_index()
        else:
            await self._create_index()
    
    async def _create_index(self):
        """Create a new FAISS index"""
        try:
            if self.index_type.upper() == 'IVF':
                # IVF (Inverted File) index
                quantizer = faiss.IndexFlatIP(self.dimension) if self.distance_metric == 'inner_product' else faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                
            elif self.index_type.upper() == 'HNSW':
                # HNSW (Hierarchical Navigable Small World) index
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = self.ef_construction
                self.index.hnsw.efSearch = self.ef_search
                
            elif self.index_type.upper() == 'FLAT':
                # Flat (brute force) index
                if self.distance_metric == 'inner_product':
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            self.docstore = {}
            self.index_to_docstore_id = {}
            self._document_count = 0
            
            logger.info(f"Created new FAISS index: {self.index_type} with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise
    
    async def _load_index(self):
        """Load existing FAISS index"""
        try:
            index_file = f"{self.index_path}.faiss"
            metadata_file = f"{self.index_path}.metadata"
            
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.docstore = metadata.get('docstore', {})
            self.index_to_docstore_id = metadata.get('index_to_docstore_id', {})
            self._document_count = metadata.get('document_count', 0)
            
            # Set search parameters for IVF
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
            
            logger.info(f"Loaded FAISS index with {self._document_count} documents")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            await self._create_index()
    
    async def save_index(self):
        """Save FAISS index to disk"""
        try:
            index_file = f"{self.index_path}.faiss"
            metadata_file = f"{self.index_path}.metadata"
            
            # Ensure directory exists
            Path(index_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, index_file)
            
            # Save metadata
            metadata = {
                'docstore': self.docstore,
                'index_to_docstore_id': self.index_to_docstore_id,
                'document_count': self._document_count,
                'index_type': self.index_type,
                'dimension': self.dimension,
                'distance_metric': self.distance_metric,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self._last_save_time = datetime.now()
            logger.info(f"Saved FAISS index with {self._document_count} documents")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to FAISS index"""
        try:
            if not self._is_initialized:
                await self.initialize()
            
            if not documents:
                return
            
            # Process documents into chunks
            chunks = []
            for doc in documents:
                doc_chunks = await self._process_document(doc)
                chunks.extend(doc_chunks)
            
            if not chunks:
                return
            
            # Generate embeddings
            texts = [chunk['content'] for chunk in chunks]
            embeddings = await self._get_embeddings_batch(texts)
            
            # Add to index
            if self.index_type.upper() == 'IVF' and not self.index.is_trained:
                # Train IVF index if not already trained
                if len(embeddings) >= self.nlist:
                    self.index.train(np.array(embeddings))
                    logger.info("Trained IVF index")
            
            # Add vectors to index
            start_idx = self.index.ntotal
            self.index.add(np.array(embeddings))
            
            # Update docstore and mapping
            for i, chunk in enumerate(chunks):
                doc_id = f"doc_{start_idx + i}"
                self.docstore[doc_id] = chunk
                self.index_to_docstore_id[str(start_idx + i)] = doc_id
            
            self._document_count += len(chunks)
            
            # Auto-save periodically
            if self._document_count % 100 == 0:
                await self.save_index()
            
            logger.info(f"Added {len(chunks)} document chunks to FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS index: {e}")
            raise
    
    async def _process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a document into chunks"""
        try:
            content = document.get('content', '')
            if not content:
                return []
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'content': chunk,
                    'metadata': document.get('metadata', {}),
                    'file_path': document.get('file_path', ''),
                    'file_name': document.get('file_name', ''),
                    'file_type': document.get('file_type', ''),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'source': document.get('source', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Copy additional fields
                for key, value in document.items():
                    if key not in chunk_data:
                        chunk_data[key] = value
                
                processed_chunks.append(chunk_data)
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            return []
    
    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        try:
            # Use asyncio to run embeddings in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self.embeddings.embed_documents, 
                texts
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            raise
    
    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                self.embeddings.embed_query, 
                text
            )
            return np.array(embedding)
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 10, 
        filter_dict: Dict[str, Any] = None,
        fetch_k: int = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        try:
            if not self._is_initialized:
                await self.initialize()
            
            if self.index.ntotal == 0:
                return []
            
            # Get query embedding
            query_embedding = await self.get_embedding(query)
            
            # Perform search
            results = await self.similarity_search_by_vector(
                query_embedding, k, filter_dict, fetch_k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def similarity_search_by_vector(
        self, 
        embedding: np.ndarray, 
        k: int = 10,
        filter_dict: Dict[str, Any] = None,
        fetch_k: int = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search by vector"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Determine how many to fetch
            if fetch_k is None:
                fetch_k = max(k * 2, 50)  # Fetch more for filtering
            
            # Search
            scores, indices = self.index.search(
                embedding.reshape(1, -1), 
                min(fetch_k, self.index.ntotal)
            )
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                doc_id = self.index_to_docstore_id.get(str(idx))
                if not doc_id:
                    continue
                
                doc = self.docstore.get(doc_id)
                if not doc:
                    continue
                
                # Apply filters
                if filter_dict and not self._matches_filter(doc, filter_dict):
                    continue
                
                # Add score and return
                result = doc.copy()
                result['score'] = float(score)
                results.append(result)
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []
    
    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 10,
        fetch_k: int = None,
        lambda_mult: float = 0.5,
        filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Perform Maximum Marginal Relevance search for diversity"""
        try:
            if not self._is_initialized:
                await self.initialize()
            
            if self.index.ntotal == 0:
                return []
            
            if fetch_k is None:
                fetch_k = max(k * 3, 50)
            
            # Get query embedding
            query_embedding = await self.get_embedding(query)
            
            # Get initial candidate documents
            candidates = await self.similarity_search_by_vector(
                query_embedding, fetch_k, filter_dict
            )
            
            if not candidates:
                return []
            
            # Extract embeddings for MMR calculation
            candidate_embeddings = []
            for doc in candidates:
                # Get embedding for document content
                doc_embedding = await self.get_embedding(doc['content'])
                candidate_embeddings.append(doc_embedding)
            
            # Perform MMR selection
            selected_indices = self._mmr_selection(
                query_embedding,
                candidate_embeddings,
                k,
                lambda_mult
            )
            
            # Return selected documents
            return [candidates[i] for i in selected_indices]
            
        except Exception as e:
            logger.error(f"MMR search failed: {e}")
            return []
    
    def _mmr_selection(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        k: int,
        lambda_mult: float
    ) -> List[int]:
        """Perform MMR selection algorithm"""
        try:
            if not candidate_embeddings:
                return []
            
            selected = []
            remaining = list(range(len(candidate_embeddings)))
            
            # Calculate similarity to query for all candidates
            query_similarities = []
            for emb in candidate_embeddings:
                sim = np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                )
                query_similarities.append(sim)
            
            for _ in range(min(k, len(candidate_embeddings))):
                if not remaining:
                    break
                
                best_score = float('-inf')
                best_idx = None
                
                for idx in remaining:
                    # Relevance score (similarity to query)
                    relevance = query_similarities[idx]
                    
                    # Diversity score (max similarity to already selected)
                    if selected:
                        max_sim_to_selected = max(
                            np.dot(candidate_embeddings[idx], candidate_embeddings[sel_idx]) / (
                                np.linalg.norm(candidate_embeddings[idx]) * 
                                np.linalg.norm(candidate_embeddings[sel_idx])
                            )
                            for sel_idx in selected
                        )
                    else:
                        max_sim_to_selected = 0
                    
                    # MMR score
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
            
            return selected
            
        except Exception as e:
            logger.error(f"MMR selection failed: {e}")
            return list(range(min(k, len(candidate_embeddings))))
    
    def _matches_filter(self, document: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if document matches filter criteria"""
        try:
            for key, value in filter_dict.items():
                doc_value = document.get(key)
                
                if isinstance(value, list):
                    if doc_value not in value:
                        return False
                elif isinstance(value, dict):
                    # Handle nested filtering
                    if 'contains' in value:
                        if value['contains'] not in str(doc_value):
                            return False
                    elif 'equals' in value:
                        if doc_value != value['equals']:
                            return False
                else:
                    if doc_value != value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Filter matching failed: {e}")
            return True  # Default to include if filter fails
    
    async def filter_search(
        self, 
        filter_dict: Dict[str, Any], 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search using only metadata filters (no vector similarity)"""
        try:
            results = []
            
            for doc_id, doc in self.docstore.items():
                if self._matches_filter(doc, filter_dict):
                    result = doc.copy()
                    result['score'] = 1.0  # No similarity score for pure filter search
                    results.append(result)
                    
                    if len(results) >= k:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Filter search failed: {e}")
            return []
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Get information about the FAISS index"""
        try:
            if not self._is_initialized:
                return {"error": "Index not initialized"}
            
            info = {
                "index_type": self.index_type,
                "dimension": self.dimension,
                "distance_metric": self.distance_metric,
                "total_vectors": self.index.ntotal if self.index else 0,
                "document_count": self._document_count,
                "is_trained": getattr(self.index, 'is_trained', True),
                "index_path": self.index_path,
                "last_save_time": self._last_save_time.isoformat() if self._last_save_time else None
            }
            
            # Add index-specific parameters
            if hasattr(self.index, 'nlist'):
                info["nlist"] = self.index.nlist
                info["nprobe"] = self.index.nprobe
            
            if hasattr(self.index, 'hnsw'):
                info["ef_construction"] = self.index.hnsw.efConstruction
                info["ef_search"] = self.index.hnsw.efSearch
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {"error": str(e)}
    
    async def get_document_count(self) -> int:
        """Get total number of documents in index"""
        return self._document_count
    
    async def clear_index(self):
        """Clear the entire index"""
        try:
            await self._create_index()
            await self.save_index()
            logger.info("FAISS index cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise
    
    async def optimize_index(self):
        """Optimize the index for better performance"""
        try:
            if self.index_type.upper() == 'IVF' and hasattr(self.index, 'make_direct_map'):
                # Add direct map for faster ID-based lookups
                self.index.make_direct_map()
                logger.info("Added direct map to IVF index")
            
            # Force save after optimization
            await self.save_index()
            
        except Exception as e:
            logger.error(f"Failed to optimize index: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on FAISS service"""
        try:
            start_time = datetime.now()
            
            # Check initialization
            init_ok = self._is_initialized
            
            # Check index
            index_ok = self.index is not None and self.index.ntotal >= 0
            
            # Check embeddings
            embeddings_ok = self.embeddings is not None
            
            # Test search if we have documents
            search_ok = True
            if self.index and self.index.ntotal > 0:
                try:
                    test_results = await self.similarity_search("test query", k=1)
                    search_ok = isinstance(test_results, list)
                except:
                    search_ok = False
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy" if all([init_ok, index_ok, embeddings_ok, search_ok]) else "unhealthy",
                "latency_ms": latency_ms,
                "checks": {
                    "initialized": init_ok,
                    "index_ready": index_ok,
                    "embeddings_ready": embeddings_ok,
                    "search_functional": search_ok
                },
                "index_info": await self.get_index_info(),
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global FAISS service instance
faiss_service = FAISSService()


# Convenience functions
async def get_faiss_service() -> FAISSService:
    """Get the global FAISS service instance"""
    if not await faiss_service.is_initialized():
        await faiss_service.initialize()
    return faiss_service