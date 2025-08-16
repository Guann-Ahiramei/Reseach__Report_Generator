"""
Configuration management for Industry Reporter 2
Enhanced version of GPT-Researcher config with modern features
"""
import os
import logging
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # LLM Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    
    # LLM Models
    fast_llm: str = Field("gpt-4o-mini", env="FAST_LLM")
    smart_llm: str = Field("gpt-4o", env="SMART_LLM")
    strategic_llm: str = Field("gpt-4o", env="STRATEGIC_LLM")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    
    # FAISS Configuration
    faiss_index_path: str = Field("./data/faiss_index", env="FAISS_INDEX_PATH")
    faiss_index_name: str = Field("documents_index", env="FAISS_INDEX_NAME")
    embedding_dimension: int = Field(1536, env="EMBEDDING_DIMENSION")
    
    # Document Configuration
    doc_path: str = Field("./data/documents", env="DOC_PATH")
    max_doc_size_mb: int = Field(50, env="MAX_DOC_SIZE_MB")
    supported_formats: List[str] = Field(
        ["pdf", "txt", "csv", "xlsx", "md", "pptx", "docx"],
        env="SUPPORTED_FORMATS"
    )
    
    # Search Configuration
    max_search_results_per_query: int = Field(10, env="MAX_SEARCH_RESULTS_PER_QUERY")
    max_total_search_results: int = Field(50, env="MAX_TOTAL_SEARCH_RESULTS")
    search_timeout_seconds: int = Field(30, env="SEARCH_TIMEOUT_SECONDS")
    
    # Cache Configuration
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    cache_max_size_mb: int = Field(512, env="CACHE_MAX_SIZE_MB")
    
    # Application Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("development", env="ENVIRONMENT")
    
    # WebSocket Configuration
    ws_heartbeat_interval: int = Field(30, env="WS_HEARTBEAT_INTERVAL")
    ws_connection_timeout: int = Field(300, env="WS_CONNECTION_TIMEOUT")
    
    # Performance Configuration
    async_concurrency_limit: int = Field(10, env="ASYNC_CONCURRENCY_LIMIT")
    faiss_search_batch_size: int = Field(32, env="FAISS_SEARCH_BATCH_SIZE")
    redis_connection_pool_size: int = Field(10, env="REDIS_CONNECTION_POOL_SIZE")
    
    # Research Configuration (from original GPT-Researcher)
    temperature: float = Field(0.4, env="TEMPERATURE")
    curate_sources: bool = Field(True, env="CURATE_SOURCES")
    max_iterations: int = Field(3, env="MAX_ITERATIONS")
    agent_role: Optional[str] = Field(None, env="AGENT_ROLE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class Config:
    """
    Enhanced configuration class based on GPT-Researcher's config
    Maintains compatibility while adding modern features
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.settings = Settings()
        self.config_path = config_path
        self._verbose = False
        
        # Create necessary directories
        self._ensure_directories()
        
        # Set up logging
        self._setup_logging()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.settings.doc_path,
            self.settings.faiss_index_path,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.settings.log_level.upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app.log'),
                logging.StreamHandler()
            ]
        )
    
    # Properties for backward compatibility with original GPT-Researcher
    @property
    def openai_api_key(self) -> str:
        return self.settings.openai_api_key
    
    @property
    def tavily_api_key(self) -> str:
        return self.settings.tavily_api_key
    
    @property
    def fast_llm_model(self) -> str:
        return self.settings.fast_llm
    
    @property
    def smart_llm_model(self) -> str:
        return self.settings.smart_llm
    
    @property
    def strategic_llm_model(self) -> str:
        return self.settings.strategic_llm
    
    @property
    def embedding_model(self) -> str:
        return self.settings.embedding_model
    
    @property
    def doc_path(self) -> str:
        return self.settings.doc_path
    
    @property
    def max_search_results_per_query(self) -> int:
        return self.settings.max_search_results_per_query
    
    @property
    def max_total_search_results(self) -> int:
        return self.settings.max_total_search_results
    
    @property
    def temperature(self) -> float:
        return self.settings.temperature
    
    @property
    def curate_sources(self) -> bool:
        return self.settings.curate_sources
    
    @property
    def max_iterations(self) -> int:
        return self.settings.max_iterations
    
    @property
    def agent_role(self) -> Optional[str]:
        return self.settings.agent_role
    
    # Modern configuration properties
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Redis connection configuration"""
        return {
            "url": self.settings.redis_url,
            "password": self.settings.redis_password,
            "db": self.settings.redis_db,
            "max_connections": self.settings.redis_connection_pool_size
        }
    
    @property
    def faiss_config(self) -> Dict[str, Any]:
        """FAISS configuration"""
        return {
            "index_path": self.settings.faiss_index_path,
            "index_name": self.settings.faiss_index_name,
            "dimension": self.settings.embedding_dimension,
            "batch_size": self.settings.faiss_search_batch_size
        }
    
    @property
    def performance_config(self) -> Dict[str, Any]:
        """Performance-related configuration"""
        return {
            "concurrency_limit": self.settings.async_concurrency_limit,
            "search_timeout": self.settings.search_timeout_seconds,
            "cache_ttl": self.settings.cache_ttl_seconds
        }
    
    def set_verbose(self, verbose: bool):
        """Set verbose mode for backward compatibility"""
        self._verbose = verbose
    
    def get_verbose(self) -> bool:
        """Get verbose mode status"""
        return self._verbose
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.settings.model_dump()


# Global configuration instance
config = Config()

# Export for easy importing
__all__ = ["config", "Config", "Settings"]