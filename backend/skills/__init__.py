"""
Enhanced Skills Module for Industry Reporter 2
Based on GPT-Researcher's skills with modern enhancements
"""

from .context_manager import ContextManager
from .researcher import ResearchConductor  
from .writer import ReportGenerator

# Export the main skills classes
__all__ = [
    "ContextManager",
    "ResearchConductor", 
    "ReportGenerator"
]

# Version info
__version__ = "2.0.0"
__description__ = "Enhanced research skills with FAISS and Redis integration"