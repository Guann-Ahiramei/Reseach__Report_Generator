"""
Enhanced Report Generator for Industry Reporter 2
Based on GPT-Researcher's writer.py with modern enhancements
"""
import asyncio
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.config import config
from core.logging import CustomLogsHandler
from services.redis_service import RedisService
from services.faiss_service import FAISSService


class ReportGenerator:
    """Enhanced report generator with caching and performance optimization"""
    
    def __init__(self, researcher):
        self.researcher = researcher
        self.redis_service = RedisService()
        self.faiss_service = FAISSService()
        self.logger = getattr(researcher, 'logger', None)
        
        # Report generation parameters
        self.research_params = {
            "query": self.researcher.query,
            "agent_role_prompt": getattr(config, 'agent_role', None) or getattr(self.researcher, 'role', ''),
            "report_type": getattr(self.researcher, 'report_type', 'research_report'),
            "report_source": getattr(self.researcher, 'report_source', 'web'),
            "tone": getattr(self.researcher, 'tone', 'objective'),
            "websocket": getattr(self.researcher, 'websocket', None),
            "cfg": config,
            "headers": getattr(self.researcher, 'headers', {}),
        }
        
        # Performance tracking
        self._generation_start_time = None
        self._step_times = {}
    
    async def write_report(
        self, 
        existing_headers: List[str] = None,
        relevant_written_contents: List[str] = None,
        ext_context: Optional[str] = None,
        custom_prompt: str = ""
    ) -> str:
        """
        Enhanced report writing with caching and performance optimization
        """
        if existing_headers is None:
            existing_headers = []
        if relevant_written_contents is None:
            relevant_written_contents = []
            
        self._generation_start_time = time.time()
        
        # Check cache for report
        cache_key = self._generate_report_cache_key(
            existing_headers, relevant_written_contents, ext_context, custom_prompt
        )
        
        cached_report = await self.redis_service.get_cached_context(cache_key)
        if cached_report:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📄 Using cached report for '{self.researcher.query}'",
                    "cache_hit": True
                })
            return cached_report
        
        # Send selected images before writing report
        research_images = getattr(self.researcher, 'research_images', [])
        if research_images and self.logger:
            await self.logger.send_json({
                "type": "images",
                "content": "selected_images",
                "output": json.dumps(research_images[:10]),  # Limit to 10 images
                "images": research_images[:10]
            })
        
        # Log report generation start
        context = ext_context or getattr(self.researcher, 'context', '')
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"✍️ Writing report for '{self.researcher.query}'...",
                "context_length": len(str(context)),
                "custom_prompt": bool(custom_prompt)
            })
        
        # Prepare report parameters
        report_params = self.research_params.copy()
        report_params.update({
            "context": context,
            "custom_prompt": custom_prompt,
            "existing_headers": existing_headers,
            "relevant_written_contents": relevant_written_contents
        })
        
        # Generate the report
        try:
            report = await self._generate_report_content(report_params)
            
            # Cache the report
            if report:
                await self.redis_service.cache_context(
                    cache_key, 
                    report, 
                    ttl=config.settings.cache_ttl_seconds * 2  # Reports have longer TTL
                )
            
            generation_time = time.time() - self._generation_start_time
            
            # Log completion
            if self.logger:
                await self.logger.log_performance_metric("report_generation_time", generation_time, "seconds")
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📝 Report written for '{self.researcher.query}' in {generation_time:.2f}s",
                    "report_length": len(report),
                    "generation_time": generation_time
                })
                
                # Store report in research logs
                await self.logger.json_handler.update_content("report", report)
            
            return report
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Report generation failed: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    async def write_report_conclusion(self, report_content: str) -> str:
        """
        Enhanced conclusion writing with caching
        """
        # Check cache
        cache_key = f"conclusion:{hash(report_content)}:{hash(self.researcher.query)}"
        cached_conclusion = await self.redis_service.get_cached_context(cache_key)
        
        if cached_conclusion:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📝 Using cached conclusion for '{self.researcher.query}'",
                    "cache_hit": True
                })
            return cached_conclusion
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"✍️ Writing conclusion for '{self.researcher.query}'...",
            })
        
        try:
            conclusion = await self._generate_conclusion(report_content)
            
            # Cache the conclusion
            if conclusion:
                await self.redis_service.cache_context(cache_key, conclusion)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📝 Conclusion written for '{self.researcher.query}'",
                    "conclusion_length": len(conclusion)
                })
            
            return conclusion
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Conclusion generation failed: {str(e)}")
            return "Error generating conclusion."
    
    async def write_introduction(self) -> str:
        """
        Enhanced introduction writing with caching
        """
        # Check cache
        cache_key = f"introduction:{hash(self.researcher.query)}:{hash(str(getattr(self.researcher, 'context', '')))}"
        cached_intro = await self.redis_service.get_cached_context(cache_key)
        
        if cached_intro:
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📝 Using cached introduction for '{self.researcher.query}'",
                    "cache_hit": True
                })
            return cached_intro
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"✍️ Writing introduction for '{self.researcher.query}'...",
            })
        
        try:
            introduction = await self._generate_introduction()
            
            # Cache the introduction
            if introduction:
                await self.redis_service.cache_context(cache_key, introduction)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📝 Introduction written for '{self.researcher.query}'",
                    "introduction_length": len(introduction)
                })
            
            return introduction
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Introduction generation failed: {str(e)}")
            return "Error generating introduction."
    
    async def get_subtopics(self) -> List[str]:
        """
        Enhanced subtopic generation with caching
        """
        cache_key = f"subtopics:{hash(self.researcher.query)}:{hash(str(getattr(self.researcher, 'context', '')))}"
        cached_subtopics = await self.redis_service.get_cached_context(cache_key)
        
        if cached_subtopics and isinstance(cached_subtopics, list):
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🌳 Using cached subtopics for '{self.researcher.query}'",
                    "cache_hit": True,
                    "subtopics": cached_subtopics
                })
            return cached_subtopics
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"🌳 Generating subtopics for '{self.researcher.query}'...",
            })
        
        try:
            subtopics = await self._generate_subtopics()
            
            # Cache the subtopics
            if subtopics:
                await self.redis_service.cache_context(cache_key, subtopics)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📊 Generated {len(subtopics)} subtopics for '{self.researcher.query}'",
                    "subtopics": subtopics
                })
            
            return subtopics
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Subtopic generation failed: {str(e)}")
            return []
    
    async def get_draft_section_titles(self, current_subtopic: str) -> List[str]:
        """
        Enhanced section title generation with caching
        """
        cache_key = f"section_titles:{hash(current_subtopic)}:{hash(self.researcher.query)}"
        cached_titles = await self.redis_service.get_cached_context(cache_key)
        
        if cached_titles and isinstance(cached_titles, list):
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"📑 Using cached section titles for '{current_subtopic}'",
                    "cache_hit": True
                })
            return cached_titles
        
        if self.logger:
            await self.logger.send_json({
                "type": "logs",
                "content": f"📑 Generating draft section titles for '{current_subtopic}'...",
            })
        
        try:
            section_titles = await self._generate_draft_section_titles(current_subtopic)
            
            # Cache the section titles
            if section_titles:
                await self.redis_service.cache_context(cache_key, section_titles)
            
            if self.logger:
                await self.logger.send_json({
                    "type": "logs",
                    "content": f"🗂️ Generated {len(section_titles)} section titles for '{current_subtopic}'",
                    "section_titles": section_titles
                })
            
            return section_titles
            
        except Exception as e:
            if self.logger:
                await self.logger.log_error(f"Section title generation failed: {str(e)}")
            return []
    
    # Content generation methods (these would integrate with LLM services)
    async def _generate_report_content(self, params: Dict[str, Any]) -> str:
        """Generate the main report content using LLM"""
        # This would integrate with the LLM service (OpenAI, etc.)
        # For now, return a structured template
        
        query = params.get("query", "")
        context = params.get("context", "")
        report_type = params.get("report_type", "research_report")
        tone = params.get("tone", "objective")
        
        # Build prompt based on report type and parameters
        if report_type == "subtopic_report":
            return await self._generate_subtopic_report(params)
        else:
            return await self._generate_standard_report(params)
    
    async def _generate_standard_report(self, params: Dict[str, Any]) -> str:
        """Generate a standard research report"""
        query = params.get("query", "")
        context = params.get("context", "")
        
        # This is a placeholder - would use actual LLM integration
        sections = []
        
        # Introduction
        intro = await self._generate_introduction()
        if intro:
            sections.append(f"## Introduction\n\n{intro}")
        
        # Main content based on context
        if context:
            # Parse context and create sections
            main_content = await self._process_context_into_sections(context, query)
            sections.append(main_content)
        
        # Conclusion
        conclusion = await self._generate_conclusion(context)
        if conclusion:
            sections.append(f"## Conclusion\n\n{conclusion}")
        
        # Sources (if available)
        sources = await self._generate_sources_section()
        if sources:
            sections.append(f"## Sources\n\n{sources}")
        
        return "\n\n".join(sections)
    
    async def _generate_subtopic_report(self, params: Dict[str, Any]) -> str:
        """Generate a subtopic-specific report"""
        # This would handle subtopic reports differently
        return await self._generate_standard_report(params)
    
    async def _generate_introduction(self) -> str:
        """Generate report introduction"""
        query = self.researcher.query
        context = getattr(self.researcher, 'context', '')
        
        # Placeholder introduction generation
        return f"""This report provides a comprehensive analysis of {query}. 
        
The research was conducted using multiple sources and covers key aspects, current trends, challenges, and future outlook. This analysis aims to provide actionable insights and a thorough understanding of the topic."""
    
    async def _generate_conclusion(self, report_content: str) -> str:
        """Generate report conclusion"""
        # Placeholder conclusion generation
        return f"""In conclusion, this analysis of {self.researcher.query} reveals several key insights:

1. The topic shows significant complexity and multiple dimensions
2. Current trends indicate ongoing development and change
3. Future prospects remain promising with continued attention and investment

This research provides a foundation for further investigation and decision-making in this area."""
    
    async def _generate_subtopics(self) -> List[str]:
        """Generate subtopics for the research query"""
        query = self.researcher.query
        
        # Placeholder subtopic generation
        subtopics = [
            f"Overview of {query}",
            f"Current trends in {query}",
            f"Key challenges and opportunities",
            f"Future outlook and recommendations"
        ]
        
        return subtopics
    
    async def _generate_draft_section_titles(self, current_subtopic: str) -> List[str]:
        """Generate section titles for a subtopic"""
        # Placeholder section title generation
        section_titles = [
            f"Introduction to {current_subtopic}",
            f"Key aspects of {current_subtopic}",
            f"Analysis and findings",
            f"Implications and recommendations"
        ]
        
        return section_titles
    
    async def _process_context_into_sections(self, context: str, query: str) -> str:
        """Process research context into structured sections"""
        # This would use LLM to structure the context into coherent sections
        # For now, return formatted context
        
        if not context:
            return "## Analysis\n\nNo research context available."
        
        # Split context into logical sections
        sections = []
        
        # Check if context has natural sections
        if "===" in context:
            parts = context.split("===")
            for i, part in enumerate(parts):
                if part.strip():
                    section_title = f"## Section {i+1}"
                    if "LOCAL DOCUMENTS" in part:
                        section_title = "## Local Document Analysis"
                    elif "WEB SOURCES" in part:
                        section_title = "## Web Research Findings"
                    
                    sections.append(f"{section_title}\n\n{part.strip()}")
        else:
            # Create a single analysis section
            sections.append(f"## Analysis\n\n{context}")
        
        return "\n\n".join(sections)
    
    async def _generate_sources_section(self) -> str:
        """Generate sources section"""
        visited_urls = getattr(self.researcher, 'visited_urls', set())
        research_sources = getattr(self.researcher, 'research_sources', [])
        
        if not visited_urls and not research_sources:
            return ""
        
        sources = ["## Sources"]
        
        # Add URLs
        if visited_urls:
            sources.append("### Web Sources")
            for i, url in enumerate(visited_urls, 1):
                sources.append(f"{i}. {url}")
        
        # Add research sources
        if research_sources:
            sources.append("### Research Sources")
            for i, source in enumerate(research_sources, 1):
                if isinstance(source, dict):
                    title = source.get('title', 'Unknown Title')
                    url = source.get('url', '')
                    sources.append(f"{i}. {title} - {url}")
                else:
                    sources.append(f"{i}. {str(source)}")
        
        return "\n".join(sources)
    
    def _generate_report_cache_key(
        self, 
        existing_headers: List[str],
        relevant_written_contents: List[str],
        ext_context: Optional[str],
        custom_prompt: str
    ) -> str:
        """Generate cache key for report"""
        key_components = [
            self.researcher.query,
            str(existing_headers),
            str(relevant_written_contents),
            ext_context or "",
            custom_prompt,
            str(getattr(self.researcher, 'context', ''))
        ]
        
        combined = "|".join(key_components)
        return f"report:{hash(combined)}"
    
    # Performance and monitoring methods
    async def get_generation_stats(self) -> Dict[str, Any]:
        """Get report generation statistics"""
        total_time = time.time() - self._generation_start_time if self._generation_start_time else 0
        
        return {
            "total_generation_time": total_time,
            "step_times": self._step_times,
            "cache_stats": await self.redis_service.get_cache_stats("report:*"),
            "report_length": len(str(getattr(self.researcher, 'context', ''))),
            "images_count": len(getattr(self.researcher, 'research_images', [])),
            "sources_count": len(getattr(self.researcher, 'visited_urls', set()))
        }