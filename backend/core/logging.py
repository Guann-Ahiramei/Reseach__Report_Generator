"""
Enhanced logging configuration for Industry Reporter 2
Based on original GPT-Researcher logging with modern improvements
"""
import logging
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from core.config import config


class JSONResearchHandler:
    """Enhanced JSON handler for research events"""
    
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.research_data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": None,
            "events": [],
            "content": {
                "query": "",
                "sources": [],
                "context": [],
                "report": "",
                "costs": 0.0,
                "performance_metrics": {}
            }
        }
        self._lock = asyncio.Lock()
    
    async def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event asynchronously"""
        async with self._lock:
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "data": data
            }
            self.research_data["events"].append(event)
            await self._save_json()
    
    async def update_content(self, key: str, value: Any):
        """Update content asynchronously"""
        async with self._lock:
            self.research_data["content"][key] = value
            await self._save_json()
    
    async def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics"""
        async with self._lock:
            self.research_data["content"]["performance_metrics"].update(metrics)
            await self._save_json()
    
    async def _save_json(self):
        """Save JSON data asynchronously"""
        def _write():
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.research_data, f, indent=2, ensure_ascii=False)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write)


class WebSocketHandler:
    """Handler for real-time WebSocket logging"""
    
    def __init__(self, websocket=None):
        self.websocket = websocket
        self.message_queue: List[Dict[str, Any]] = []
        self._connected = False
    
    def set_websocket(self, websocket):
        """Set or update the WebSocket connection"""
        self.websocket = websocket
        self._connected = websocket is not None
    
    async def send_log(self, log_type: str, content: str, data: Dict[str, Any] = None):
        """Send log message via WebSocket"""
        if not self._connected or not self.websocket:
            # Queue the message for later
            self.message_queue.append({
                "type": log_type,
                "content": content,
                "data": data or {},
                "timestamp": datetime.now().isoformat()
            })
            return
        
        try:
            message = {
                "type": log_type,
                "content": content,
                "output": data.get("output", content) if data else content,
                "timestamp": datetime.now().isoformat()
            }
            
            if data:
                message.update(data)
            
            await self.websocket.send_json(message)
        except Exception as e:
            logging.getLogger(__name__).error(f"WebSocket send error: {e}")
            self._connected = False
    
    async def flush_queue(self):
        """Send all queued messages"""
        if not self._connected or not self.websocket:
            return
        
        for message in self.message_queue:
            try:
                await self.websocket.send_json(message)
            except Exception as e:
                logging.getLogger(__name__).error(f"Error flushing queue: {e}")
                break
        
        self.message_queue.clear()


class CustomLogsHandler:
    """Enhanced logs handler combining file, JSON, and WebSocket logging"""
    
    def __init__(self, websocket=None, task: str = "research"):
        self.task = task
        self.timestamp = datetime.now()
        
        # Setup file paths
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        sanitized_task = self._sanitize_filename(task)
        
        self.log_file = self.log_dir / f"research_{timestamp_str}_{sanitized_task}.log"
        self.json_file = self.log_dir / f"research_{timestamp_str}_{sanitized_task}.json"
        
        # Setup handlers
        self.json_handler = JSONResearchHandler(str(self.json_file))
        self.websocket_handler = WebSocketHandler(websocket)
        
        # Setup file logger
        self.logger = logging.getLogger(f"research.{sanitized_task}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for cross-platform compatibility"""
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        return sanitized[:100] if len(sanitized) > 100 else sanitized
    
    async def send_json(self, data: Dict[str, Any]) -> None:
        """Send data to all handlers"""
        # Extract log level and message
        log_type = data.get("type", "info")
        content = data.get("content", "")
        output = data.get("output", content)
        
        # Log to file
        self.logger.info(f"{log_type}: {content}")
        
        # Log to JSON
        await self.json_handler.log_event(log_type, data)
        
        # Send to WebSocket
        await self.websocket_handler.send_log(log_type, content, data)
    
    async def log_research_start(self, query: str, config_info: Dict[str, Any]):
        """Log the start of a research session"""
        await self.json_handler.update_content("query", query)
        await self.send_json({
            "type": "research_start",
            "content": f"Starting research for: {query}",
            "query": query,
            "config": config_info
        })
    
    async def log_research_step(self, step: str, details: Dict[str, Any]):
        """Log a research step"""
        await self.send_json({
            "type": "research_step",
            "content": f"Research step: {step}",
            "step": step,
            "details": details
        })
    
    async def log_search_results(self, retriever: str, results_count: int, query: str):
        """Log search results"""
        await self.send_json({
            "type": "search_results",
            "content": f"Found {results_count} results from {retriever}",
            "retriever": retriever,
            "count": results_count,
            "query": query
        })
    
    async def log_performance_metric(self, metric_name: str, value: Any, unit: str = ""):
        """Log performance metrics"""
        metric_data = {
            metric_name: {
                "value": value,
                "unit": unit,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        await self.json_handler.update_performance_metrics(metric_data)
        await self.send_json({
            "type": "performance_metric",
            "content": f"Performance: {metric_name} = {value} {unit}",
            "metric": metric_name,
            "value": value,
            "unit": unit
        })
    
    async def log_error(self, error: str, context: Dict[str, Any] = None):
        """Log an error"""
        await self.send_json({
            "type": "error",
            "content": f"Error: {error}",
            "error": error,
            "context": context or {}
        })
    
    def set_websocket(self, websocket):
        """Update WebSocket connection"""
        self.websocket_handler.set_websocket(websocket)
    
    async def finalize(self):
        """Finalize logging session"""
        await self.send_json({
            "type": "research_complete",
            "content": "Research session completed",
            "log_file": str(self.log_file),
            "json_file": str(self.json_file)
        })


@asynccontextmanager
async def research_session(task: str, websocket=None):
    """Context manager for research logging sessions"""
    handler = CustomLogsHandler(websocket, task)
    try:
        yield handler
    finally:
        await handler.finalize()


# Setup application-wide logging
def setup_logging():
    """Setup application-wide logging configuration"""
    log_level = getattr(logging, config.settings.log_level.upper())
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Suppress verbose libraries
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    if not config.settings.debug:
        logging.getLogger('uvicorn.access').setLevel(logging.WARNING)


# Initialize logging on import
setup_logging()

# Export for easy importing
__all__ = [
    "CustomLogsHandler", 
    "JSONResearchHandler", 
    "WebSocketHandler",
    "research_session",
    "setup_logging"
]