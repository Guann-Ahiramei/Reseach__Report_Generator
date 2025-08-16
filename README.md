# Industry Reporter 2

A modern hybrid search platform built with FAISS + Redis, combining web search and local document analysis.

## 🔧 Architecture

- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python + LCEL (LangChain Expression Language)
- **Vector Search**: FAISS for similarity search
- **Cache**: Redis for performance optimization
- **Search**: Tavily API + Local documents hybrid search
- **Deployment**: Docker + Docker Compose

## 🚀 Features

- **Hybrid Search**: Combines web search (Tavily) with local document analysis
- **Real-time Updates**: WebSocket-based live progress streaming
- **Smart Caching**: Redis-powered result caching for faster responses
- **Vector Similarity**: FAISS-powered semantic search
- **Skills-based Architecture**: Modular research skills system
- **Multi-retriever Support**: Extensible retriever system

## 🏗️ Project Structure

```
industry_reporter2/
├── backend/
│   ├── core/           # Configuration and logging
│   ├── skills/         # Research skills (context_manager, researcher, writer)
│   ├── retrievers/     # Multiple retriever implementations
│   ├── services/       # Business logic services
│   ├── api/           # FastAPI routes and WebSocket handlers
│   └── utils/         # Utility modules
├── frontend/
│   └── src/           # React application
├── data/
│   ├── documents/     # Local documents storage
│   └── faiss_index/   # FAISS index files
└── docker-compose.yml # Container orchestration
```

## 🔧 Quick Start

1. **Prerequisites**
   - Docker and Docker Compose
   - Python 3.11+
   - Node.js 18+

2. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start Services**
   ```bash
   docker-compose up -d
   ```

4. **Access Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## 🔑 Environment Variables

```env
# LLM Configuration
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key

# Redis Configuration
REDIS_URL=redis://redis:6379

# FAISS Configuration
FAISS_INDEX_PATH=/app/data/faiss_index
DOC_PATH=/app/data/documents

# Application Configuration
LOG_LEVEL=INFO
```

## 📈 Development

See individual README files in `backend/` and `frontend/` directories for detailed development instructions.

## 🤝 Contributing

This project is derived from GPT-Researcher with modern enhancements. Contributions welcome!

## 📄 License

Apache 2.0 License - see LICENSE file for details.