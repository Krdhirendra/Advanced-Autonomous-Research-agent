# 🤖 AARA: Advanced Autonomous Research Agent

> An intelligent, self-orchestrating multi-agent system that performs deep research by synthesizing web sources, evaluating information quality, and generating comprehensive reports with citations.

**Live Demo:** [AARA Frontend](https://aara-frontend.vercel.app/)

![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent%20LangGraph-blue)
![Stack](https://img.shields.io/badge/Stack-FastAPI%20%2B%20PostgreSQL-green)
![RAG](https://img.shields.io/badge/RAG-Hybrid%20Retrieval-orange)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Agent Workflow](#agent-workflow)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Support](#support)

---

## 🎯 Overview

AARA is a sophisticated Advanced autonomous research system powered by a **multi-agent architecture** that intelligently decomposes research queries, searches the web, ingests content, synthesizes findings, and generates peer-reviewed quality reports. The system uses **advanced RAG techniques** with hybrid retrieval, iterative refinement, and self-critique mechanisms to ensure research quality.

### What Problems Does It Solve?

- **Manual research bottleneck:** Automate the tedious process of finding, validating, and synthesizing information
- **Information overload:** Intelligently filter low-quality sources and aggregate relevant data
- **Citation tracking:** Maintain full provenance of sources with proper referencing
- **Iterative refinement:** Loop back on incomplete research with targeted follow-up searches

---

## 🏗️ Architecture

### System Topology
![alt text](architecture.png)

---

## ✨ Key Features

### 🧠 Intelligent Agent Orchestration
- **Intent Classification:** Gatekeeper routes queries with semantic understanding
- **Multi-Agent Coordination:** LangGraph-based state machine prevents hallucination
- **Self-Critique:** Synthesis agent evaluates drafts for completeness and accuracy

### 🔍 Advanced RAG System
- **Hybrid Retrieval:** BM25 (keyword) + Vector Search (semantic) for best-of-both-worlds
- **Parent-Child Chunking:** Intelligent document segmentation for context preservation
- **Vector Store Integration:** AWS RDS PostgreSQL with pgvector for scalable embeddings

### 🌐 Web Intelligence
- **Smart Web Scraping:** Handles multiple content types (HTML, PDF, Markdown)
- **Domain Filtering:** Targeted searches within authoritative sources
- **Failure Recovery:** Gracefully handles dead links and inaccessible content

### 📊 Report Generation
- **Structured Output:** Markdown reports with self-contained structure
- **PDF Export:** Server-side compilation with proper formatting
- **Citation Tracking:** Full source attribution and URL references
- **Iterative Refinement:** Loops back for missing information up to max iterations

### ⚡ Production-Ready
- **Containerization:** Docker support for consistent deployments
- **Streaming API:** Server-Sent Events for real-time telemetry
- **CORS Enabled:** Cross-origin requests for frontend integration
- **Error Handling:** Graceful degradation and detailed logging

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | Multi-agent state machine & routing |
| **LLM Framework** | LangChain | Prompt management & chains |
| **API Server** | FastAPI + Uvicorn | REST API & streaming responses |
| **Vector DB** | PostgreSQL + pgvector | Embeddings storage at scale |
| **Embeddings** | HuggingFace (BAAI/bge-small-en-v1.5) | Semantic representation |
| **Search** | Tavily API | Web search with domain filters |
| **LLM Models** | Google Gemini 1.5 Flash | Fast, reliable inference |
| **Web Scraping** | BeautifulSoup4, PyMuPDF, Readability | Content extraction |
| **Text Splitting** | LangChain Splitters | Intelligent chunking |
| **Report Generation** | Markdown + markdown-pdf | PDF compilation |
| **Frontend** | Vanilla JS + React-Zero | Interactive dashboard |
| **Deployment** | Docker + Vercel | Containerized backend + serverless frontend |

---

## 📦 Prerequisites

### System Requirements
- **Python:** 3.10+
- **PostgreSQL:** 13+ (with pgvector extension)
- **Memory:** 8GB RAM minimum
- **Storage:** 2GB for vector embeddings cache

### API Keys Required
- **Google Generative AI:** For Gemini LLM access
- **Tavily Search:** For web search capability
- **HuggingFace Token:** For embedding models

---

## 🚀 Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/Krdhirendra/Advanced-Autonomous-Research-agent
cd A_ARA
```

### 2. Create Virtual Environment
```bash
python -m venv aienv
source aienv/Scripts/activate  # On Windows: aienv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:

```env
# LLM Configuration
GOOGLE_API_KEY=your_google_api_key

# Search & Data Retrieval
TAVILY_API_KEY=your_tavily_api_key
HF_TOKEN=your_huggingface_token

# Database Configuration
AWS_RDS_URI=postgresql+psycopg2://user:password@host:5432/dbname?sslmode=require

# Optional
DEBUG=false
MAX_ITERATIONS=3
```

### 5. Initialize Database
```bash
# Ensure PostgreSQL has pgvector extension installed
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Create database and tables (handled by application on first run)
```

### 6. Start the Application
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

**Backend URL:** `http://localhost:8000`

---

## ⚙️ Configuration

### Environment Variables Reference

```ini
# Core LLM Configuration
GOOGLE_API_KEY              # Google Generative AI API key
TEMPERATURE                 # LLM temperature (0.0-1.0, default: 0)
MAX_ITERATIONS             # Max research loops (default: 3)

# Vector Store
AWS_RDS_URI                # PostgreSQL connection string with pgvector
EMBEDDING_MODEL            # Model name (default: BAAI/bge-small-en-v1.5)
EMBEDDING_DEVICE           # 'cpu' or 'cuda' (default: cpu)

# Search
TAVILY_API_KEY            # Web search API
SEARCH_MAX_RESULTS        # Results per search query (default: 3)

# Report Generation
REPORT_FOLDER             # Output directory for PDFs (default: ./reports)
PDF_TIMEOUT               # PDF generation timeout in seconds (default: 30)
```

### Tuning Parameters

**For Better Quality:**
```python
MAX_ITERATIONS = 4           # More refinement loops
CHUNK_SIZE = 1500           # Larger context windows
MAX_SEARCH_RESULTS = 5      # More sources per query
```

**For Faster Processing:**
```python
MAX_ITERATIONS = 2           # Fewer loops
CHUNK_SIZE = 500            # Smaller chunks
EMBEDDING_DEVICE = "cuda"   # Use GPU if available
```

---

## 📂 Project Structure

```
A_ARA/
├── 📄 server.py                    # FastAPI application & endpoint handlers
├── 📄 orchestrator_1.py            # LangGraph state machine & agent routing
├── 📄 RAG.py                       # Document chunking (parent-child strategy)
├── 📄 retriever.py                 # Hybrid retrieval (semantic + BM25)
├── 📄 vector_store.py              # PostgreSQL pgvector wrapper
├── 📄 tools.py                     # External tool integrations (Tavily, scraping)
├── 📄 requirements.txt             # Python dependencies
├── 🐳 Dockerfile                   # Container image definition
├── .env                            # Configuration (never commit to git)
│
├── 👥 agents/                      # Multi-agent system
│   ├── __init__.py
│   ├── gate_keeper.py             # Intent classification node
│   ├── supervisor.py              # Search strategy generation
│   ├── web_ingestion.py           # Content scraping & indexing
│   ├── synthesis_n_critic.py      # Report generation & evaluation
│   └── prompts.py                 # Centralized LLM prompts
│
├── 🎨 frontend/                    # Interactive dashboard
│   ├── index.html                 # Main page
│   ├── script.js                  # Event handling & WebSocket
│   └── style.css                  # Styling
├── 🎨 reports/                    # Example reports generated by AARA
│   ├── report.pdf
│
├── 📚 archive/                     # Legacy code (development history) (will add)
└── README.md                       # This file
```

---

## 🔌 API Reference

### Base URL
```
POST http://localhost:8000/api/research
```

### Endpoint: Research Request

**Request:**
```json
{
  "query": "Compare thermal mass properties of compressed stabilized earth blocks versus fired clay bricks"
}
```

**Response (Server-Sent Events):**
```json
data: {"type": "status", "message": "Gatekeeper: Classifying query..."}
data: {"type": "searches", "queries": [...]}
data: {"type": "ingestion", "sources": 5}
data: {"type": "draft", "content": "..."}
data: {"type": "complete", "report_url": "/reports/AARA_Report_12345.pdf"}
```

### Response Types

| Type | Description | Payload |
|------|-------------|---------|
| `status` | Agent status update | `{message: string}` |
| `searches` | Search queries generated | `{queries: Array}` |
| `ingestion` | Documents ingested | `{sources: number, urls: Array}` |
| `draft` | Report draft ready | `{content: string}` |
| `missing_info` | Incomplete research detected | `{gaps: Array}` |
| `complete` | Research finished | `{report_url: string}` |

---

## 🔄 Agent Workflow Details

### Phase 1: Gatekeeper (Intent Classification)

```python
Input:  "What is photosynthesis?"
Output: 
  - classification: "general_knowledge"
  - direct_response: "Photosynthesis is..."
  - reasoning: "Standard biology fact"
```

**Routing:**
- ✅ `general_knowledge` → Return direct response
- ✅ `chit_chat` → Friendly greeting
- ❌ `out_of_scope` → Polite refusal
- 🔄 `deep_research` → Continue to Supervisor

### Phase 2: Supervisor (Search Strategy)

Generates 3-5 diverse search queries with:
- Domain targeting (e.g., academic, industry)
- Keyword variations
- Synonym expansion
- Temporal filters when relevant

```python
Input:  "Analyze renewable energy adoption in Nordic countries"
Output:
  - "Nordic renewable energy statistics 2024"
  - "Sweden wind power infrastructure"
  - "Norway hydroelectric generation capacity"
  - "Denmark offshore wind farms"
```

### Phase 3: Ingestion (Web Search + RAG)

For each search query:
1. Execute Tavily search (3 results per query)
2. Extract content (HTML → text, PDF parsing)
3. Chunk documents (Parent: 1500 chars, Child: 300 chars)
4. Generate embeddings (HuggingFace BAAI/bge-small-en-v1.5)
5. Store in PostgreSQL pgvector

### Phase 4: Synthesis & Critique

1. **Retrieve Context:** Hybrid search (BM25 + semantic)
2. **Draft Report:** LLM synthesizes findings with citations
3. **Self-Evaluation:** Check for missing information
4. **Iterate:** Loop back to Supervisor if incomplete (max 3 iterations)
5. **Finalize:** Generate PDF report with references

---

## 👨‍💻 Development

### Running in Development Mode

```bash
# Terminal 1: Start backend
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Serve frontend (optional local testing)
python -m http.server 3000 --directory frontend
```

Visit: `http://localhost:3000`

### Testing the API

```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are vector databases?"}'
```

### Adding New Agents

1. Create `agents/my_agent.py`
2. Implement agent function: `def my_agent(state: dict, llm, embed_model) -> dict`
3. Register in `orchestrator_1.py`:
   ```python
   graph.add_node("my_node", partial(my_agent, ...))
   graph.add_edge(previous_node, "my_node")
   ```

### Debugging

Enable verbose logging:
```python
# In orchestrator_1.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check execution state:
```python
# Print state at any node
print(f"Current state: {json.dumps(state, indent=2)}")
```

---

## 🐳 Deployment

### Docker Build & Run

```bash
# Build image
docker build -t aara:latest .

# Run container
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=xxx \
  -e AWS_RDS_URI=xxx \
  -e TAVILY_API_KEY=xxx \
  aara:latest
```

### Production Checklist

- [ ] Rotate all API keys in `.env`
- [ ] Set `DEBUG=false`
- [ ] Configure PostgreSQL SSL (`sslmode=require`)
- [ ] Set up error logging/monitoring
- [ ] Configure CORS for your domain
- [ ] Use environment-specific `.env` files
- [ ] Run behind reverse proxy (nginx/Caddy)
- [ ] Enable request rate limiting

### Deploying to Vercel (Frontend)

```bash
# Frontend is pre-deployed at:
# https://aara-frontend.vercel.app/

# To redeploy after changes:
vercel deploy --prod
```

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** changes: `git commit -m "Add feature: description"`
4. **Push** to branch: `git push origin feature/my-feature`
5. **Submit** a Pull Request

### Code Standards

- Follow PEP 8 for Python code
- Add docstrings to all functions
- Keep agent functions pure (no side effects)
- Test API changes with curl or Postman

---

## 📝 License

This project is open source. See LICENSE file for details.

---

## 💬 Support

**Issues?** Please reach out:
- 📧 Email: [krdhirendra2006@gmail.com](mailto:krdhirendra2006@gmail.com)
- 🌐 Live Demo: [aara-frontend.vercel.app](https://aara-frontend.vercel.app/)
- 📋 GitHub Issues: [Create an issue](https://github.com/your-repo/issues)

---

## 🙏 Acknowledgments

Built with:
- [LangGraph](https://langchain.com/langgraph) - Multi-agent orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [PostgreSQL + pgvector](https://pgvector.org/) - Vector database
- [Google Generative AI](https://ai.google.dev/) - LLM backbone
- [Tavily API](https://tavily.com/) - Web search

---

<div align="center">

**Made with ❤️ for researchers and knowledge seekers**

[⬆ Back to top](#-aara-advanced-autonomous-research-agent)

</div>
