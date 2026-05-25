# Local Agent KB

<div align="center">

**Local Knowledge Base — Empower AI Assistants with Your Private Document Retrieval**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green)](https://modelcontextprotocol.io/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

</div>

---

## 📖 Overview

**Local Agent KB** is a local knowledge base system built on the **MCP (Model Context Protocol)**. It provides semantic search over private documents for AI coding assistants such as GitHub Copilot and Kimi Code.

By vectorizing local documents (Markdown, PDF, Word, TXT) and storing them in a Qdrant vector database, AI assistants can retrieve relevant knowledge snippets in real-time during conversations — enabling **intelligent Q&A based on your proprietary technical materials**.

### ✨ Key Features

- 🚀 **MCP Native** — Built on the Model Context Protocol standard, seamlessly compatible with any MCP-enabled AI assistant
- 📄 **Multi-format Support** — Markdown (.md), PDF (.pdf), Word (.docx), Plain Text (.txt)
- ⚡ **Incremental Indexing** — Only processes new or modified files; re-runs complete in seconds
- 🧠 **Semantic Search** — Powered by multilingual-e5-large, supporting both Chinese and English queries
- 💾 **On-Demand Model Loading** — Embedding model auto-releases during idle to save memory
- 📊 **Rich Progress Display** — Real-time progress bars and timing statistics during indexing
- 🐳 **Minimal Dependencies** — Only Qdrant runs in Docker; all other components are pure Python
- 🔒 **100% Local** — Your data never leaves your machine; no internet connection required

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────┐
│     AI Assistant (Copilot / Kimi Code)       │
│              (Cloud LLM)                     │
│                   │                          │
│         MCP Tool Call (stdio)               │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│      kb_mcp_server.py (Local Process)       │
│  ┌─────────────┐    ┌─────────────────────┐  │
│  │  fastembed   │    │   Qdrant Client     │  │
│  │ (ONNX)       │    │  (Vector Search)    │  │
│  └──────┬──────┘    └────────┬────────────┘  │
└─────────┼────────────────────┼───────────────┘
          │                    │
          │           ┌────────▼────────┐
          │           │  Qdrant Server  │
          │           │  (Docker)        │
          │           │  named volume   │
          │           └─────────────────┘
          │
┌─────────▼────────────────────────────────────┐
│      index_docs.py (Indexing Script)         │
│   Parse → Vectorize → Upsert (Incremental)   │
│   Pre-index Quality Check → Full Zero Scan   │
└──────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Choice | Description |
|-------|--------|-------------|
| **Protocol** | MCP (stdio) | Standard communication protocol between AI assistants and local tools |
| **Gateway** | Python MCP Server | Receives MCP requests, orchestrates embedding + vector search |
| **Embedding** | fastembed (multilingual-e5-large) | Text → 1024-dim vector, pure ONNX local inference |
| **Vector Store** | Qdrant | Semantic search, metadata filtering, Top-K retrieval |
| **Document Parsing** | PyMuPDF / python-docx | PDF table→Markdown conversion, header/footer filtering |
| **Data Sources** | MD / PDF / DOCX / TXT | Technical docs, specification manuals, logs, etc. |

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (for running Qdrant)
- Python 3.10+

### 1. Start Qdrant

```bash
docker compose up -d
```

This starts the Qdrant vector database with data persisted in a Docker named volume (`qdrant_data`).

> **Windows Users**: This project uses a Docker named volume instead of a bind mount to avoid data corruption caused by unreliable mmap write synchronization on the Windows filesystem.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The first run will automatically download the embedding model (`intfloat/multilingual-e5-large`, ~2.1GB). Make sure you have internet access. The model is cached in `models/` and subsequent runs work fully offline.

### 3. Index Your Documents

```bash
python index_docs.py /path/to/your/docs/folder
```

This recursively scans the folder, extracts document content, vectorizes it, and stores it in Qdrant.

**Supported formats**: `.md`, `.pdf`, `.docx`, `.txt`

> **Tip**: Re-running `index_docs.py` automatically detects added, modified, or deleted files and only processes the changes.

### 4. Start the MCP Server

```bash
python kb_mcp_server.py
```

On successful startup, you'll see:
```
[KB-MCP] Service started (model loads on demand, auto-releases when idle)
```

Then configure the MCP server in your AI assistant.

---

## 🧩 MCP Tools

### `search_tech_kb` — Semantic Knowledge Retrieval

Searches the knowledge base for document snippets most relevant to your query.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | — | Search keywords (2–5 core terms) |
| `doc_type` | string | ❌ | `all` | Document type filter: `api_reference`, `user_manual`, `tutorial`, `design_spec`, `all` |
| `top_k` | integer | ❌ | `5` | Number of snippets to return |

**Example**:
```
search_tech_kb(query="REST API authentication flow", top_k=3)
```

### `get_class_api` — Precise Class/Method API Lookup

Retrieves API documentation for a specific class or method.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `class_name` | string | ✅ | Class name |
| `method_name` | string | ❌ | Method name (optional) |

**Example**:
```
get_class_api(class_name="HttpClient", method_name="PostAsync")
```

---

## 📂 Project Structure

```
.
├── kb_mcp_server.py          # MCP Server (on-demand model loading + idle release)
├── index_docs.py             # Batch indexing script (incremental + data quality safeguards)
├── document_parsers.py       # Document parsers (PDF / Word / TXT / MD)
├── index_docs.bat            # Windows quick-index batch script
├── docker-compose.yml        # Qdrant Docker orchestration (named volume persistence)
├── requirements.txt          # Python dependencies
├── index_state.json          # Index state (auto-generated, for incremental updates)
├── models/                   # Embedding model cache (auto-downloaded)
├── logs/                     # Indexing run logs (auto-generated)
└── md-source/                # Default document source directory (configurable)
```

---

## ⚙️ Configuration

Runtime behavior is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server address |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `KB_COLLECTION` | `emulate3d_docs` | Qdrant collection name |
| `KB_USE_GPU` | `0` | Enable GPU acceleration (set to `1` for CUDA) |
| `KB_IDLE_TIMEOUT` | `600` | Model idle release timeout (seconds); `0` to disable |

---

## 🛠️ Document Parsing

### Supported Formats

| Format | Engine | Features |
|--------|--------|----------|
| **Markdown** | Native | Split by H1–H4 headings, code block protection, short fragment merging |
| **PDF** | PyMuPDF | Reading-order extraction, table→Markdown conversion, auto header/footer filtering |
| **Word** | python-docx | Split by headings, table→Markdown conversion, image placeholders |
| **TXT** | Native | Split by blank lines, preserves indentation and code blocks |

### Incremental Indexing

`index_docs.py` tracks every file's MD5 hash and modification time in `index_state.json`:

1. **New files** → Parsed, vectorized, and upserted
2. **Modified files** → Old vectors deleted → re-parsed and upserted
3. **Deleted files** → Corresponding vectors cleaned up automatically
4. **Unchanged files** → Skipped, completes in seconds

> **Safeguard**: When more than 80% of indexed files disappear from the source directory, automatic cleanup is blocked to prevent accidental data loss.

### Data Quality Safeguards

The indexing script includes multi-layer protection to ensure vector data integrity:

| Check | Description |
|-------|-------------|
| **Model Health Check** | Pre-index warmup verification to confirm the embedding model outputs non-zero vectors |
| **Batch Zero-Vector Monitor** | Real-time detection during embedding; auto-rebuilds session if zero-vector ratio exceeds 10% |
| **Pre-index Quality Check** | Samples existing data before incremental indexing; alerts and blocks if zero vectors are found |
| **Full Zero-Vector Scan** | Scrolls all vectors after indexing completes; only passes if no zero vectors are found |
| **Optimizer Safety Config** | `indexing_threshold=50000` to reduce segment merge frequency; `max_optimization_threads=1` |
| **Periodic Session Rebuild** | Rebuilds embedding session every 5000 chunks to prevent GPU state accumulation |
| **Segment Health Monitor** | Alerts when segment count exceeds threshold, warning of storage bloat risk |

---

## 🤝 Contributing

Issues and Pull Requests are welcome!

Potential development directions:
- Support more document formats (HTML, EPUB, etc.)
- Add a Web UI for document preview
- Support multi-collection / multi-knowledge-base switching
- Improve chunking strategy for large documents

---

## 📄 License

[MIT](LICENSE)

---

<div align="center">
Made with ❤️ for the AI-assisted development community
</div>
