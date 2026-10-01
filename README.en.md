# Local Agent KB

<div align="center">

**A local knowledge base that gives your AI assistant private-document retrieval**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green)](https://modelcontextprotocol.io/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%201.13-red)](https://qdrant.tech/)
[![Tools](https://img.shields.io/badge/MCP%20Tools-11-orange)](./kb_mcp_server.py)
[![Tests](https://img.shields.io/badge/acceptance-T1--T14-brightgreen)](./本地知识库开发计划_20261001.md)

</div>

---

## 📖 Overview

**Local Agent KB** is a local knowledge base built on the **MCP (Model Context Protocol)**. It gives AI coding assistants **hybrid retrieval** over private documents: every retrieved snippet carries its channel, trust level, applicable version, document date and image paths, so the model can answer *and* cite evidence.

The goal in one sentence: **turn "what the docs say" into "evidence the AI can quote"**. Beyond vector search that means a metadata spec (v2), version filtering with drift warnings, an image pipeline (captions + image return), a knowledge-ingestion path (draft → validate → publish), and a reproducible acceptance regression.

### ✨ Key Features

- 🔀 **Four-channel hybrid retrieval** — dense (0.7) + sparse BM42 (0.3) + exact string (1.0) + term-anchor (1.0), client-side weighted RRF, issued concurrently (latency overhead <30%)
- 🧭 **Metadata spec v2** — five required front-matter keys (`title` / `applies_to.framework` / `doc_date` / `trust` / `canonical`); missing values are down-weighted, never filtered (R1); filters for `version` / `model` / `trust` / `kind` / `section` / `path_prefix` plus `⚠version-drift` warnings
- 🖼️ **Image pipeline** — links URL-decoded so local images are reachable; five-level caption extraction (explicit marker / alt / adjacent prose / heading path / filename) **embedded into the index**; top-1 snippet returns its **main figure** as a thumbnail
- 📚 **Knowledge types & ingestion** — 8 templates (fact card / signature table / error FAQ / defect log / enum examples / QLP snippet / version matrix / asset pointer) with a draft → validate → publish loop
- 🔍 **Purpose-built tools** — general search, full-document read, section outline, error-string lookup, document inventory, metadata lint, duplicate inspection
- ⚡ **Incremental indexing** — content hash + parser-version gating; models load on demand and release when idle
- 🛡️ **Data-quality guards** — model health check, zero-vector monitoring (pre-check / in-batch / full scan), segment health monitoring
- 🔒 **Fully local** — data never leaves the machine; only Qdrant runs in Docker (bound to loopback)

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────┐
│           AI coding assistant (Copilot / Kimi Code)       │
│                        (cloud LLM)                        │
│                            │                              │
│                    MCP Tool Call (stdio)                  │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│               kb_mcp_server.py (local process)            │
│                                                           │
│  query ──┬─► dense  (e5-large, 1024d) ─┐                  │
│          ├─► sparse (BM42)             ├─► weighted RRF   │
│          ├─► exact  (search_text)      │       │          │
│          └─► anchor (term df check)    ┘       ▼          │
│                                     post-process: TOC      │
│                                     penalty / trust /      │
│                                     recency / ≤2 per doc / │
│                                     image return           │
│                                               │            │
│         ┌─────────────────────┬───────────────┘            │
│         │  kb_schema.py       │  kb_templates.py           │
│         │  metadata v2/images │  8 templates / validation  │
│         └─────────────────────┴───────────────┐            │
└───────────────────────────────────────────────┼────────────┘
                                                │
                              ┌─────────────────▼──────────┐
                              │      Qdrant Server 1.13     │
                              │  dense(1024/Cosine) + bm42  │
                              │  Docker container / volume  │
                              └─────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  index_docs.py (indexer)                                  │
│  parse → chunk → embed → upsert (incremental) → scan      │
└──────────────────────────────────────────────────────────┘
```

### Tech stack

| Layer | Choice | Notes |
|-------|--------|-------|
| **Protocol** | MCP (stdio) | Standard transport between assistant and local tools |
| **Retrieval gateway** | Python MCP server (`kb_mcp_server.py`) | Orchestrates four channels, filters, post-processing, response rendering |
| **Dense vectors** | fastembed `intfloat/multilingual-e5-large` | 1024-d Cosine, local ONNX inference |
| **Sparse vectors** | fastembed `Qdrant/bm42-all-minilm-l6-v2-attentions` | Named vector `bm42`, same collection as dense |
| **Vector store** | Qdrant 1.13 (Docker) | Multi-vector search, payload filters, count aggregation |
| **Metadata layer** | `kb_schema.py` | front-matter v2 (nested YAML) / sidecar / image path resolution / lint |
| **Template layer** | `kb_templates.py` + `templates/` | 8 knowledge templates, ingest validation, draft files |
| **Parsers** | `document_parsers.py` (PyMuPDF / python-docx) | PDF tables → Markdown, header/footer filtering, caption extraction |

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (for Qdrant)
- Python 3.12 (this repo ships an embedded runtime at `python312\`)

> Commands below assume Windows and the bundled runtime: `python312\python.exe`.

### 1. Start Qdrant

```powershell
docker compose up -d
```

Data lives in a Docker **named volume** (`local_kb_qdrant_data`); ports are **loopback-only**:

```yaml
ports:
  - "127.0.0.1:6333:6333"   # REST API
  - "127.0.0.1:6334:6334"   # gRPC
```

> **Two traps worth knowing**
> 1. **Never switch back to a bind mount** (`./data:/qdrant/storage`): Windows' mmap write synchronisation is unreliable and Qdrant zeroed vector data during segment optimisation (historically triggered past a 12.57x expansion ratio). That is why **only snapshots are usable for backup** — see *Operations*.
> 2. Qdrant has **no authentication** by default, hence loopback-only binding. To expose it on a LAN, set `QDRANT__SERVICE__API_KEY` and mirror it in the MCP env (`QDRANT_API_KEY`).

### 2. Install dependencies

```powershell
python312\python.exe -m pip install -r requirements.txt
```

The embedding model (~2.1 GB) is downloaded on first run into `models/`. If HuggingFace is unreachable, use a mirror:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 3. Index your documents

```powershell
# incremental (only new / modified / deleted files)
python312\python.exe index_docs.py md-source

# full re-embed (after metadata or parser changes)
python312\python.exe index_docs.py md-source --reindex-all
```

Supported: `.md`, `.pdf`, `.docx`, `.txt`. Default source dir is `md-source/` (override with `KB_DOCS_DIR`).

### 4. Configure the MCP server

Point your assistant's MCP config at this repo (example for CodeBuddy / Kimi `mcp.json`):

```json
{
  "mcpServers": {
    "local-agent-kb": {
      "command": "E:\\Local_agent_kb\\python312\\python.exe",
      "args": ["E:\\Local_agent_kb\\kb_mcp_server.py"],
      "env": {
        "KB_USE_GPU": "0",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "KB_COLLECTION": "emulate3d_docs",
        "KB_IDLE_TIMEOUT": "600",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

> **After editing `kb_mcp_server.py` you must restart the MCP server** (the process does not hot-reload).

### 5. Verify connectivity

```powershell
# launches the server the same way the MCP client does, then does a handshake + tools/list
python312\python.exe tools\probe_mcp_stdio.py
```

Expected: `OK: MCP 握手与 tools/list 均正常` plus 11 tools.

---

## 🧩 MCP Tools (11)

### Retrieval

| Tool | Purpose | Key params |
|------|---------|------------|
| `search_tech_kb` | Main hybrid retrieval entry | `query`, `mode` (hybrid/semantic/exact), `version`, `model`, `trust`, `kind`, `section`, `path_prefix`, `canonical_only`, `top_k`, `score_threshold`, `attach_image` |
| `kb_get_doc` | Full continuous text of a document or section | `source`, `heading`, `budget` (default 3000 chars) |
| `kb_outline` | Section tree (use before reading a long doc) | `source`, `max_depth` |
| `kb_error_lookup` | Error-string lookup (root cause + fix pattern) | `error_text`, `top_k`, `exact_only` |

### Inventory

| Tool | Purpose |
|------|---------|
| `kb_stats` | Size / type / trust distribution, vector config, governance metrics; full scan only with `detail=true` |
| `kb_list_docs` | Document inventory with metadata (chunks / version / date / trust / canonical) |
| `kb_meta_lint` | Metadata health: missing required keys, un-backfilled index fields, `canonical_uid` conflicts |
| `kb_caption` | Caption governance: extract captions / report whether they reached the vectors |
| `kb_dupes` | Duplicate inspection (hash / name / text) with canonical suggestion |

### Ingestion & governance

| Tool | Purpose |
|------|---------|
| `kb_note` | Register a failed retrieval / pitfall as a structured draft in `kb-inbox/drafts/` |
| `kb_ingest` | Publish a draft: `kind` + `payload`/`path` + `meta`; **dry-run first** to see the validation report |

### `search_tech_kb` response contract

- Score semantics: `hybrid` returns an **RRF fusion score** (≈0.5 for a single channel, 1.0 for two — **not cosine similarity**); `semantic` returns cosine.
- `score_threshold`: applies to final results in `semantic`; in `hybrid` it only gates the dense pre-filter (sparse/anchor/exact are unaffected) — **do not pass high values like 0.65**.
- Every snippet is annotated with channels, trust weight, applicable version (`⚠version-drift` means it differs from the requested version), document date (`(inferred)` = fell back to file time) and canonical flag.
- `attach_image=true` (default): if the top-1 snippet has local images, a thumbnail of its **main figure** (largest local image, ≤512 px JPEG) is attached; plain-text clients degrade to a path list automatically.

---

## 🎯 Retrieval mechanics

### Four-channel weighted fusion

```
score = Σ w_c · 1/(k + rank_c)        k = 1 (matches Qdrant's built-in RRF behaviour)
w: dense 0.7 | sparse 0.3 | exact 1.0 | anchor 1.0
```

- **dense** — semantic recall (e5-large)
- **sparse** — BM42, recovers keyword / identifier recall
- **exact** — `search_text` exact string (error text, class names, control syntax; normalised quotes/whitespace)
- **anchor** — when the query contains an explicit identifier (e.g. `RackConfigurator`) with df ≤ 60, literal hits are weighted up

### Post-processing

| Rule | Value | Notes |
|------|-------|-------|
| Candidate pool | `top_k × 3` | Pre-post-processing pool |
| TOC penalty | `×0.6` **and never in top-3** | Hard constraint: keep recall, lose the best slots |
| Trust weight | `measured 1.0` / `tutorial 0.9` / `pending 0.75` | Missing → `pending` |
| Recency decay | linear over 6 months → floor `0.85` | **Conclusion-type `kind` only** (`note`/`defect_log`/`error_faq`/`model_fact_card`/`version_matrix`); tutorials / manuals / API references stay at 1.0 (their staleness is expressed by version filters); `date_inferred` always exempt |
| Per-document cap | `≤2` snippets | Prevent one document from dominating |
| Low-confidence hint | single-channel support <0.5 (hybrid); cosine <0.60 (semantic) | Suggests `kb_note` feedback |

---

## 📐 Metadata spec v2

### Required keys and fallbacks

`title`, `applies_to.framework`, `doc_date`, `trust`, `canonical`. Missing values fall back per **R1 (down-weight, never filter)**: `unknown` / file time + `date_inferred` / `pending`; `kb_meta_lint` reports the gaps.

```yaml
---
title: Conveyor Properties
source-id: demo3d_2026/conveyors_properties
source-url: https://store.sim3d.com/demo3d_2026/conveyors_properties
fetched: 2026-09-26          # crawl date (original export field, kept)
doc_date: 2026-09-26         # document date (≤ conclusion date)
trust: tutorial              # measured | tutorial | pending
canonical: true              # true = canonical; false needs superseded_by
source_origin: web           # kb | plugin | web | decompile
applies_to:
  framework: "2026"          # applicable framework version (version filter + drift warning)
  model: general             # applicable machine/controller (general = any)
kind: doc
---
```

> **`.md` metadata lives in the document's own front-matter** (sidecar `<doc>.meta.yaml` applies to PDF/DOCX/TXT only). After editing, run `tools/backfill_payload.py --all-in-state` (seconds, **no re-embedding**) to make it live.

### Knowledge types (8 + draft)

`model_fact_card`, `part_signature`, `version_matrix`, `error_faq`, `defect_log`, `enum_examples`, `qlp_snippet`, `asset_pointer`, `draft` — each with default `trust`/`source_origin` and a fixed section skeleton in `templates/`.

---

## 🖼️ Image pipeline

1. **Reachable paths** — relative links must be URL-decoded (`%20` etc.); ≈15% reachable without decoding vs 100% with it.
2. **Five-level captions** (`document_parsers._image_caption`): explicit marker (`图注：…` / `<!-- caption: … -->`) → `alt` → adjacent prose → heading path → filename.
3. **Captions are embedded** — injected as a `[图注] …` line into the chunk text, so *what the figure says* becomes searchable (parser version `2.2`+).
4. **Image return** — thumbnail of the top-1 snippet's main figure (largest local image, ≤512 px JPEG; tune with `KB_THUMB_*`).
5. **External-image recovery** — `tools/fetch_site_images.py` pulls externally-linked images from the site's raw-content API into local `assets/`, rewriting the markdown to relative paths (with dry-run / backup / rollback).

---

## 📂 Project layout

```text
.
├── kb_mcp_server.py          # MCP server: four channels + 11 tools + post-processing
├── kb_schema.py              # Metadata v2: front-matter, lint, payload mapping, image paths
├── kb_templates.py           # Template layer: render / validate / draft files
├── document_parsers.py       # Parsers: MD / PDF / DOCX / TXT (incl. captions)
├── index_docs.py             # Batch indexer (incremental + quality guards)
├── index_docs.bat            # Windows shortcut
├── docker-compose.yml        # Qdrant (named volume / loopback binding)
├── requirements.txt
├── index_state.json          # Full-index state (generated)
├── kb_ingest_state.json      # Single-file ingest state (generated)
├── templates/                # 8 knowledge templates
├── kb-inbox/                 # Drafts and pending artifacts
├── md-source/                # Default document source (incl. assets/ images)
├── tools/                    # Delivered tools and smoke tests
│   ├── normalize_frontmatter.py   # Bulk front-matter fill (insert-only / backup / rollback)
│   ├── backfill_payload.py        # Metadata backfill (set_payload; --sync prunes stale keys)
│   ├── fetch_site_images.py       # External-image recovery
│   ├── audit_images.py            # Image audit (no Qdrant needed)
│   ├── ensure_schema.py           # Collection & payload index creation
│   ├── probe_mcp_stdio.py         # MCP handshake + tools/list probe
│   └── smoke_p1..p4.py            # Per-phase smoke tests
├── regression/               # Acceptance regression
│   ├── queries.jsonl              # Fixed query set (24)
│   ├── eval.py / eval_p2.py       # Metrics (dense / four-channel)
│   ├── summarize.py               # Result summariser
│   ├── baseline_20261001.md       # Baseline report
│   └── results/                   # Per-run results (incl. baseline.json / latest.json)
├── backup/                   # Snapshots and rollback backups
└── logs/                     # Index logs (generated)
```

---

## ⚙️ Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant endpoint |
| `KB_COLLECTION` | `emulate3d_docs` | Collection name |
| `KB_DOCS_DIR` | `<repo>/md-source` | Default document source |
| `KB_ASSETS_ROOT` | `<repo>/md-source` | Asset root (HTML `images/x.jpg` reverse lookup) |
| `KB_SITE_BASE` | `https://store.sim3d.com` | Domain used to complete site-relative image links |
| `KB_INBOX_DIR` | `<repo>/kb-inbox` | Drafts / pending artifacts root |
| `KB_CANONICAL_HINT` | `local_agent_kb` | Canonical heuristic (path containing this wins) |
| `KB_DUPES_EXTRA_DIRS` | `E:\kimi code workbentch` | Extra dirs scanned by `kb_dupes` (never indexed) |
| `KB_USE_GPU` | `0` | `1` enables CUDA (ONNX Runtime) |
| `KB_IDLE_TIMEOUT` | `600` | Model idle release (seconds; `0` = never) |
| `KB_CHUNK_MAX_CHARS` | `1200` | Max characters per chunk |
| `KB_DOC_BUDGET` | `3000` | `kb_get_doc` character budget |
| `KB_THUMB_MAX_SIDE` / `KB_THUMB_QUALITY` | `512` / `80` | Thumbnail size / JPEG quality |
| `KB_DF_CACHE_TTL` | `600` | Anchor-channel df count cache (seconds) |

---

## 🛠️ Operations & troubleshooting

### Common Docker commands

```powershell
cd e:\Local_agent_kb
docker compose up -d              # start / recreate (after editing compose)
docker compose stop|start qdrant  # stop / start (data preserved)
docker logs --tail 50 local_kb_qdrant
```

> Docker may be shared with other services on this machine — restarting Docker affects them too.

### Backup (snapshots only)

```powershell
# 1) ask Qdrant to create a snapshot
curl.exe -X POST "http://localhost:6333/collections/emulate3d_docs/snapshots"
# 2) copy it out of the container (the named volume lives inside the WSL2 VM)
docker cp "local_kb_qdrant:/qdrant/snapshots/emulate3d_docs/<snapshot-name>" "E:\Local_agent_kb\backup\"
```

### Troubleshooting

| Symptom | Check |
|---------|-------|
| `MCP error -32000: Connection closed` | Usually an **import failure**, not a protocol issue. This repo runs on **embedded Python** (`python312._pth`), where `sys.path` **excludes the script directory** — new top-level modules must call `sys.path.insert(0, <script dir>)`. Diagnose with `tools\probe_mcp_stdio.py` |
| Edits to `kb_mcp_server.py` have no effect | The MCP process does not hot-reload — **restart the MCP server** |
| New metadata not visible | Run `tools\backfill_payload.py --all-in-state` first (payload is instant); only text/image changes need re-embedding |
| Images not visible | Verify they are local (external URLs are never returned); check `kb_caption` for "captions in vectors" |
| Single-file vs full-index state mismatch | `index_docs.py` uses `index_state.json`; `kb_ingest`/`kb_caption` use `kb_ingest_state.json` — by design |

---

## ✅ Quality assurance

### Parser & index guards

- Model health check (warm-up, non-zero vectors), **zero-vector monitoring** (>10% in batch → rebuild session), **pre-index quality check**, **full zero-vector scan**;
- Optimiser safety (`indexing_threshold=50000`, `max_optimization_threads=1`), session rebuild every 5000 chunks, segment-count health monitoring;
- Deletion protection: aborts automatic cleanup when >80% of indexed files disappear at once.

### Acceptance regression

```powershell
python312\python.exe regression\eval_p2.py --compare baseline   # four-channel (authoritative)
python312\python.exe regression\eval.py    --compare baseline   # dense only
python312\python.exe tools\smoke_p4.py --commit                 # phase smoke test (auto rollback)
```

- Fixed query set of **24** (`regression/queries.jsonl`: API usage, framework behaviour, error strings, version differences, model facts, gap probes);
- Metrics: `MRR@5`, `Recall@5`, score gradient, TOC-at-top-1 count, per-document ratio, low-confidence count, latency, plus **T1–T14** assertions (version drift, caption retrieval, image-return degradation, recency scope, …);
- Full acceptance record (with measured values and corrections) lives in `本地知识库开发计划_20261001.md`.

---

## 🤝 Contributing

Issues and PRs welcome. Suggested directions:

- Phase 2 `kb_caption(mode=summary)`: use a VLM to write **real captions** (today only rule-based "weak captions")
- Better time/version signals: enable finer recency policies once true publish dates are available
- Grow the evaluation set: feed production failure queries back into `regression/queries.jsonl`

---

## 📄 License

MIT. This repository **does not ship a `LICENSE` file yet** — add the MIT text with your copyright line if you need a formal declaration.

---

<div align="center">
Made with ❤️ for the AI-assisted development community
</div>
