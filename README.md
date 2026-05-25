# Local Agent KB

<div align="center">

**本地知识库 —— 让 AI 助手拥有你的私有文档检索能力**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green)](https://modelcontextprotocol.io/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

</div>

---

## 📖 概述

**Local Agent KB** 是一个基于 **MCP (Model Context Protocol)** 协议的本地知识库系统，为 AI 编程助手（如 GitHub Copilot、Kimi Code 等）提供私有文档的语义检索能力。

通过将本地文档（Markdown、PDF、Word、TXT）向量化后存入 Qdrant 向量数据库，AI 助手可以在对话中实时检索相关知识片段，实现**基于自有技术资料的智能问答**。

### ✨ 核心特性

- 🚀 **MCP 原生支持** — 遵循 Model Context Protocol 标准，无缝对接支持 MCP 的 AI 编程助手
- 📄 **多格式文档支持** — Markdown (.md)、PDF (.pdf)、Word (.docx)、纯文本 (.txt)
- ⚡ **增量索引** — 仅处理新增/修改的文件，重复运行秒级完成
- 🧠 **语义检索** — 基于 multilingual-e5-large 模型，支持中英文多语义搜索
- 💾 **按需加载模型** — Embedding 模型空闲自动释放，避免常驻内存
- 📊 **Rich 进度展示** — 索引过程实时显示进度条、耗时统计
- 🐳 **最小依赖** — 仅 Qdrant 运行于 Docker，其余均为纯 Python 进程
- 🔒 **完全本地** — 数据不出本机，无需外网连接

---

## 🏗️ 架构

```text
┌─────────────────────────────────────────┐
│        AI 编程助手 (Copilot / Kimi)      │
│              (云端 LLM)                  │
│                   │                      │
│         MCP Tool Call (stdio)           │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│      kb_mcp_server.py (本地进程)        │
│  ┌─────────────┐   ┌─────────────────┐  │
│  │  fastembed   │   │  Qdrant Client  │  │
│  │ (ONNX 推理)  │   │  (向量检索)      │  │
│  └──────┬──────┘   └────────┬────────┘  │
└─────────┼───────────────────┼───────────┘
          │                   │
          │          ┌────────▼────────┐
          │          │  Qdrant Server  │
          │          │  (Docker 容器)   │
          │          │  named volume   │
          │          └─────────────────┘
          │
┌─────────▼────────────────────────────────┐
│      index_docs.py (索引脚本)            │
│  解析 → 向量化 → 入库 (增量更新)         │
│  预索引质量检查 → 全量零向量扫描          │
└──────────────────────────────────────────┘
```

### 技术栈

| 层级 | 选型 | 说明 |
|------|------|------|
| **协议层** | MCP (stdio) | AI 助手与本地工具的标准通信协议 |
| **检索网关** | Python MCP Server | 接收 MCP 请求，编排 Embedding + 向量检索 |
| **向量化** | fastembed (multilingual-e5-large) | 文本 → 1024 维向量，纯 ONNX 本地推理 |
| **向量存储** | Qdrant | 语义检索、元数据过滤、Top-K 召回 |
| **文档解析** | PyMuPDF / python-docx | PDF 表格转 Markdown、页眉/页脚过滤 |
| **数据源** | MD / PDF / DOCX / TXT | 技术文档、规范手册、日志文本等 |

---

## 🚀 快速开始

### 前置条件

- [Docker](https://docs.docker.com/get-docker/) (用于运行 Qdrant)
- Python 3.10+

### 1. 启动 Qdrant

```bash
docker compose up -d
```

此命令将启动 Qdrant 向量数据库，数据持久化在 Docker named volume (`qdrant_data`) 中。

> **Windows 用户注意**: 项目已采用 Docker named volume 替代 bind mount，避免 Windows 文件系统对 mmap 写入同步不可靠导致的数据损坏问题。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**: 首次运行时会自动下载 Embedding 模型（`intfloat/multilingual-e5-large`，约 2.1GB），请确保网络通畅。模型会缓存到 `models/` 目录，后续使用无需联网。

### 3. 索引入库文档

```bash
python index_docs.py /path/to/your/docs/folder
```

该命令会递归扫描指定文件夹，提取文档内容并向量化后存入 Qdrant。

**支持的格式**: `.md`, `.pdf`, `.docx`, `.txt`

> **提示**: 重复运行 `index_docs.py` 会自动检测新增、修改或删除的文件，仅处理有变动的部分。

### 4. 启动 MCP Server

```bash
python kb_mcp_server.py
```

正常启动后终端输出：
```
[KB-MCP] 服务启动（模型按需加载，空闲自动释放）
```

然后在 AI 编程助手中配置 MCP 服务即可使用。

---

## 🧩 MCP 工具

### `search_tech_kb` — 语义检索知识库

在知识库中搜索与查询最相关的文档片段。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | ✅ | — | 搜索关键词（2-5 个核心词） |
| `doc_type` | string | ❌ | `all` | 文档类型过滤：`api_reference`, `user_manual`, `tutorial`, `design_spec`, `all` |
| `top_k` | integer | ❌ | `5` | 返回的片段数量 |

**示例**:
```
search_tech_kb(query="REST API 认证流程", top_k=3)
```

### `get_class_api` — 精确类/方法 API 检索

精确检索指定类或方法的 API 文档。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `class_name` | string | ✅ | 类名 |
| `method_name` | string | ❌ | 方法名（可选） |

**示例**:
```
get_class_api(class_name="HttpClient", method_name="PostAsync")
```

---

## 📂 项目结构

```
.
├── kb_mcp_server.py          # MCP Server 主程序（模型按需加载 + 空闲释放）
├── index_docs.py             # 批量索引脚本（增量更新 + 数据质量保障）
├── document_parsers.py       # 文档解析器（PDF / Word / TXT / MD）
├── index_docs.bat            # Windows 快捷索引脚本
├── docker-compose.yml        # Qdrant Docker 编排（named volume 持久化）
├── requirements.txt          # Python 依赖
├── index_state.json          # 索引状态（自动生成，增量更新用）
├── models/                   # Embedding 模型缓存（自动下载）
├── logs/                     # 索引运行日志（自动生成）
└── md-source/                # 默认文档源目录（可配置）
```

---

## ⚙️ 配置

通过环境变量配置运行时行为：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `QDRANT_HOST` | `localhost` | Qdrant 服务地址 |
| `QDRANT_PORT` | `6333` | Qdrant 服务端口 |
| `KB_COLLECTION` | `emulate3d_docs` | Qdrant 集合名称 |
| `KB_USE_GPU` | `0` | 启用 GPU 加速（设为 `1` 启用 CUDA） |
| `KB_IDLE_TIMEOUT` | `600` | 模型空闲释放时间（秒），`0` 为不释放 |

---

## 🛠️ 文档解析功能

### 支持的格式

| 格式 | 解析引擎 | 功能特性 |
|------|----------|----------|
| **Markdown** | 原生解析 | 按 H1-H4 标题分块，代码块保护，短片段合并 |
| **PDF** | PyMuPDF | 阅读顺序提取，表格→Markdown 转换，页眉/页脚自动过滤 |
| **Word** | python-docx | 按标题分块，表格→Markdown 转换，图片占位标记 |
| **TXT** | 原生解析 | 按空行分段，保留缩进和代码块 |

### 增量索引机制

`index_docs.py` 通过 `index_state.json` 记录每个文件的 MD5 哈希和修改时间，实现增量更新：

1. **新增文件** → 解析并向量化入库
2. **修改文件** → 删除旧向量 → 重新解析入库
3. **删除文件** → 自动清理对应向量数据
4. **未变更文件** → 跳过，秒级完成

> **安全保护**: 当超过 80% 的已索引文件在目录中消失时，阻止自动清理以防止误删。

### 数据质量保障

索引脚本内置多层防护机制，确保向量数据的完整性和可靠性：

| 检查项 | 说明 |
|--------|------|
| **模型健康检查** | 索引前热身验证，确认 embedding 模型输出非零向量 |
| **Batch 零向量监控** | 嵌入过程中实时检测，零向量比例 >10% 自动重建 session |
| **预索引质量检查** | 增量索引前抽样检测已有数据，发现零向量时告警阻止 |
| **全量零向量扫描** | 索引完成后 scroll 全部向量，确认无零向量才视为成功 |
| **优化器安全配置** | `indexing_threshold=50000`，降低段合并频率；`max_optimization_threads=1` |
| **定期 session 重建** | 每 5000 chunks 重建 embedding session，防止 GPU 状态累积 |
| **段健康监控** | 段数量超过阈值时告警，提示存储膨胀风险 |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

开发方向建议：
- 支持更多文档格式（HTML、EPUB 等）
- 添加文档预览 Web UI
- 支持多集合/多知识库切换
- 优化大文档的 chunk 策略

---

## 📄 License

[MIT](LICENSE)

---

<div align="center">
Made with ❤️ for the AI-assisted development community
</div>
