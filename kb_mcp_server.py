#!/usr/bin/env python3
"""
Kimi Code MCP Server - 本地技术知识库检索（按需加载 + 空闲释放）
依赖: pip install mcp qdrant-client fastembed
"""
import os
import sys
import asyncio
import gc
import threading
from typing import Any, Sequence, Optional

# 禁用 HuggingFace 网络检查，强制使用本地模型
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 禁用 ONNXRuntime 冗长日志，避免 VS Code 输出乱码
os.environ['ORT_LOGGING_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Windows 下强制 UTF-8 输出，避免 stderr 乱码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from qdrant_client import QdrantClient
# ==================== GPU 环境配置 ====================
_nvdiapath = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'nvidia')
if os.path.exists(_nvdiapath):
    for _pkg in os.listdir(_nvdiapath):
        _bindir = os.path.join(_nvdiapath, _pkg, 'bin')
        if os.path.exists(_bindir) and _bindir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _bindir + os.pathsep + os.environ.get('PATH', '')

from fastembed import TextEmbedding

# 进一步抑制 ONNXRuntime C++ 日志
import onnxruntime as _ort
_ort.set_default_logger_severity(3)  # 3 = ERROR
_ort.set_default_logger_verbosity(0)


# ==================== 配置 ====================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("KB_COLLECTION", "emulate3d_docs")
EMBED_MODEL = "intfloat/multilingual-e5-large"  # 可换 "BAAI/bge-small-zh-v1.5" 节省内存
IDLE_TIMEOUT = int(os.getenv("KB_IDLE_TIMEOUT", "600"))  # 秒，0=不释放
# 脚本所在目录作为项目根目录，确保 cache_dir 绝对路径正确
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")

# ==================== 按需加载 + 空闲释放 ====================
_embedder: Optional[TextEmbedding] = None
_qdrant: Optional[QdrantClient] = None
_idle_timer: Optional[threading.Timer] = None

def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant

def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        print(f"[KB-MCP] 加载模型 {EMBED_MODEL}...", file=sys.stderr)
        _embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE_DIR)
        print("[KB-MCP] 模型就绪", file=sys.stderr)
    return _embedder

def release_embedder():
    global _embedder, _idle_timer
    if _embedder is not None:
        del _embedder
        _embedder = None
        gc.collect()
        print("[KB-MCP] 模型已释放（空闲超时）", file=sys.stderr)
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None

def reset_idle_timer():
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
    if IDLE_TIMEOUT > 0:
        _idle_timer = threading.Timer(IDLE_TIMEOUT, release_embedder)
        _idle_timer.daemon = True
        _idle_timer.start()

# ==================== MCP Server ====================
app = Server("local-kb-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_tech_kb",
            description=(
                "从本地Qdrant知识库检索自动化仓储、Emulate3D、WMS/WCS、PLC相关技术文档片段。"
                "当用户询问API用法、类方法、仿真逻辑、硬件参数、系统对接方案时，必须调用此工具获取权威文档片段，禁止凭记忆回答技术细节。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索意图，提取2-5个核心关键词。例如：'RackConfigurator GenerateSlots JSON格式'"
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["api_reference", "user_manual", "tutorial", "design_spec", "all"],
                        "description": "文档类型过滤，不确定时传all"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "返回片段数量，建议5-8"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_class_api",
            description="精确检索指定类名或方法名的API文档。当用户明确提到类名（如RackConfigurator）或方法名时调用。",
            inputSchema={
                "type": "object",
                "properties": {
                    "class_name": {"type": "string", "description": "类名，如 RackConfigurator"},
                    "method_name": {"type": "string", "description": "方法名，可选，如 ExportJSON"}
                },
                "required": ["class_name"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    try:
        if name == "search_tech_kb":
            return await _handle_search(arguments)
        elif name == "get_class_api":
            return await _handle_class_api(arguments)
        return [TextContent(type="text", text=f"未知工具: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"[知识库查询错误] {str(e)}")]
    finally:
        reset_idle_timer()

async def _handle_search(args: dict) -> Sequence[TextContent]:
    query = args["query"]
    doc_type = args.get("doc_type", "all")
    top_k = args.get("top_k", 5)

    embedder = get_embedder()
    vector = list(embedder.embed([query]))[0].tolist()

    qdrant = get_qdrant()
    filter_must = []
    if doc_type != "all":
        filter_must.append({"key": "doc_type", "match": {"value": doc_type}})

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        query_filter={"must": filter_must} if filter_must else None,
        with_payload=True,
        score_threshold=0.65
    )

    if not results:
        return [TextContent(type="text", text="未在知识库中找到相关文档片段。建议更换关键词或确认文档已入库。")]

    chunks = []
    for i, hit in enumerate(results, 1):
        p = hit.payload
        extra = []
        if p.get('page_number'):
            extra.append(f"页码:{p['page_number']}")
        if p.get('has_image'):
            extra.append("含图")
        extra_str = (' | ' + ' | '.join(extra)) if extra else ''
        chunks.append(
            f"[片段{i}] 相关度:{hit.score:.3f} | 来源:{p.get('source','未知')}{extra_str}\n"
            f"标题:{p.get('title','')} | 章节:{p.get('heading','')}\n"
            f"{p.get('text','')[:1000]}"
        )
    return [TextContent(type="text", text="\n---\n".join(chunks))]

async def _handle_class_api(args: dict) -> Sequence[TextContent]:
    class_name = args["class_name"]
    method_name = args.get("method_name")
    query = f"{class_name} {method_name or ''}".strip()

    embedder = get_embedder()
    vector = list(embedder.embed([query]))[0].tolist()

    qdrant = get_qdrant()
    filter_must = [{"key": "class_name", "match": {"value": class_name}}]
    if method_name:
        filter_must.append({"key": "method_name", "match": {"value": method_name}})

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=3,
        query_filter={"must": filter_must},
        with_payload=True
    )

    if not results:
        return [TextContent(type="text", text=f"未找到类 {class_name} 的文档。")]

    texts = []
    for r in results:
        p = r.payload
        texts.append(
            f"类:{p.get('class_name')} | 方法:{p.get('method_name','N/A')}\n"
            f"命名空间:{p.get('namespace','N/A')}\n"
            f"{p.get('text','')}"
        )
    return [TextContent(type="text", text="\n---\n".join(texts))]

async def main():
    print("[KB-MCP] 服务启动（模型按需加载，空闲自动释放）", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
