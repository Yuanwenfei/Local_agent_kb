#!/usr/bin/env python3
"""
Kimi Code MCP Server - 本地技术知识库检索（按需加载 + 空闲释放）
依赖: pip install mcp qdrant-client fastembed
"""
import os
import sys
import asyncio
import gc
import math
import threading
import logging
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
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue
)
import re
import json
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

# ==================== 日志配置 ====================
logger = logging.getLogger("kb_mcp_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler(sys.stderr)
    _ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_ch)


# ==================== 配置 ====================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("KB_COLLECTION", "emulate3d_docs")
EMBED_MODEL = "intfloat/multilingual-e5-large"  # 可换 "BAAI/bge-small-zh-v1.5" 节省内存
IDLE_TIMEOUT = int(os.getenv("KB_IDLE_TIMEOUT", "600"))  # 秒，0=不释放
# GPU 支持：设置 KB_USE_GPU=1 启用 CUDA 推理
USE_GPU = os.getenv("KB_USE_GPU", "0") == "1"
ONNX_PROVIDERS = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                  if USE_GPU else None)
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
        logger.info(f"加载模型 {EMBED_MODEL}...")
        _embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE_DIR,
                                  providers=ONNX_PROVIDERS)
        logger.info("模型就绪")
    return _embedder

def release_embedder():
    global _embedder, _idle_timer
    if _embedder is not None:
        del _embedder
        _embedder = None
        gc.collect()
        logger.info("模型已释放（空闲超时）")
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

def _validate_query_vector(vec: list) -> None:
    """校验查询向量：检测零向量/NaN/Inf，防止无效查询导致无法定位问题"""
    if all(abs(x) < 1e-9 for x in vec):
        logger.error("查询向量为零向量，embedding 模型可能异常")
        raise ValueError("查询向量全零，embedding 模型异常")
    if any(math.isnan(x) or math.isinf(x) for x in vec):
        logger.error("查询向量含 NaN/Inf，embedding 模型异常")
        raise ValueError("查询向量含 NaN/Inf，embedding 模型异常")


def _embed_query(query: str) -> list:
    """嵌入查询文本并验证向量有效性"""
    embedder = get_embedder()
    vector = list(embedder.embed([query]))[0].tolist()
    _validate_query_vector(vector)
    return vector


# ==================== 查询辅助函数 ====================

def _parse_yaml_frontmatter(text: str) -> dict:
    """从 Markdown 文本中提取 YAML frontmatter（简易解析，无第三方依赖）"""
    m = re.match(r'^---\s*\n(.*?)\n---', text, re.DOTALL)
    if not m:
        return {}
    result = {}
    for line in m.group(1).split('\n'):
        kv = re.match(r'^(\w[\w_-]*)\s*:\s*(.*)', line)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            # 去除可能的多行值引号
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            result[key] = val
    return result


def _scroll_class_docs(qdrant: QdrantClient, class_name: str,
                        method_name: str = None) -> list[dict]:
    """用 scroll 获取某个类的所有文档 chunk（Python 端按 source 路径过滤）。
    
    由于 class_name payload 字段在索引时用 `class\\s+(\\w+)` 正则提取，
    无法匹配 YAML frontmatter 的 `class: Foo`（带冒号），
    大部分方法/属性文档的 class_name 为空，无法用 Qdrant filter 精准匹配。
    
    因此回退方案：全量 scroll（仅 payload 无向量，~2万条约 300~500ms），
    Python 端按 source 路径过滤。结果缓存避免重复扫描。
    """
    # 缓存：全量文档列表（仅 payloads）
    if not hasattr(_scroll_class_docs, "_cache"):
        _scroll_class_docs._cache = None

    if _scroll_class_docs._cache is None:
        all_points = []
        offset = None
        while True:
            batch, offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=3000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            if offset is None:
                break
        _scroll_class_docs._cache = all_points
        logger.info(f"全量文档缓存: {len(all_points)} 条")

    # Python 端过滤：source 路径包含 `_ClassName[._]`（如 _TransferState_ 或 _TransferState.）
    pattern = f"_{class_name}[._]"
    matched = []
    for pt in _scroll_class_docs._cache:
        src = pt.payload.get("source", "")
        if re.search(pattern, src.replace('\\', '/')):
            if method_name:
                mn = pt.payload.get("method_name", "")
                if mn and method_name.lower() in mn.lower():
                    matched.append(pt)
            else:
                matched.append(pt)

    logger.info(f"类文档: class={class_name}, method={method_name}, 匹配={len(matched)}")
    return matched

    all_points = []
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=must),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if offset is None:
            break
    return all_points


def _extract_signature(text: str) -> str:
    """从文本中提取第一个 ```csharp 代码块的签名"""
    m = re.search(r'```csharp\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _build_api_summary(points: list, class_name: str) -> str:
    """把 scroll 结果按 api_kind 分组，输出结构化 API 表面"""
    # 按 api_kind 分组
    constructors: list[dict] = []
    methods: list[dict] = []
    properties: list[dict] = []
    operators: list[dict] = []
    overview: dict = {}
    namespace = ""
    assembly = ""
    summary = ""

    for pt in points:
        p = pt.payload
        text = p.get("text", "")
        front = _parse_yaml_frontmatter(text)
        api_kind = front.get("api_kind", "")
        member = front.get("member", "")
        sig = front.get("signature", "") or _extract_signature(text)

        if not namespace:
            namespace = front.get("namespace", "")
        if not assembly:
            assembly = front.get("assembly", "")
        if not summary:
            summary = front.get("summary", "")

        entry = {
            "member": member,
            "signature": sig,
            "title": p.get("title", ""),
            "heading": p.get("heading", ""),
            "text": text,
        }

        # 类概览文档（api_kind=class）→ overview
        if api_kind == "class":
            if not overview:
                overview = entry
        # 构造函数
        elif api_kind == "constructor" and member:
            constructors.append(entry)
        # 方法
        elif api_kind == "method" and member:
            methods.append(entry)
        # 属性
        elif api_kind == "property" and member:
            properties.append(entry)
        # 运算符重载
        elif api_kind in ("method", "constructor") and "Operator" in p.get("title", ""):
            operators.append(entry)

    lines = [f"## {class_name} 类"]
    if namespace:
        lines.append(f"**命名空间:** `{namespace}`")
    if assembly:
        lines.append(f"**程序集:** {assembly}")
    if summary:
        lines.append(f"\n{summary}\n")

    # 构造函数
    if constructors:
        lines.append("\n### 🔧 构造函数")
        for c in _dedupe_by_member(constructors):
            lines.append(f"- `{c['signature']}`" if c['signature'] else f"- {c['member']}()")
        lines.append("")

    # 属性
    if properties:
        lines.append("### 📋 属性")
        for p in _dedupe_by_member(properties):
            lines.append(f"- `{p['signature']}`" if p['signature'] else f"- {p['member']}")
        lines.append("")

    # 方法
    if methods:
        lines.append("### ⚙️ 方法")
        for m in _dedupe_by_member(methods):
            lines.append(f"- `{m['signature']}`" if m['signature'] else f"- {m['member']}()")
        lines.append("")

    # 运算符
    if operators:
        lines.append("### 🔣 运算符重载")
        for o in operators:
            lines.append(f"- `{o['signature']}`" if o['signature'] else f"- {o['member']}")
        lines.append("")

    if overview and overview.get("text"):
        lines.append(f"---\n{overview['text'][:800]}")
    elif not constructors and not methods and not properties and not operators:
        # fallback：直接返回原始文本
        for pt in points[:5]:
            lines.append(pt.payload.get("text", "")[:1000])

    return "\n".join(lines).strip()


def _dedupe_by_member(entries: list[dict]) -> list[dict]:
    """按 member 字段去重，保留有签名的"""
    seen = {}
    for e in entries:
        key = e["member"]
        if key not in seen or (e["signature"] and not seen[key]["signature"]):
            seen[key] = e
    return list(seen.values())


# ==================== MCP 工具注册 ====================

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
                    },
                    "score_threshold": {
                        "type": "number",
                        "default": 0.65,
                        "description": "最低相关度阈值(0~1)，阈值越低结果越多但噪声越大。默认0.65，无结果时可降至0.5"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_class_api",
            description=(
                "精确检索指定类的完整API文档，自动聚合构造函数、方法、属性、运算符重载。"
                "当用户明确提到类名（如BBox、RackConfigurator）或方法名时调用。"
                "可选 method_name 参数可聚焦某个具体方法及其重载。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "class_name": {"type": "string", "description": "类名，如 RackConfigurator"},
                    "method_name": {"type": "string", "description": "方法名，可选，如 ExportJSON。指定后仅返回该方法的重载签名"}
                },
                "required": ["class_name"]
            }
        ),
        Tool(
            name="list_classes",
            description=(
                "浏览知识库中已索引的类，可按命名空间过滤。"
                "当用户想了解知识库覆盖了哪些类、探索某个命名空间下的API时调用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "命名空间前缀过滤，如 Demo3D.Visuals。不指定则返回所有命名空间概览"
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["api_reference", "user_manual", "tutorial", "design_spec", "all"],
                        "default": "api_reference",
                        "description": "文档类型过滤，默认 api_reference"
                    },
                    "top_n": {
                        "type": "integer",
                        "default": 50,
                        "description": "最多返回的类数量"
                    }
                }
            }
        ),
        Tool(
            name="kb_stats",
            description=(
                "查看本地知识库的统计概览：文档总数、各类型分布、向量维度等。"
                "当用户询问知识库规模、覆盖范围、索引状态时调用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
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
        elif name == "list_classes":
            return await _handle_list_classes(arguments)
        elif name == "kb_stats":
            return await _handle_kb_stats(arguments)
        return [TextContent(type="text", text=f"未知工具: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"[知识库查询错误] {str(e)}")]
    finally:
        reset_idle_timer()

# ==================== 工具实现 ====================

async def _handle_search(args: dict) -> Sequence[TextContent]:
    query = args["query"]
    doc_type = args.get("doc_type", "all")
    top_k = args.get("top_k", 5)
    score_threshold = args.get("score_threshold", 0.65)

    vector = _embed_query(query)

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
        score_threshold=score_threshold
    )

    if not results:
        logger.warning(f"检索无结果: query='{query}', score_threshold={score_threshold}")
        hint = "建议降低 score_threshold 或更换关键词" if score_threshold > 0.5 else "建议更换关键词或确认文档已入库"
        return [TextContent(type="text", text=f"未在知识库中找到相关文档片段。{hint}。")]

    chunks = []
    for i, hit in enumerate(results, 1):
        p = hit.payload
        extra = []
        if p.get('page_number'):
            extra.append(f"页码:{p['page_number']}")
        if p.get('has_image'):
            extra.append("含图")
        extra_str = (' | ' + ' | '.join(extra)) if extra else ''
        # 清洗 YAML frontmatter，只展示正文
        text_body = p.get('text', '')
        text_body = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text_body, flags=re.DOTALL).strip()
        chunks.append(
            f"[片段{i}] 相关度:{hit.score:.3f} | 来源:{p.get('source','未知')}{extra_str}\n"
            f"标题:{p.get('title','')} | 章节:{p.get('heading','')}\n"
            f"{text_body[:1000]}"
        )
    return [TextContent(type="text", text="\n---\n".join(chunks))]


async def _handle_class_api(args: dict) -> Sequence[TextContent]:
    class_name = args["class_name"]
    method_name = args.get("method_name")

    qdrant = get_qdrant()
    points = _scroll_class_docs(qdrant, class_name, method_name)

    if not points:
        logger.warning(f"类API检索无结果: class={class_name}, method={method_name}")
        return [TextContent(type="text", text=f"未找到类 {class_name} 的文档。")]

    logger.info(f"类API: {class_name} 找到 {len(points)} 个文档片段")

    if method_name:
        # 聚焦某个方法：展示所有重载签名 + 详细文本
        lines = [f"## {class_name}.{method_name} — 重载概览\n"]
        seen_sigs = set()
        for pt in points:
            text = pt.payload.get("text", "")
            front = _parse_yaml_frontmatter(text)
            sig = front.get("signature", "") or _extract_signature(text)
            if sig and sig not in seen_sigs:
                seen_sigs.add(sig)
                lines.append(f"- `{sig}`")
        if not seen_sigs:
            lines.append("(未找到重载签名)")
        # 附加第一个文档的正文
        if points:
            text_body = re.sub(r'^---\s*\n.*?\n---\s*\n', '', points[0].payload.get("text", ""),
                               flags=re.DOTALL).strip()
            lines.append(f"\n---\n{text_body[:1200]}")
        return [TextContent(type="text", text="\n".join(lines))]
    else:
        # 聚合展示完整 API 表面
        summary = _build_api_summary(points, class_name)
        return [TextContent(type="text", text=summary)]


async def _handle_list_classes(args: dict) -> Sequence[TextContent]:
    namespace = args.get("namespace", "")
    doc_type = args.get("doc_type", "api_reference")
    top_n = args.get("top_n", 50)

    qdrant = get_qdrant()

    must = []
    if doc_type != "all":
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))

    # 从 source 路径文件名提取类名，按命名空间分组
    # 由于 class_name payload 字段不可靠，改为解析文件名中的类名
    ns_classes: dict[str, set] = {}
    offset = None
    scanned = 0
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=must) if must else None,
            limit=1000,
            offset=offset,
            with_payload=["source"],
            with_vectors=False,
        )
        for pt in points:
            src = pt.payload.get("source", "")
            cn = _extract_class_name_from_source(src)
            if not cn:
                continue
            ns = _infer_namespace(src, cn)
            if namespace and not ns.startswith(namespace):
                continue
            if ns not in ns_classes:
                ns_classes[ns] = set()
            ns_classes[ns].add(cn)
        scanned += len(points)
        if offset is None:
            break

    if not ns_classes:
        msg = f"未找到类（doc_type={doc_type}"
        if namespace:
            msg += f", namespace={namespace}"
        return [TextContent(type="text", text=msg + "）")]

    lines = [f"## 知识库中的类（doc_type={doc_type}）\n"]
    total = 0
    for ns in sorted(ns_classes.keys()):
        classes = ns_classes[ns]
        if total >= top_n:
            break
        lines.append(f"### {ns}（{len(classes)} 个）")
        for c in sorted(classes)[:30]:
            lines.append(f"  - `{c}`")
            total += 1
            if total >= top_n:
                break
        lines.append("")

    lines.append(f"\n*共扫描 {scanned} 条记录，聚合 {sum(len(v) for v in ns_classes.values())} 个类*")
    return [TextContent(type="text", text="\n".join(lines))]


def _extract_class_name_from_source(source: str) -> str:
    """从 source 路径的文件名提取类名"""
    filename = source.replace('\\', '/').rsplit('/', 1)[-1].replace('.md', '')
    parts = filename.split('_')
    if not parts:
        return ""

    prefix = parts[0]

    # Properties_T_ / Methods_T_ / Events_T_ 汇总页 -> 最后一个 part
    if prefix in ('Properties', 'Methods', 'Events') and len(parts) > 2:
        return parts[-1]

    # Overload_Demo3D_Visuals_BBox__ctor -> BBox (__ctor 前一个)
    if prefix == 'Overload':
        for p in parts:
            if p.startswith('__'):
                idx = parts.index(p)
                return parts[idx - 1] if idx > 0 else ""
        return parts[-1] if len(parts) > 2 else ""

    # T_ / M_ / P_ / E_ 文档
    if prefix in ('T', 'M', 'P', 'E') and len(parts) >= 4:
        # M_Demo3D_Visuals_BBox_Contains -> BBox = parts[-2]
        # T_Demo3D_Visuals_BBox -> BBox = parts[-1]
        if prefix == 'T':
            return parts[-1]
        return parts[-2]

    # N_ 命名空间文档 -> 不是类
    if prefix == 'N':
        return ""

    return parts[-1] if len(parts) > 1 else ""


def _infer_namespace(source: str, class_name: str) -> str:
    """从文件路径推断命名空间。利用已知的 class_name 定位文件名中的命名空间部分。"""
    # 典型路径: D:\...\md-source\API 7\M_Demo3D_Visuals_BBox_Contains.md
    # 或: /path/to/html/M_Demo3D_Visuals_BBox_Contains.md
    src = source.replace('\\', '/')
    filename = src.rsplit('/', 1)[-1].replace('.md', '')
    parts = filename.split('_')
    prefix = parts[0] if parts else ''

    # N_ 命名空间文档: N_Demo3D_Visuals -> Demo3D.Visuals
    if prefix == 'N' and len(parts) > 1:
        return '.'.join(parts[1:])

    # M_ / P_ / T_ / E_ 文档：利用 class_name 在 parts 中的位置来截取命名空间
    if prefix in ('M', 'P', 'T', 'E') and class_name in parts:
        idx = parts.index(class_name)
        if idx > 1:
            return '.'.join(parts[1:idx])
        return '.'.join(parts[1:idx]) if idx > 1 else '.'.join(parts[1:-1])

    # Overload_ 文档回退
    if prefix == 'Overload' and len(parts) > 2:
        return '.'.join(parts[1:-1])

    # 最终回退
    if len(parts) > 3:
        return '.'.join(parts[1:-2])
    elif len(parts) > 2:
        return '.'.join(parts[1:-1])
    return "unknown"


async def _handle_kb_stats(args: dict) -> Sequence[TextContent]:
    qdrant = get_qdrant()
    ci = qdrant.get_collection(COLLECTION_NAME)

    # 统计各 doc_type 数量
    type_counts: dict[str, int] = {}
    product_counts: dict[str, int] = {}
    offset = None
    total_scanned = 0
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=["doc_type", "product"],
            with_vectors=False,
        )
        for pt in points:
            dt = pt.payload.get("doc_type", "unknown")
            pr = pt.payload.get("product", "unknown")
            type_counts[dt] = type_counts.get(dt, 0) + 1
            product_counts[pr] = product_counts.get(pr, 0) + 1
        total_scanned += len(points)
        if offset is None:
            break

    # 读取 index_state.json 获取最后索引时间
    index_time = "未知"
    try:
        index_state_file = os.path.join(PROJECT_ROOT, "index_state.json")
        if os.path.exists(index_state_file):
            mtime = os.path.getmtime(index_state_file)
            from datetime import datetime
            index_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    lines = [
        f"## 知识库概览",
        f"",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| 集合名称 | `{COLLECTION_NAME}` |",
        f"| 向量维度 | {ci.config.params.vectors.size} |",
        f"| 距离算法 | {ci.config.params.vectors.distance} |",
        f"| 数据点数 | {ci.points_count} |",
        f"| 段数量 | {ci.segments_count} |",
        f"| 最后索引 | {index_time} |",
        f"",
        f"### 文档类型分布",
    ]
    for dt in sorted(type_counts.keys()):
        lines.append(f"- **{dt}**: {type_counts[dt]}")

    lines.append(f"\n### 产品分布")
    for pr in sorted(product_counts.keys()):
        lines.append(f"- **{pr}**: {product_counts[pr]}")

    return [TextContent(type="text", text="\n".join(lines))]

async def main():
    logger.info("服务启动（模型按需加载，空闲自动释放）")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
