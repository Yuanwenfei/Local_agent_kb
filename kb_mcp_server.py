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
import time
import logging
from typing import Any, Sequence, Optional
from datetime import datetime

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
from mcp.types import TextContent, ImageContent, Tool

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import io
import re

# 元数据规范 v2 共享层（图片路径解析、lint，见 kb_schema.py）
# ---------------------------------------------------------------------------
# 本地模块导入引导：本项目跑在**嵌入式 Python** 上（python312 目录含 python312._pth），
# 此时 sys.path 只取 ._pth 里列的三条路径，**不会自动加入脚本所在目录**，
# 于是 MCP 客户端按绝对路径启动时 `import kb_schema` 必然 ModuleNotFoundError
# （报错形如 "No module named 'kb_schema'"，连接随即关闭 -32000）。
# 这里显式补上脚本目录，使入口点从任意 cwd 启动都自洽。
# ---------------------------------------------------------------------------
_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _LOCAL_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_DIR)

from kb_schema import (
    resolve_image_paths, parse_meta_yaml, load_sidecar_meta, lint_meta, get_nested,
    normalize_for_search,
)
# P3 知识类型模板层（front-matter v2 渲染 / 入库校验）
import kb_templates as kbt

# ==================== GPU 环境配置 ====================
_nvdiapath = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'nvidia')
if os.path.exists(_nvdiapath):
    for _pkg in os.listdir(_nvdiapath):
        _bindir = os.path.join(_nvdiapath, _pkg, 'bin')
        if os.path.exists(_bindir) and _bindir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _bindir + os.pathsep + os.environ.get('PATH', '')

from fastembed import TextEmbedding, SparseTextEmbedding

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
# GPU 支持：设置 KB_USE_GPU=1 启用 CUDA 推理；否则显式锁定 CPU
# 注意：不能写 None，None 会让 ONNX Runtime 默认优先挑选 CUDA，
# 导致 KB_USE_GPU=0 实际上仍在用 GPU（本服务为常驻查询进程，默认应走 CPU）
USE_GPU = os.getenv("KB_USE_GPU", "0") == "1"
ONNX_PROVIDERS = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                  if USE_GPU else ["CPUExecutionProvider"])
# 脚本所在目录作为项目根目录，确保 cache_dir 绝对路径正确
PROJECT_ROOT = _LOCAL_DIR
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")

# ==================== P2 检索层配置 ====================
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
SPARSE_VECTOR_NAME = "bm42"          # 与 index_docs.py 保持一致
DENSE_VECTOR_NAME = ""               # 匿名 dense 向量
DEFAULT_TOPK = 5
DEFAULT_THRESHOLD = 0.65             # 语义模式的余弦阈值；hybrid 下只作用于 dense 预筛
HYBRID_DENSE_THRESHOLD = 0.55        # hybrid 的 dense 预筛默认阈值（低于语义模式，避免召回枯竭）
FUSION_K = 1                         # RRF 平滑常数（与 Qdrant 内置 RRF 行为一致）
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
EXACT_WEIGHT = 1.0                   # 精确命中通道权重（"精确命中置顶"）
# 术语锚点通道：查询里的标识符（CamelCase / 带点下划线 / 缩写）且全库稀有 → 高精度强信号
ANCHOR_WEIGHT = 1.0
ANCHOR_MAX_DF = 60                   # 锚点词命中片段数上限（更常见就不具备区分度，不做锚点）
CANDIDATE_POOL = 3                   # 候选池 = top_k * CANDIDATE_POOL（后处理前的池子）
TOC_PENALTY = 0.6                    # 目录页降权（设计 §6.3）
MAX_PER_SOURCE = 2                   # 单文档最多命中数
TRUST_WEIGHT = {"measured": 1.0, "tutorial": 0.9, "pending": 0.75}
RECENCY_FLOOR = 0.85                 # 6 个月线性衰减到 0.85，之后不再衰减
RECENCY_DAYS = 182.5
# 时效衰减只对「结论类」kind 生效（2026-10-01 口径修正，见 _recency_factor）：
# 教程/手册/API 参考的"过时"应由 applies_to.framework + version= 过滤 + ⚠version-drift 表达，
# 用日期隐性降权会误伤仍有效的 Legacy 教程；且本库 doc_date 全部来自抓取日（同质、无区分力）。
RECENCY_KINDS = {"note", "defect_log", "error_faq", "model_fact_card", "version_matrix"}
LOW_CONF_HYBRID = 0.5                # hybrid：只有单通道支持（<0.5）即视为低置信
LOW_CONF_SEMANTIC = 0.60             # 语义：余弦低于此值视为低置信

# ==================== P3 知识类型与入库通道配置 ====================
DOCS_ROOT = os.getenv("KB_DOCS_DIR") or os.path.join(PROJECT_ROOT, "md-source")
# D-6：Kimi 的副本目录纳入 kb_dupes 巡检范围，但**不入库**
DUPES_EXTRA_DIRS = [d.strip() for d in
                    (os.getenv("KB_DUPES_EXTRA_DIRS") or r"E:\kimi code workbentch").split(os.pathsep)
                    if d.strip()]
CANONICAL_HINT = os.getenv("KB_CANONICAL_HINT", "local_agent_kb").lower()  # 正本判据：路径含此片段
DOC_READ_BUDGET = int(os.getenv("KB_DOC_BUDGET", "3000"))   # kb_get_doc 单次返回字符预算
THUMB_MAX_SIDE = int(os.getenv("KB_THUMB_MAX_SIDE", "512"))  # P3-7 缩略图最长边（设计 §11）
THUMB_QUALITY = int(os.getenv("KB_THUMB_QUALITY", "80"))
_PLUGIN_EXTS = {".md", ".yaml", ".yml", ".json", ".qlp", ".txt"}

# ==================== 按需加载 + 空闲释放 ====================
_embedder: Optional[TextEmbedding] = None
_sparse_embedder: Optional[SparseTextEmbedding] = None
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


def get_sparse_embedder() -> Optional[SparseTextEmbedding]:
    """稀疏通道（BM42，约 90MB）。模型缺失时返回 None——hybrid 自动降级为 dense+精确。"""
    global _sparse_embedder
    if _sparse_embedder is None:
        try:
            logger.info(f"加载稀疏模型 {SPARSE_MODEL}...")
            _sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL,
                                                   cache_dir=MODEL_CACHE_DIR,
                                                   providers=ONNX_PROVIDERS)
            logger.info("稀疏模型就绪")
        except Exception as e:
            logger.warning(f"稀疏模型不可用，hybrid 降级为 dense+精确: {e}")
            return None
    return _sparse_embedder


def release_embedder():
    global _embedder, _sparse_embedder, _idle_timer
    if _embedder is not None:
        del _embedder
        _embedder = None
        gc.collect()
        logger.info("模型已释放（空闲超时）")
    if _sparse_embedder is not None:
        del _sparse_embedder
        _sparse_embedder = None
        gc.collect()
        logger.info("稀疏模型已释放（空闲超时）")
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


# ==================== MCP 工具注册 ====================

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_tech_kb",
            description=(
                "从本地Qdrant知识库检索自动化仓储、Emulate3D、WMS/WCS、PLC相关技术文档片段。"
                "当用户询问API用法、类方法、仿真逻辑、硬件参数、系统对接方案时，必须调用此工具获取权威文档片段，禁止凭记忆回答技术细节。\n"
                "检索模式 mode：hybrid（默认，dense+sparse+精确三通道 RRF 融合，最稳）／semantic（纯语义，"
                "适合概念性描述）／exact（精确串，适合报错原文、类名、控件语法）。\n"
                "分数口径：hybrid 返回 RRF 融合分（单通道命中最高约 0.5，双通道 1.0，**不是余弦相似度**）；"
                "semantic 返回余弦相似度（0~1）。score_threshold 在 semantic 下作用于最终结果，"
                "在 hybrid 下只作用于 dense 预筛（sparse 与精确通道不受其限制），因此 hybrid 下不要用 0.65 这类高阈值。\n"
                "返回体标注每个片段的通道、可信度、适用版本与是否正本；出现「⚠version-drift」表示该片段版本与请求版本不一致。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索意图，提取2-5个核心关键词。例如：'RackConfigurator GenerateSlots JSON格式'"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "exact"],
                        "default": "hybrid",
                        "description": "hybrid=dense+sparse+精确融合（默认）；semantic=纯语义；exact=精确串（报错原文/类名/控件语法）"
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["api_reference", "tutorial", "user_manual",
                                 "design_spec", "release_note", "all"],
                        "description": "文档类型过滤，不确定时传all"
                    },
                    "section": {
                        "type": "string",
                        "description": (
                            "章节过滤（可选）。可传完整路径如 'Catalogs and Plugins > AMRs'，"
                            "也可只传末级名如 'AMRs'（自动解析成完整路径）；不传则全库检索。"
                            "已知只在特定章节里提问时使用，可显著减少跨章节噪声。"
                        )
                    },
                    "version": {
                        "type": "string",
                        "description": "适用版本精确过滤，如 '2.4.0.0'。命中为空时自动放宽并给旧版附 ⚠version-drift 告警"
                    },
                    "model": {
                        "type": "string",
                        "description": "适用机型/控制器过滤，如 'ACR'；含 general（通用）条目"
                    },
                    "trust": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["measured", "tutorial", "pending"]},
                        "description": "可信度过滤：measured=插件产物实测 / tutorial=人工文档 / pending=待验"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["doc", "fact", "error_faq", "note", "signature", "screen"],
                        "description": "知识类型过滤"
                    },
                    "path_prefix": {
                        "type": "string",
                        "description": "源文件路径前缀过滤（可选），如 'md-source\\\\E3D Tutorials'"
                    },
                    "canonical_only": {
                        "type": "boolean",
                        "default": True,
                        "description": "只返回正本（排除已标记 canonical:false 的副本），默认 true"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "返回片段数量，建议5-8"
                    },
                    "score_threshold": {
                        "type": "number",
                        "description": (
                            "最低相关度阈值。semantic 默认 0.65（余弦）；"
                            "hybrid 默认 0.55 且只作用于 dense 预筛（sparse/精确通道不受限）"
                        )
                    },
                    "attach_image": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "top1 命中含本地图片时，随结果附带首图缩略图（MCP image content）。"
                            "默认开启：教程类问题的关键信息常只在图里，收到图后请直接读图作答。"
                            "纯文本客户端会自动降级为路径列表；不需要图片时置 false 省 token。"
                        )
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="kb_stats",
            description=(
                "查看本地知识库的统计概览：总量、类型/可信度分布、向量配置、治理指标（含图/目录页/副本）。"
                "当用户询问知识库规模、覆盖范围、索引状态时调用。默认用 count 聚合、速度快；"
                "需要产品分布与章节分布时传 detail=true（会全量扫描，较慢）。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "detail": {
                        "type": "boolean",
                        "default": False,
                        "description": "是否输出产品/章节分布等明细（需全量扫描，10K 点约数秒）"
                    }
                },
            }
        ),
        Tool(
            name="kb_list_docs",
            description=(
                "列出知识库文档清单及其元数据（片段数/类型/适用版本/文档日期/可信度/是否正本）。"
                "当用户询问'库里有哪文档'、'哪些文档没元数据'、按版本或类型盘点文档时调用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_type": {
                        "type": "string",
                        "enum": ["api_reference", "tutorial", "user_manual",
                                 "design_spec", "release_note", "all"],
                        "description": "文档类型过滤，默认 all"
                    },
                    "trust": {
                        "type": "string",
                        "enum": ["measured", "tutorial", "pending"],
                        "description": "按可信度过滤（可选）"
                    },
                    "section": {"type": "string", "description": "章节关键词过滤（可选，子串匹配）"},
                    "keyword": {"type": "string", "description": "文件名/标题关键词过滤（可选）"},
                    "limit": {"type": "integer", "default": 50, "description": "最多显示篇数"}
                }
            }
        ),
        Tool(
            name="kb_meta_lint",
            description=(
                "元数据体检：列出缺必填元数据（title/applies_to.framework/doc_date/trust/canonical）的文档、"
                "索引字段未回填的老切片、canonical_uid 冲突的副本组。"
                "在补元数据、做正本合并、确认是否需要重跑入库前调用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "top_docs": {
                        "type": "integer", "default": 0,
                        "description": "只体检片段数最多的 N 篇（0=全量，高频文档优先补，建议 20）"
                    },
                    "limit": {"type": "integer", "default": 30, "description": "每类清单最多列出的文档数"}
                }
            }
        ),
        Tool(
            name="kb_get_doc",
            description=(
                "按 source 取整篇文档或某个章节的**连续全文**（检索只给 1000 字符片段，这里给全）。"
                "当片段够用但仍缺上下文、需要看完整步骤/前后文、或要核对某个表的全部行时调用。"
                "source 可传绝对路径、文件名或路径片段，工具会做唯一匹配。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "文档路径/文件名/片段，如 'AMRFramework-QLP-使用手册.md'"},
                    "heading": {"type": "string", "description": "只取该章节（含子章节）；不传则取整篇"},
                    "budget": {"type": "integer", "default": 3000, "description": "返回字符预算上限，超出截断并提示"}
                },
                "required": ["source"]
            }
        ),
        Tool(
            name="kb_outline",
            description=(
                "取某篇文档的章节树（层级 + 各级片段数），用于判断「这份文档里到底有没有讲某主题」"
                "以及该翻哪一节。读长文档前先调用它可以少读很多无关内容。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "文档路径/文件名/片段"},
                    "max_depth": {"type": "integer", "default": 3, "description": "最大展开层级"}
                },
                "required": ["source"]
            }
        ),
        Tool(
            name="kb_error_lookup",
            description=(
                "错误串专用检索：拿日志原文/异常串去精确匹配「错误串 FAQ」，返回**根因 + 守卫/修复范式**。"
                "当出现报错、异常、控件校验失败、插件加载失败时优先用本工具（比 search_tech_kb 更准）。"
                "返回里 `error_faq` 类型是已登记的权威结论；`doc` 类型只是原文出现，需自行判断。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "error_text": {"type": "string", "description": "报错原文或其中最有辨识度的一段（不要只给 'Exception' 这种通用词）"},
                    "top_k": {"type": "integer", "default": 5, "description": "返回条数"},
                    "exact_only": {"type": "boolean", "default": False, "description": "只在已登记的 error_faq 里找，不退回全库"}
                },
                "required": ["error_text"]
            }
        ),
        Tool(
            name="kb_note",
            description=(
                "把「检索失败/踩坑现场」一键登记为规范草稿（带 front-matter v2 头）。"
                "当检索没有命中、或从对话/实测得到一条文档里没有的结论时调用，避免知识随会话蒸发。"
                "草稿落在 kb-inbox\\\\drafts\\\\，人工确认后用 kb_ingest 转正。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "一句话主题（会成为标题）"},
                    "evidence": {"type": "string", "description": "证据：检索失败的 query / 报错原文 / 实测现象"},
                    "kind": {"type": "string", "enum": ["draft"], "default": "draft", "description": "目前仅支持 draft"},
                    "meta": {"type": "object", "description": "可选元数据覆盖（framework/model/revision 等）"}
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="kb_ingest",
            description=(
                "把知识正式入库：`payload`（现写正文）或 `path`（已有文件/插件产物目录）+ `kind` + `meta`。"
                "**先 dry_run=true 看校验报告**（重复 hash / 缺必填元数据 / 版本冲突 / 正文过短 / 编码乱码 / 未填小节），"
                "报告没问题再去掉 dry_run 真正写盘并向量化。传入插件导出目录时会按产物名自动映射知识类型。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["model_fact_card", "part_signature", "version_matrix",
                                 "error_faq", "defect_log", "enum_examples", "qlp_snippet", "asset_pointer"],
                        "description": "知识类型（决定模板与必填小节的元数据）"
                    },
                    "payload": {"type": "string", "description": "正文（markdown）。与 path 二选一"},
                    "path": {"type": "string", "description": "已有文件或目录。与 payload 二选一"},
                    "meta": {"type": "object", "description": "元数据：title/framework/model/revision/trust/canonical/supersedes 等"},
                    "dry_run": {"type": "boolean", "default": True, "description": "默认 true：只出校验报告不落盘。确认无误后传 false"},
                    "index": {"type": "boolean", "default": True, "description": "写盘后是否立即向量化入库"}
                },
                "required": ["kind"]
            }
        ),
        Tool(
            name="kb_dupes",
            description=(
                "副本巡检：找出同名/同内容/同文本的多份文档，给出**正本建议**（判据：路径在 Local_agent_kb 且较新/较大）。"
                "按 D-5 只标记不删——输出可直接粘贴的 `canonical: false` + `superseded_by` 元数据。"
                "整理库前、发现同一文档两处版本不一致时调用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string", "default": "hash",
                        "enum": ["hash", "name", "content"],
                        "description": "hash=磁盘文件内容哈希（含 kimi code workbentch 巡检目录）；name=同名文件；content=库内已入库文本哈希"
                    },
                    "limit": {"type": "integer", "default": 30, "description": "最多列出多少组"},
                    "include_extra_dirs": {"type": "boolean", "default": True, "description": "是否含 KB_DUPES_EXTRA_DIRS（默认 E:\\kimi code workbentch）"}
                }
            }
        ),
        Tool(
            name="kb_caption",
            description=(
                "图注治理（P4-3）：把「图片所在小节标题 / 紧邻正文 / 图片文件名」抽成图注写回 image_captions，"
                "使图内信息至少可被文字检索到。候选口径是 **image_refs 非空**（不是 has_image）。"
                "默认 dry_run=true 只出报告；确认后传 dry_run=false 写回。"
                "reindex=true 会按篇增量重嵌（图注因此进入向量，约每篇数秒，无需全库停机）。"
                "mode=summary（多模态真图注）为二期 VLM，本期未实现。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string", "default": "extract",
                        "enum": ["extract", "summary"],
                        "description": "extract=规则抽取（本期限定）；summary=多模态摘要（二期 VLM，未实现）"
                    },
                    "source": {"type": "string", "description": "只处理某篇文档（路径/文件名/片段，唯一匹配）"},
                    "top": {"type": "integer", "default": 0, "description": "只处理片段数最多的 N 篇文档（0=全部）"},
                    "dry_run": {"type": "boolean", "default": True, "description": "默认 true：只出报告不写库"},
                    "reindex": {"type": "boolean", "default": False, "description": "写回后按篇增量重嵌，让图注参与嵌入（T9 前提）"},
                    "limit": {"type": "integer", "default": 20, "description": "报告里最多展示多少条样例"}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    try:
        if name == "search_tech_kb":
            return await _handle_search(arguments)
        elif name == "kb_stats":
            return await _handle_kb_stats(arguments)
        elif name == "kb_list_docs":
            return await _handle_kb_list_docs(arguments or {})
        elif name == "kb_meta_lint":
            return await _handle_kb_meta_lint(arguments or {})
        # ---------- P3 知识类型与入库通道 ----------
        elif name == "kb_get_doc":
            return await _handle_kb_get_doc(arguments or {})
        elif name == "kb_outline":
            return await _handle_kb_outline(arguments or {})
        elif name == "kb_error_lookup":
            return await _handle_kb_error_lookup(arguments or {})
        elif name == "kb_note":
            return await _handle_kb_note(arguments or {})
        elif name == "kb_ingest":
            return await _handle_kb_ingest(arguments or {})
        elif name == "kb_dupes":
            return await _handle_kb_dupes(arguments or {})
        # ---------- P4 图注治理 ----------
        elif name == "kb_caption":
            return await _handle_kb_caption(arguments or {})
        return [TextContent(type="text", text=f"未知工具: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"[知识库查询错误] {str(e)}")]
    finally:
        reset_idle_timer()

# ==================== 章节解析（section 过滤用） ====================

_section_cache: list[str] | None = None


def _all_sections() -> list[str]:
    """库里所有 section 全路径（懒加载 + 进程内缓存）。

    仅在首次使用 section 过滤时扫一遍 payload，之后直接命中缓存。
    """
    global _section_cache
    if _section_cache is not None:
        return _section_cache
    qdrant = get_qdrant()
    seen: set[str] = set()
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=["section"],
            with_vectors=False,
        )
        for pt in points:
            sec = (pt.payload or {}).get("section")
            if sec:
                seen.add(sec)
        if offset is None:
            break
    _section_cache = sorted(seen)
    logger.info(f"章节缓存建立: {len(_section_cache)} 个章节")
    return _section_cache


def _resolve_section(user_input: str) -> list[str]:
    """把用户输入解析成一组完整 section 路径。

    支持三种写法：完整路径精确匹配 / 大小写与空格不敏感 / 只给末级名称。
    解析不到时返回空列表（调用方应忽略该过滤条件）。
    """
    val = (user_input or "").strip()
    if not val:
        return []
    sections = _all_sections()
    low = val.lower()

    exact = [s for s in sections if s.lower() == low]
    if exact:
        return exact

    if ">" in val:
        norm = " > ".join(p.strip() for p in val.split(">") if p.strip()).lower()
        hit = [s for s in sections if s.lower() == norm]
        if hit:
            return hit

    return [s for s in sections if low in s.lower()]


# ==================== P2 检索层：过滤 / 三通道 / 融合 / 后处理 ====================

_source_cache: list[str] | None = None

# 精确通道预筛时忽略的高频短词（MatchText 是 AND 语义，放进去会把召回打成 0）
_EXACT_STOP = {"the", "a", "an", "of", "to", "is", "are", "and", "or", "in", "on", "for",
               "with", "as", "by", "at", "be", "it", "this", "that", "from", "if", "when",
               "must", "not", "no", "do", "does", "was", "were", "you", "your", "can"}


def _all_sources() -> list[str]:
    """全库 source 清单（一次全扫 + 进程内缓存）。

    path_prefix 用 MatchAny(精确路径列表) 实现——Qdrant 无前缀匹配算子，
    用 keyword 索引 + 枚举命中路径既精确又不退化成全表扫描。
    """
    global _source_cache
    if _source_cache is not None:
        return _source_cache
    qdrant = get_qdrant()
    seen: set[str] = set()
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=1000, offset=offset,
            with_payload=["source"], with_vectors=False,
        )
        for pt in points:
            s = (pt.payload or {}).get("source")
            if s:
                seen.add(s)
        if offset is None:
            break
    _source_cache = sorted(seen)
    logger.info(f"source 缓存建立: {len(_source_cache)} 个文档")
    return _source_cache


def _build_filter(doc_type="all", section="", version="", model="", trust=None,
                  kind="", path_prefix="", canonical_only=True):
    """构造 Qdrant 过滤条件（设计 §6.2 过滤扩展）。返回 (filter, 说明列表)"""
    must, must_not, notes = [], [], []

    if doc_type and doc_type != "all":
        must.append(qm.FieldCondition(key="doc_type", match=qm.MatchValue(value=doc_type)))

    if section:
        secs = _resolve_section(section)
        if secs:
            must.append(qm.FieldCondition(key="section", match=qm.MatchAny(any=secs)))
            notes.append(f"章节过滤命中 {len(secs)} 个章节")
        else:
            notes.append(f"⚠ 章节 {section!r} 未匹配任何章节，已忽略该过滤")

    if version:
        must.append(qm.FieldCondition(key="framework_version", match=qm.MatchValue(value=version)))

    if model:
        # 通用条目（general）对所有机型适用，一并纳入
        must.append(qm.FieldCondition(key="model_scope", match=qm.MatchAny(any=[model, "general"])))

    if trust:
        trust_list = [trust] if isinstance(trust, str) else list(trust)
        must.append(qm.FieldCondition(key="trust", match=qm.MatchAny(any=trust_list)))

    if kind:
        must.append(qm.FieldCondition(key="kind", match=qm.MatchValue(value=kind)))

    if path_prefix:
        srcs = [s for s in _all_sources() if s.lower().startswith(path_prefix.lower())]
        if srcs:
            must.append(qm.FieldCondition(key="source", match=qm.MatchAny(any=srcs)))
            notes.append(f"路径前缀命中 {len(srcs)} 个文档")
        else:
            notes.append(f"⚠ 路径前缀 {path_prefix!r} 未匹配任何文档，已忽略")

    if canonical_only:
        # 只排除显式标记为副本（is_canonical=False）的点；老切片无该字段视为正本
        must_not.append(qm.FieldCondition(key="is_canonical", match=qm.MatchValue(value=False)))

    flt = qm.Filter(must=must or None, must_not=must_not or None)
    return (flt if (must or must_not) else None), notes


def _merge_filter(base, *conds) -> qm.Filter:
    must = list(base.must or []) if base else []
    must.extend(conds)
    must_not = list(base.must_not or []) if base else []
    return qm.Filter(must=must or None, must_not=must_not or None)


def _dense_hits(vec, flt, limit, threshold=None):
    return get_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        using=DENSE_VECTOR_NAME,
        query_filter=flt,
        limit=limit,
        with_payload=True,
        score_threshold=threshold if threshold else None,
    ).points


def _sparse_query_vec(query: str):
    """查询侧稀疏向量（BM42 query_embed：只做 token 哈希，快了不必跑注意力）"""
    se = get_sparse_embedder()
    if se is None:
        return None
    try:
        emb = list(se.query_embed(query))[0]
    except TypeError:      # 旧版本无 query_embed
        emb = list(se.embed([query]))[0]
    if len(emb.indices) == 0:
        return None
    pairs = sorted(zip((int(i) for i in emb.indices), (float(v) for v in emb.values)))
    return qm.SparseVector(indices=[p[0] for p in pairs], values=[p[1] for p in pairs])


def _sparse_hits(sv, flt, limit):
    """稀疏通道检索。集合未配置该 sparse 向量时返回 None，由调用方降级（不炸整次检索）。"""
    try:
        return get_qdrant().query_points(
            collection_name=COLLECTION_NAME,
            query=sv,
            using=SPARSE_VECTOR_NAME,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        ).points
    except Exception as e:
        logger.warning(f"稀疏通道不可用（集合可能缺少 sparse 向量配置 {SPARSE_VECTOR_NAME}）: {e}")
        return None


def _exact_terms(tokens: list[str]) -> list[str]:
    """挑预筛 token：去掉高频短词后按长度降序（长的更稀有、更有区分度）"""
    cand = [t for t in dict.fromkeys(tokens) if len(t) >= 3 and t.lower() not in _EXACT_STOP]
    cand.sort(key=len, reverse=True)
    return cand


def _exact_hits(norm_query: str, vec, flt, limit):
    """精确串通道（P2-1 / G-F）：text 索引预筛 + Python 子串精排。

    实测：Qdrant 的 MatchText 是 **AND** 语义，整条错误串（如
    `gripper component must exist as a child of`）直接进过滤会命中 0 点，
    所以只拿最稀有的 1–3 个 token 做预筛（宽进），再用规范化子串在候选内精排（严出）。
    候选集受 text 索引限制，不是全量 scroll + 内存正则。
    返回 [(hit, exactness)]，exactness=1.0 表示整串命中，否则为 token 覆盖率。
    """
    tokens = norm_query.split()
    terms = _exact_terms(tokens)
    if not terms:
        return []

    hits = []
    for probe in (terms[:3], terms[:1]):          # 先严后宽：3 token AND 失败就退到最长 1 个
        f = _merge_filter(flt, qm.FieldCondition(
            key="search_text", match=qm.MatchText(text=" ".join(probe))))
        hits = _dense_hits(vec, f, limit)
        if hits:
            break
    if not hits:
        return []

    uniq = set(tokens)
    out = []
    for h in hits:
        p = h.payload or {}
        st = normalize_for_search(p.get("search_text") or p.get("text") or "")
        if norm_query and norm_query in st:
            out.append((h, 1.0))
            continue
        # 预筛是「最长 token AND」，宽进；这里才是权威判定。
        # 只有整串命中才算精确命中——实测按覆盖率放宽（≥0.6）会把
        # `gripper component must exist as a child of` 这类**库里不存在**的错误串
        # 误报成「精确命中」（真的命中的是讲 gripper 的教程），给出假证据。
        cov = sum(1 for t in uniq if t in st) / max(len(uniq), 1)
        if cov >= 0.8:
            logger.info(f"精确通道：token 覆盖 {cov:.0%}（非整串）— {p.get('source')}")
    out.sort(key=lambda kv: (-kv[1], -kv[0].score))
    return out


# 锚点词 df 缓存：同一标识符的全库命中数在两次重跑之间是常量，而每次查询都要重查
# （实测落在 hybrid 关键路径上，每候选约 10ms，是锚点通道的主要耗时）。
# TTL 到期自动重查 → 重跑语料后最多 KB_DF_CACHE_TTL 秒自愈；重启服务立即生效。
_DF_CACHE: dict[str, tuple[float, int]] = {}
DF_CACHE_TTL = int(os.getenv("KB_DF_CACHE_TTL", "600"))
_DF_CACHE_MAX = 4096


def _text_match_count(text: str) -> int:
    """用 text 索引统计某词命中的片段数（判断稀有度用；常数级，不全扫）"""
    now = time.monotonic()
    cached = _DF_CACHE.get(text)
    if cached and now - cached[0] < DF_CACHE_TTL:
        return cached[1]
    try:
        n = get_qdrant().count(
            collection_name=COLLECTION_NAME,
            count_filter=qm.Filter(must=[qm.FieldCondition(
                key="search_text", match=qm.MatchText(text=text))]),
            exact=True,
        ).count
    except Exception:
        return -1                      # 查询失败不缓存，下次重试
    if len(_DF_CACHE) >= _DF_CACHE_MAX:
        _DF_CACHE.clear()
    _DF_CACHE[text] = (now, n)
    return n


def _anchor_terms(norm_query: str) -> list[str]:
    """挑出查询里的「标识符型」token 作为术语锚点。

    实测动机：q02「RackConfigurator GenerateSlots 用法」——sparse 的 Top1/Top2 正是
    全库仅 2 个真正含 `RackConfigurator` 的片段（相似度 1.567/1.209），但因为
    dense 权重(0.7) > sparse(0.3)，Top1 被 dense 的无关片段占住（dense Top3 一个都不含该词）。
    这类"查询里有明确标识符"的诉求（D5：插件 API 名查不到）需要一个高精度锚点信号。

    规则：CamelCase / 含 `.`/`_` / 全大写缩写，且全库命中片段数 ≤ ANCHOR_MAX_DF（稀有才有区分度）。
    """
    cands = []
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{2,}", norm_query or ""):
        if not (re.search(r'[A-Z]', t[1:]) or '_' in t or '.' in t):
            continue                      # 纯小写普通词不做锚点
        cands.append(t.rstrip('.'))
    out = []
    for t in dict.fromkeys(cands):
        n = _text_match_count(t)
        if 0 < n <= ANCHOR_MAX_DF:
            out.append(t)
        else:
            logger.info(f"锚点候选 {t!r} 命中 {n} 片段，区分度不足，忽略")
    return out


def _anchor_hits(anchors: list[str], vec, flt, limit) -> list:
    """术语锚点通道：字面含锚点的片段（text 索引预筛，非全扫）→ 按 dense 排序"""
    merged: dict = {}
    for t in anchors[:3]:
        f = _merge_filter(flt, qm.FieldCondition(
            key="search_text", match=qm.MatchText(text=t)))
        for h in _dense_hits(vec, f, limit):
            if h.id not in merged or h.score > merged[h.id].score:
                merged[h.id] = h
    return sorted(merged.values(), key=lambda h: -h.score)


def _anchor_channel(norm_query: str, vec, flt, limit) -> tuple[list, list[str]]:
    """锚点通道整体（挑词 → 取命中）。独立于 dense/sparse/精确，可并行执行。"""
    anchors = _anchor_terms(norm_query)
    if not anchors:
        return [], []
    return _anchor_hits(anchors, vec, flt, limit), anchors


# 通道并行池：dense/sparse/精确/锚点四条通道互不依赖，串行发 HTTP 纯属白等。
# 实测（tools/_probe_latency.py）串行时 hybrid 中位 144ms vs semantic 81ms（+78%），
# 并行后主要耗时收敛到最慢一条通道。
_channel_pool: Optional["ThreadPoolExecutor"] = None


def _get_channel_pool() -> "ThreadPoolExecutor":
    global _channel_pool
    if _channel_pool is None:
        _channel_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb-channel")
    return _channel_pool


def _fuse(channels: dict, exact_info: dict) -> list[tuple]:
    """加权 RRF 融合。

    Qdrant 内置 `FusionQuery(RRF)` 不支持按通道加权（实测各通道等权），
    而设计要 dense(0.7)+sparse(0.3)+精确置顶，故在客户端自算：
    score = Σ w_c · 1/(k + rank_c)。k=1 与 Qdrant 内置 RRF 行为一致，
    单通道命中最高 0.5、双通道 1.0——这个量级差异正好可用于低置信判定。
    """
    weights = {"dense": DENSE_WEIGHT, "sparse": SPARSE_WEIGHT, "exact": EXACT_WEIGHT,
               "anchor": ANCHOR_WEIGHT}
    acc: dict = {}
    for name, hits in channels.items():
        w = weights.get(name, 1.0)
        for rank, h in enumerate(hits, 1):
            d = acc.setdefault(h.id, {"hit": h, "score": 0.0, "channels": []})
            d["score"] += w * (1.0 / (FUSION_K + rank))
            d["channels"].append(name)
            if d["hit"].score <= h.score:
                d["hit"] = h
    out = [(d["hit"], d["score"], d["channels"], exact_info.get(d["hit"].id, 0.0))
           for d in acc.values()]
    out.sort(key=lambda x: -x[1])
    return out


def _recency_factor(p: dict) -> tuple[float, str]:
    """时效衰减（设计 §6.3）：6 个月内线性衰减到 0.85 后不再衰减。

    **口径（2026-10-01 修正，A 方案）**：只对 `kind ∈ RECENCY_KINDS` 的「结论类」文档生效
    （note / defect_log / error_faq / model_fact_card / version_matrix）——设计里 `doc_date` 的
    定义本就是「**结论形成日期**」，这类条目的新旧才直接决定结论是否仍成立。

    教程/手册/API 参考/站点文档恒定 1.0，理由：
      ① 「教程过时」的正确表达是**版本**（`applies_to.framework` + `version=` 精确过滤 +
         `⚠version-drift`），不是日期；用日期降权会误伤仍有效的 Legacy 教程；
      ② 本库 `doc_date` 全部来自站点抓取日（五个值、跨度 5 天），同质日期 + 统一因子
         **不改变相对排序**，只会在近分位制造伪差异。

    豁免：`date_inferred`（日期由 mtime/抓取日推断，不是文档真实日期，不该因此被降权）。
    """
    if p.get("kind") not in RECENCY_KINDS:
        return 1.0, ""
    if p.get("date_inferred"):
        return 1.0, ""
    d = p.get("doc_date")
    if not d:
        return 1.0, ""
    try:
        doc = datetime.strptime(str(d), "%Y%m%d")
    except ValueError:
        return 1.0, ""
    days = (datetime.now() - doc).days
    if days <= 0:
        return 1.0, ""
    factor = max(RECENCY_FLOOR, 1.0 - (1.0 - RECENCY_FLOOR) * (days / RECENCY_DAYS))
    return factor, (f"时效×{factor:.2f}" if factor < 0.999 else "")


def _postprocess(ranked: list, top_k: int) -> list[dict]:
    """排序后处理（设计 §6.3）：TOC 降权 → trust 权重 → 时效衰减 → 单文档最多 2 条"""
    items = []
    for hit, score, channels, exactness in ranked:
        p = hit.payload or {}
        reasons = []
        if p.get("is_toc"):
            score *= TOC_PENALTY
            reasons.append(f"目录页降权×{TOC_PENALTY}（不占 Top3）")
        tw = TRUST_WEIGHT.get(p.get("trust"))
        if tw and tw != 1.0:
            score *= tw
            reasons.append(f"可信度{p.get('trust')}×{tw}")
        rf, rnote = _recency_factor(p)
        if rf != 1.0:
            score *= rf
            reasons.append(rnote)
        items.append({"hit": hit, "payload": p, "score": score, "channels": channels,
                      "exactness": exactness, "reasons": reasons})

    items.sort(key=lambda x: -x["score"])

    kept: list[dict] = []
    per_source: dict[str, int] = {}
    deferred_toc: list[dict] = []

    def _try_add(it: dict) -> None:
        if len(kept) >= top_k:
            return
        src = (it["payload"] or {}).get("source", "")
        if per_source.get(src, 0) >= MAX_PER_SOURCE:
            return
        per_source[src] = per_source.get(src, 0) + 1
        kept.append(it)

    # 目录页（链接列表型的索引页）降权 ×0.6 后仍可能因语义相近挤进 Top3
    # （实测 q04/q24 仍居首），故再补一条硬约束：目录页不占前 3 位，
    # 先让非目录页占位，目录页顺延到后面——保留召回，但不占最优位置。
    for it in items:
        if (it["payload"] or {}).get("is_toc") and len(kept) < 3:
            deferred_toc.append(it)
            continue
        _try_add(it)
    for it in deferred_toc:
        _try_add(it)
    return kept


# ==================== 工具实现 ====================

def _fmt_date(v) -> str:
    """payload 的 doc_date 存 int(YYYYMMDD)，展示为 YYYY-MM-DD"""
    s = str(v or '')
    return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 and s.isdigit() else s


def _search_structured(args: dict) -> tuple[list[dict], dict]:
    """检索主流程（结构化输出）：三通道融合 + 过滤 + 后处理。

    与渲染解耦，便于验收脚本直接拿 payload 做量化断言（T3/T4），
    P3 的 kb_get_doc / kb_dupes 等工具也复用这一层。
    返回 (items, meta)；items 为空表示无结果。
    """
    query = (args.get("query") or "").strip()
    if not query:
        return [], {"error": "query 不能为空"}

    mode = (args.get("mode") or "hybrid").lower()
    version = (args.get("version") or "").strip()
    doc_type = args.get("doc_type", "all")
    section = (args.get("section") or "").strip()
    model = (args.get("model") or "").strip()
    kind = (args.get("kind") or "").strip()
    path_prefix = (args.get("path_prefix") or "").strip()
    canonical_only = args.get("canonical_only", True)
    trust = args.get("trust")
    top_k = max(1, min(int(args.get("top_k") or DEFAULT_TOPK), 20))
    pool = max(top_k * CANDIDATE_POOL, 15)
    norm_query = normalize_for_search(query)

    notes: list[str] = []
    vec = _embed_query(query)
    drift = False

    def _add_note(msg: str) -> None:
        """去重追加（drift 场景会重试检索一次，避免同一条提示重复出现）"""
        if msg not in notes:
            notes.append(msg)

    def _retrieve(with_version: bool) -> list:
        flt, fnotes = _build_filter(doc_type, section, version if with_version else "",
                                    model, trust, kind, path_prefix, canonical_only)
        for n in fnotes:
            _add_note(n)

        if mode == "exact":
            ex = _exact_hits(norm_query, vec, flt, pool)
            return [(h, cov, ["exact"], cov) for h, cov in ex]

        if mode == "semantic":
            th = args.get("score_threshold")
            th = float(th) if th is not None else DEFAULT_THRESHOLD
            return [(h, h.score, ["dense"], 0.0) for h in _dense_hits(vec, flt, pool, th)]

        # hybrid：dense 预筛阈值 + sparse 全量 + 精确串预筛 + 术语锚点，四通道加权融合。
        # 四条通道互不依赖 → 并发发出（T11：串行时 hybrid 相对单通道 +78%，超出 <30% 验收线）
        th = args.get("score_threshold")
        th = float(th) if th is not None else HYBRID_DENSE_THRESHOLD
        sv = _sparse_query_vec(query)          # 纯 token 哈希，0.1ms 级，留在主线程
        ex_ = _get_channel_pool()
        f_dense = ex_.submit(_dense_hits, vec, flt, pool, th)
        f_sparse = ex_.submit(_sparse_hits, sv, flt, pool) if sv is not None else None
        f_exact = ex_.submit(_exact_hits, norm_query, vec, flt, pool)
        f_anchor = ex_.submit(_anchor_channel, norm_query, vec, flt, pool)

        channels: dict = {"dense": f_dense.result()}
        if f_sparse is not None:
            sh = f_sparse.result()
            if sh is None:
                _add_note(f"⚠ 稀疏通道不可用（集合缺少 `{SPARSE_VECTOR_NAME}` 配置），"
                          f"本次降级为 dense+精确+锚点")
            else:
                channels["sparse"] = sh
        else:
            _add_note("⚠ 稀疏模型不可用，本次降级为 dense+精确+锚点")

        ex = f_exact.result()
        if ex:
            channels["exact"] = [h for h, _ in ex]

        # 术语锚点通道：查询含明确标识符（如 RackConfigurator）时，字面命中的片段强力加权，
        # 避免被 dense 的"语义相近但不含该标识符"的片段压住（D5）
        ah, anchors = f_anchor.result()
        if ah:
            channels["anchor"] = ah
            _add_note(f"术语锚点:{'/'.join(anchors)}")

        return _fuse(channels, {h.id: cov for h, cov in ex})

    ranked = _retrieve(with_version=bool(version))
    if not ranked and version:
        drift = True
        ranked = _retrieve(with_version=False)

    if not ranked:
        return [], {"mode": mode, "query": query, "notes": notes, "drift": drift,
                    "pool": 0, "version": version, "raw_top": 0.0, "low_conf": ""}

    raw_top = ranked[0][1]
    items = _postprocess(ranked, top_k)

    # 低置信判定（设计 §6.4）：给可执行的下一步，而不是让 Agent 硬答。
    # hybrid 用「通道数」而不是分数阈值——加权后单通道最高只有 0.35/0.30，
    # 用固定阈值会把「dense+sparse 互相印证」和「单通道」混为一谈。
    top_channels = ranked[0][2]
    if mode == "hybrid" and len(top_channels) < 2:
        low_conf = (f"⚠ 低置信：Top1 仅由 {top_channels[0]} 单通道支持（无第二通道印证）。"
                    f"建议：改用 mode=exact 搜报错原文/类名，或补充更精确的关键词；"
                    f"若仍无可靠结果，应明确告知用户「知识库未覆盖」，不要硬答。")
    elif mode == "semantic" and raw_top < LOW_CONF_SEMANTIC:
        low_conf = (f"⚠ 低置信：最高余弦相似度 {raw_top:.3f} < {LOW_CONF_SEMANTIC}。"
                    f"建议改 mode=hybrid（三通道融合）或补充关键词。")
    else:
        low_conf = ""

    return items, {"mode": mode, "query": query, "notes": notes, "drift": drift,
                   "pool": len(ranked), "version": version, "raw_top": raw_top,
                   "low_conf": low_conf}


# ---------- P3-7：top1 缩略图返图（设计 §11 / 验收 T13） ----------

def _make_thumbnail(abs_path: str) -> Optional[tuple[str, str]]:
    """本地图片 → (base64 JPEG, mime)。Pillow 缺失或解码失败返回 None（不抛错）"""
    try:
        from PIL import Image
    except ImportError:
        logger.warning("未安装 Pillow，返图降级为路径列表（pip install Pillow）")
        return None
    try:
        with Image.open(abs_path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=THUMB_QUALITY)
        return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"
    except Exception as e:
        logger.warning(f"缩略图生成失败 {abs_path}: {e}")
        return None


def _top_image_blocks(args: dict, items: list) -> list:
    """top1 命中含本地图片时附首图缩略图。

    任何一步失败都只是"不附图"，绝不抛错——纯文本客户端因此天然降级为
    现有的路径列表（T13 断言：降级不报错）。
    """
    if not args.get("attach_image", True) or not items:
        return []
    p = items[0]["payload"]
    refs = p.get("image_refs") or []
    if not refs:
        return []
    paths = resolve_image_paths(p.get("source", ""), refs)
    local = [x for x in paths if not x.lower().startswith(("http://", "https://"))]
    existing = [x for x in local if os.path.exists(x)]
    if not existing:
        return []
    # 2026-10-01：取**体积最大**的本地图而不是第一张——页面首图常是页首小图标
    # （实测 Analytics.md：icon 2.6KB vs 对话框截图 65KB，取第一张会拿到没信息量的图标）。
    # 体积拿不到时退回原顺序。
    def _size(x):
        try:
            return os.path.getsize(x)
        except OSError:
            return -1

    ap = max(existing, key=_size)
    thumb = _make_thumbnail(ap)
    if not thumb:
        return []
    data, mime = thumb
    return [
        ImageContent(type="image", data=data, mimeType=mime),
        TextContent(type="text", text=(
            f"🖼 以上为 top1 片段**主要插图**缩略图（{os.path.basename(ap)}，"
            f"该片段本地图 {len(existing)} 张里体积最大的一张，≤{THUMB_MAX_SIDE}px）。"
            f"原图：{ap}\n**图里的信息请直接读图作答**，不要再假设自己看不到图片。")),
    ]


async def _handle_search(args: dict) -> Sequence[TextContent]:
    """MCP 工具入口：调用 _search_structured 并渲染返回体 v2"""
    items, meta = _search_structured(args)
    if meta.get("error"):
        return [TextContent(type="text", text=meta["error"] + "。")]

    mode, version = meta["mode"], meta.get("version", "")
    if not items:
        logger.warning(f"检索无结果: query={meta['query']!r} mode={mode} version={version!r}")
        if mode == "exact":
            msg = (f"知识库中没有与「{meta['query']}」**完全一致**的片段——该原文/标识符未被收录"
                   f"（错误串类问题常见：报错来自插件或反编译产物，未进文档库）。"
                   f"建议：改用 mode=hybrid 找语义相近的段落，或改用更短的关键标识符片段（类名/方法名）重试。")
        else:
            msg = (f"未在知识库中找到相关文档片段（模式:{mode}）。"
                   f"建议：换关键词、用 mode=exact 搜报错原文/类名，或放宽 version/trust 过滤条件。")
        return [TextContent(type="text", text=msg)]

    head = [f"## 检索结果（模式:{mode}｜候选池:{meta['pool']}｜返回:{len(items)}）"]
    if meta["notes"]:
        head.append("说明: " + "；".join(meta["notes"]))
    if meta["drift"]:
        head.append(f"⚠ version-drift：未找到适用 {version} 的条目，以下为其他版本片段，"
                    f"回答时必须说明版本差异。")
    if meta["low_conf"]:
        head.append(meta["low_conf"])

    chunks = []
    for i, it in enumerate(items, 1):
        hit, p = it["hit"], it["payload"]
        refs = p.get('image_refs') or []
        extra = []
        if p.get('page_number'):
            extra.append(f"页码:{p['page_number']}")
        if refs or p.get('has_image'):
            extra.append(f"含图×{len(refs)}" if refs else "含图")
        extra_str = (' | ' + ' | '.join(extra)) if extra else ''

        # 分数口径按 mode 分离（G-E/D12）：hybrid=RRF 融合分，semantic=余弦，exact=精确通道分。
        # 标注「×权重」是因为后处理（TOC/可信度/时效）会乘在分数上，下面的 ⚙ 会列出具体倍率。
        if mode == "semantic":
            score_label = f"相似度分:{it['score']:.3f}(余弦×权重)"
        elif mode == "exact":
            score_label = f"精确分:{it['score']:.3f}(×权重)"
        else:
            score_label = f"融合分:{it['score']:.4f}(RRF×权重)"

        flags = []
        if it["exactness"] >= 1.0:
            flags.append("精确串命中")
        elif it["exactness"] > 0:
            flags.append(f"精确覆盖{it['exactness']:.0%}")
        if "anchor" in it["channels"]:
            flags.append("含查询标识符")
        flags.extend(it["reasons"])
        flag_str = (" | ⚙" + "；".join(flags)) if flags else ''

        # 元数据行 v2（版本/日期/可信度/正本）：缺字段自动省略，老切片无感
        meta_bits = []
        if p.get('framework_version') and p['framework_version'] != 'unknown':
            drift_tag = " ⚠version-drift" if (version and p['framework_version'] != version) else ""
            meta_bits.append(f"适用版本:{p['framework_version']}{drift_tag}")
        if p.get('doc_date'):
            inferred = "(推断)" if p.get('date_inferred') else ""
            meta_bits.append(f"文档日期:{_fmt_date(p['doc_date'])}{inferred}")
        if p.get('trust'):
            meta_bits.append(f"可信度:{p['trust']}")
        if p.get('kind') and p['kind'] != 'doc':
            meta_bits.append(f"类型:{p['kind']}")
        if p.get('is_canonical'):
            meta_bits.append("正本:是")
        elif p.get('superseded_by'):
            meta_bits.append(f"⚠已被取代:{p['superseded_by']}")
        meta_line = ('\n' + ' | '.join(meta_bits)) if meta_bits else ''

        # 清洗 YAML frontmatter，只展示正文
        text_body = p.get('text', '')
        text_body = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text_body, flags=re.DOTALL).strip()

        # 图片块拼在正文截断之后：路径绝不会被 1000 字符上限切成残缺路径（G-H 影响 5）
        img_block = ''
        if refs:
            abs_paths = resolve_image_paths(p.get('source', ''), refs)
            local = [x for x in abs_paths if not x.lower().startswith(('http://', 'https://'))]
            external_urls = [x for x in abs_paths if x.lower().startswith(('http://', 'https://'))]
            found = [x for x in local if os.path.exists(x)]
            missing = len(local) - len(found)
            shown = found[:5]
            lines = [f"📎 本片段含 {len(refs)} 张图，回答涉及图示内容前**必须**先查看："]
            if p.get('image_captions'):
                lines.append(f"  图注（弱，自动生成）: {p['image_captions']}")
            lines += [f"  {j}. {ap}" for j, ap in enumerate(shown, 1)]
            if len(found) > len(shown):
                lines.append(f"  （其余 {len(found) - len(shown)} 张在同目录，先看上面这些）")
            if missing:
                lines.append(f"  ⚠ 另有 {missing} 张本地链接指向的路径已不存在（历史文档失效链接）")
            if external_urls:
                # 只报数量不给 URL 会让 Agent 无法自救（旧文案还写着"可直接抓取该 URL"）；
                # 实测这批 sim3d `docfetch.php` 链接匿名请求多为 404（服务器端已缺失），故如实说明。
                lines.append(f"  ℹ 另有 {len(external_urls)} 张**外链图**（需联网抓取；实测这批链接多为"
                             f"服务器端已失效，返回 404）：")
                lines += [f"    - {u}" for u in external_urls[:3]]
                if len(external_urls) > 3:
                    lines.append(f"    （其余 {len(external_urls) - 3} 张同类）")
            img_block = '\n' + '\n'.join(lines)
        elif p.get('has_image'):
            page = f"第 {p['page_number']} 页" if p.get('page_number') else "对应页"
            img_block = f"\n📎 本片段含图（PDF/DOCX 内嵌图），涉及图示内容前**必须**打开源文件查看{page}。"

        chunks.append(
            f"[片段{i}] {score_label} | 通道:{'+'.join(it['channels'])}{flag_str} | "
            f"来源:{p.get('source','未知')}{extra_str}\n"
            f"标题:{p.get('title','')} | 章节:{p.get('heading','')}{meta_line}\n"
            f"{text_body[:1000]}{img_block}"
        )
    out: list[Any] = [TextContent(type="text", text="\n".join(head) + "\n\n" + "\n---\n".join(chunks))]
    out.extend(_top_image_blocks(args, items))
    return out


# ---------- 治理工具支撑层（P1-6：kb_list_docs / kb_meta_lint） ----------

_FM_RE = re.compile(r'^\s*---\s*\n(.*?)\n---\s*\n', re.DOTALL)

# 治理视图关心的 payload 字段（老切片没有的字段自动为空）
_DOC_KEYS = ["source", "title", "doc_type", "product", "trust", "framework_version",
             "doc_date", "kind", "is_canonical", "canonical_uid", "section", "superseded_by"]


def _read_source_meta(source_file: str) -> tuple[dict, str]:
    """读取文档元数据：MD 取 front-matter，PDF/DOCX/TXT 取 sidecar `<名>.meta.yaml`"""
    if source_file.lower().endswith('.md'):
        try:
            with open(source_file, encoding='utf-8', errors='replace') as f:
                m = _FM_RE.match(f.read())
            return (parse_meta_yaml(m.group(1)) if m else {}), "front-matter"
        except OSError:
            return {}, "front-matter"
    return load_sidecar_meta(source_file), "sidecar"


def _scroll_docs() -> dict[str, dict]:
    """按 source 聚合全库 payload（治理工具用；一次全扫，结果供两个工具复用）"""
    qdrant = get_qdrant()
    docs: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=1000, offset=offset,
            with_payload=_DOC_KEYS, with_vectors=False,
        )
        for pt in points:
            p = pt.payload or {}
            src = p.get("source") or "未知"
            d = docs.setdefault(src, {"chunks": 0, "v2_chunks": 0})
            d["chunks"] += 1
            if p.get("trust") and p.get("doc_date"):
                d["v2_chunks"] += 1
            for k in _DOC_KEYS[1:]:
                if k not in d and p.get(k) not in (None, "", []):
                    d[k] = p[k]
        if offset is None:
            break
    return docs


async def _handle_kb_list_docs(args: dict) -> Sequence[TextContent]:
    doc_type = args.get("doc_type", "all")
    trust = (args.get("trust") or "").strip()
    section = (args.get("section") or "").strip().lower()
    keyword = (args.get("keyword") or "").strip().lower()
    limit = max(1, int(args.get("limit", 50)))

    docs = _scroll_docs()
    rows = []
    for src, d in docs.items():
        if doc_type != "all" and d.get("doc_type") != doc_type:
            continue
        if trust and d.get("trust") != trust:
            continue
        if section and section not in str(d.get("section", "")).lower():
            continue
        if keyword and keyword not in (src + " " + str(d.get("title", ""))).lower():
            continue
        rows.append((src, d))
    rows.sort(key=lambda kv: -kv[1]["chunks"])

    total, shown = len(rows), rows[:limit]
    lines = [
        f"## 文档清单（命中 {total} 篇，显示前 {len(shown)}，按片段数降序）",
        "",
        "| # | 文档 | 片段 | 类型 | 适用版本 | 文档日期 | 可信度 | 正本 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, (src, d) in enumerate(shown, 1):
        canon = {True: "是", False: "否"}.get(d.get("is_canonical"), "-")
        lines.append(
            f"| {i} | {os.path.basename(src)} | {d['chunks']} | {d.get('doc_type', '-')} | "
            f"{d.get('framework_version', '-') or '-'} | {_fmt_date(d.get('doc_date')) or '-'} | "
            f"{d.get('trust', '-') or '-'} | {canon} |"
        )
    if total > len(shown):
        lines.append(f"\n（其余 {total - len(shown)} 篇可用 keyword / section / doc_type 过滤或提高 limit）")
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_kb_meta_lint(args: dict) -> Sequence[TextContent]:
    limit = max(1, int(args.get("limit", 30)))
    top_docs = int(args.get("top_docs", 0))  # 0 = 全量；>0 只体检片段数最多的 N 篇

    docs = _scroll_docs()
    rows = sorted(docs.items(), key=lambda kv: -kv[1]["chunks"])
    if top_docs > 0:
        rows = rows[:top_docs]

    missing_rows, stale_rows = [], []
    canon_map: dict[str, list[str]] = {}
    for src, d in rows:
        meta, origin = _read_source_meta(src)
        miss = lint_meta(meta)
        if miss:
            missing_rows.append((src, d["chunks"], origin, miss))
        if d["v2_chunks"] < d["chunks"]:
            stale_rows.append((src, d["chunks"] - d["v2_chunks"]))
        uid = d.get("canonical_uid") or get_nested(meta, "canonical_uid")
        if uid:
            canon_map.setdefault(str(uid), []).append(os.path.basename(src))

    scope = f"片段数前 {top_docs} 篇" if top_docs > 0 else "全量"
    lines = [
        "## 元数据体检报告",
        "",
        f"- 检查范围：{len(rows)} 篇文档（{scope}）",
        f"- 源文件元数据缺失：**{len(missing_rows)}** 篇（缺 title/applies_to.framework/doc_date/trust/canonical）",
        f"- 索引字段未回填：**{len(stale_rows)}** 篇（老切片，需重跑入库或 set_payload 回填）",
        "",
    ]

    if missing_rows:
        lines += ["### 1. 缺必填元数据（按片段数排序，优先补高频文档）", "",
                  "| 文档 | 片段 | 元数据来源 | 缺失项 |", "|---|---|---|---|"]
        for src, n, origin, miss in missing_rows[:limit]:
            lines.append(f"| {os.path.basename(src)} | {n} | {origin} | {', '.join(miss)} |")
        if len(missing_rows) > limit:
            lines.append(f"\n（其余 {len(missing_rows) - limit} 篇略，提高 limit 查看）")
        lines.append("")

    if stale_rows:
        lines += ["### 2. 索引字段未回填（重跑入库窗口一次解决）", "",
                  "| 文档 | 未回填片段数 |", "|---|---|"]
        for src, n in sorted(stale_rows, key=lambda x: -x[1])[:limit]:
            lines.append(f"| {os.path.basename(src)} | {n} |")
        lines.append("")

    dupes = {k: v for k, v in canon_map.items() if len(v) > 1}
    if dupes:
        lines += ["### 3. canonical_uid 冲突（副本组，正本合并用）", ""]
        for k, v in list(dupes.items())[:limit]:
            lines.append(f"- `{k}`: {', '.join(v)}")
        lines.append("")

    if not missing_rows and not stale_rows and not dupes:
        lines.append("✅ 体检通过：必填元数据齐全、索引字段已回填、无 canonical 冲突。")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_kb_stats(args: dict) -> Sequence[TextContent]:
    """库统计（P2-7 / D11）：默认走 count(过滤聚合)，不做全量 scroll；detail=true 才全扫。

    全量 scroll 在 10K 点上要秒级且随库增长线性变差，而 count 走 payload 索引是常数级。
    """
    detail = bool(args.get("detail", False))
    qdrant = get_qdrant()
    ci = qdrant.get_collection(COLLECTION_NAME)

    def _c(key=None, value=None, must_not=None) -> int:
        flt = None
        if key is not None:
            flt = qm.Filter(
                must=[qm.FieldCondition(key=key, match=qm.MatchValue(value=value))],
                must_not=([qm.FieldCondition(key=must_not[0], match=qm.MatchValue(value=must_not[1]))]
                          if must_not else None),
            )
        return qdrant.count(collection_name=COLLECTION_NAME, count_filter=flt, exact=True).count

    index_time = "未知"
    try:
        index_state_file = os.path.join(PROJECT_ROOT, "index_state.json")
        if os.path.exists(index_state_file):
            index_time = datetime.fromtimestamp(os.path.getmtime(index_state_file)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    sparse_names = ", ".join((ci.config.params.sparse_vectors or {}).keys()) or "无（仅 dense）"
    lines = [
        "## 知识库概览",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 集合名称 | `{COLLECTION_NAME}` |",
        f"| 稠密向量 | {ci.config.params.vectors.size} 维 / {ci.config.params.vectors.distance} |",
        f"| 稀疏向量 | {sparse_names} |",
        f"| 数据点数 | {ci.points_count} |",
        f"| 段数量 | {ci.segments_count} |",
        f"| 最后索引 | {index_time} |",
        "",
        "### 文档类型分布（count 聚合，未全扫）",
    ]
    for dt in ("api_reference", "tutorial", "user_manual", "design_spec", "release_note"):
        n = _c("doc_type", dt)
        if n:
            lines.append(f"- **{dt}**: {n}")

    lines += ["", "### 可信度分布"]
    for t in ("measured", "tutorial", "pending"):
        n = _c("trust", t)
        if n:
            lines.append(f"- **{t}**: {n}")

    def _c_notempty(key: str) -> int:
        """字段存在且非空（P4-3①：图注候选口径用 image_refs，不用 has_image）"""
        try:
            return qdrant.count(collection_name=COLLECTION_NAME, count_filter=qm.Filter(
                must_not=[qm.IsEmptyCondition(is_empty=qm.PayloadField(key=key))]),
                exact=True).count
        except Exception:
            return -1

    lines += [
        "",
        "### 治理指标",
        f"- 含图片段（`image_refs` 非空）: {_c_notempty('image_refs')}",
        f"- 已生成图注（`image_captions` 非空）: {_c_notempty('image_captions')}",
        f"- 目录页片段（检索已降权×{TOC_PENALTY}）: {_c('is_toc', True)}",
        f"- 非正本片段（检索默认排除）: {_c('is_canonical', False)}",
    ]

    if not detail:
        lines += ["", "_提示：需要产品分布/章节分布/段详情时传 `detail=true`（会全量扫描，较慢）。_"]
        return [TextContent(type="text", text="\n".join(lines))]

    # detail=true：全量扫描（原行为 + 更多维度）
    type_counts: dict[str, int] = {}
    product_counts: dict[str, int] = {}
    section_counts: dict[str, int] = {}
    offset = None
    total_scanned = 0
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=1000, offset=offset,
            with_payload=["doc_type", "product", "section", "trust"], with_vectors=False,
        )
        for pt in points:
            p = pt.payload or {}
            type_counts[p.get("doc_type", "unknown")] = type_counts.get(p.get("doc_type", "unknown"), 0) + 1
            product_counts[p.get("product", "unknown")] = product_counts.get(p.get("product", "unknown"), 0) + 1
            sec = p.get("section") or "(无)"
            section_counts[sec] = section_counts.get(sec, 0) + 1
        total_scanned += len(points)
        if offset is None:
            break

    lines += ["", f"### 产品分布（全扫 {total_scanned} 点）"]
    for pr, n in sorted(product_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- **{pr}**: {n}")
    lines += ["", "### 章节分布 Top10"]
    for sec, n in sorted(section_counts.items(), key=lambda kv: -kv[1])[:10]:
        lines.append(f"- **{sec}**: {n}")

    return [TextContent(type="text", text="\n".join(lines))]

# ==================== P3 知识类型与入库通道（设计 §8/§9） ====================

_CHUNK_KEYS = ["source", "title", "text", "chunk_index", "chunk_total", "heading",
               "heading_path", "section", "doc_type", "product", "trust", "kind",
               "framework_version", "doc_version", "doc_date", "revision",
               "is_canonical", "superseded_by", "page_number", "image_refs"]


def _invalidate_caches():
    """入库/写盘后必须调用：source/section 是进程内缓存，不清会看不到新文档"""
    global _source_cache, _section_cache
    _source_cache = None
    _section_cache = None


def _norm_path(s: str) -> str:
    return (s or "").replace("/", "\\").strip().strip('"')


def _resolve_source(source: str) -> tuple[Optional[str], list[str]]:
    """路径/文件名/片段 → 全库唯一的 source。返回 (命中或 None, 候选列表)"""
    srcs = _all_sources()
    q = _norm_path(source)
    if not q:
        return None, []
    low = q.lower()
    for s in srcs:                                                    # ① 完全一致
        if _norm_path(s).lower() == low:
            return s, []
    base = os.path.basename(q).lower()
    cands = [s for s in srcs if os.path.basename(s).lower() == base]  # ② 文件名一致
    if len(cands) == 1:
        return cands[0], []
    if not cands:                                                     # ③ 路径片段
        cands = [s for s in srcs if low in s.lower()]
    return (cands[0], []) if len(cands) == 1 else (None, cands)


def _scroll_source(source: str) -> tuple[dict, list[dict]]:
    """取某 source 的全部切片（按 chunk_index 排序），返回 (元数据, 切片列表)"""
    qdrant = get_qdrant()
    flt = qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))])
    out: list[dict] = []
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=500, offset=offset,
            scroll_filter=flt, with_payload=_CHUNK_KEYS, with_vectors=False,
        )
        out.extend(pt.payload or {} for pt in points)
        if offset is None:
            break
    out.sort(key=lambda p: (p.get("chunk_index") or 0, p.get("page_number") or 0))
    meta = {k: v for k, v in (out[0] if out else {}).items() if k != "text"}
    return meta, out


def _strip_fm(text: str) -> str:
    return re.sub(r'^---\s*\n.*?\n---\s*\n', '', text or '', flags=re.DOTALL).strip()


def _doc_meta_line(meta: dict, n_chunks: int) -> str:
    bits = [f"片段数:{n_chunks}"]
    if meta.get("doc_type"):
        bits.append(f"类型:{meta['doc_type']}")
    if meta.get("kind") and meta["kind"] != "doc":
        bits.append(f"知识类型:{meta['kind']}")
    if meta.get("framework_version") and meta["framework_version"] != "unknown":
        bits.append(f"适用版本:{meta['framework_version']}")
    if meta.get("revision"):
        bits.append(f"revision:{meta['revision']}")
    if meta.get("doc_date"):
        bits.append(f"文档日期:{_fmt_date(meta['doc_date'])}")
    if meta.get("trust"):
        bits.append(f"可信度:{meta['trust']}")
    bits.append("正本:是" if meta.get("is_canonical") else
                (f"⚠副本→{meta['superseded_by']}" if meta.get("superseded_by") else "正本:-"))
    return " | ".join(bits)


def _select_heading(chunks: list[dict], heading: str) -> list[dict]:
    """取目标章节及其子章节（按 heading_path 前缀收敛）"""
    h = heading.lower()
    start = -1
    for i, c in enumerate(chunks):
        hp = str(c.get("heading_path") or c.get("heading") or "")
        if h in hp.lower():
            start = i
            break
    if start < 0:
        return []
    base = str(chunks[start].get("heading_path") or "")
    depth = base.count(" > ")
    sel = [chunks[start]]
    for c in chunks[start + 1:]:
        hp = str(c.get("heading_path") or "")
        if not hp:
            sel.append(c)
            continue
        if hp.count(" > ") <= depth and not hp.startswith(base):
            break
        sel.append(c)
    return sel


def _parser_version() -> str:
    """PARSER_VERSION 的权威定义在 index_docs（变更它会触发全量重嵌）"""
    try:
        from index_docs import PARSER_VERSION
        return PARSER_VERSION
    except Exception:
        return "unknown"


def _normalize_ingest(text: str, kind: str, meta: dict, title_hint: str = "") -> tuple[str, dict]:
    """把待入库文本规范化成目标 kind 的 front-matter（草稿转正的关键一步）。

    规则（顺序即优先级，后者覆盖前者）：
      1. 该 kind 的默认元数据（含默认 trust / source_origin / ingested_by）
      2. 原文 front-matter 里已填的值（空值忽略）
      3. 调用方显式传入的 meta
      4. `kind` 强制为请求值；`canonical` 默认 true（入库即正本，
         除非 meta 显式给 canonical=false）；草稿遗留的 `pending` 回退到 kind 默认
    正文一律保持原样，只换头——避免"转正"时悄悄改内容。
    """
    fm, _body = kbt._split_front_matter(text)
    src = parse_meta_yaml(fm.strip().strip('-').strip()) if fm else {}
    merged = kbt.default_fields(kind, meta, title=title_hint or str(src.get("title") or ""))
    for k, v in src.items():
        if v not in (None, "", [], {}):
            merged[k] = v
    for k, v in (meta or {}).items():
        if v not in (None, ""):
            merged[k] = v
    merged["kind"] = kind
    if not (meta and "canonical" in meta):
        merged["canonical"] = True
    if merged.get("trust") in (None, "", "pending") and not (meta and meta.get("trust")):
        merged["trust"] = kbt.KINDS[kind][2]
    body = _strip_fm(text)
    return kbt.render(kind, merged, body=body if body else None), merged


async def _handle_kb_get_doc(args: dict) -> Sequence[TextContent]:
    source = args.get("source", "")
    heading = (args.get("heading") or "").strip()
    try:
        budget = max(500, int(args.get("budget", DOC_READ_BUDGET)))
    except (TypeError, ValueError):
        budget = DOC_READ_BUDGET

    resolved, cands = _resolve_source(source)
    if not resolved:
        if cands:
            lines = [f"⚠ 「{source}」匹配到 {len(cands)} 个文档，请给更完整的路径：", ""]
            lines += [f"- {s}" for s in cands[:15]]
            return [TextContent(type="text", text="\n".join(lines))]
        return [TextContent(type="text", text=f"未找到文档：{source}（可用 kb_list_docs 查看清单）")]

    meta, chunks = _scroll_source(resolved)
    if not chunks:
        return [TextContent(type="text", text=f"文档 {resolved} 在库中无切片")]

    total = len(chunks)
    if heading:
        picked = _select_heading(chunks, heading)
        if not picked:
            paths = sorted({str(c.get("heading_path") or c.get("heading") or "") for c in chunks})
            lines = [f"⚠ 未找到章节「{heading}」。该文档的章节有：", ""]
            lines += [f"- {p}" for p in paths[:40] if p]
            lines.append("\n（用 kb_outline 看层级结构）")
            return [TextContent(type="text", text="\n".join(lines))]
    else:
        picked = chunks

    head = [f"## {os.path.basename(resolved)}", _doc_meta_line(meta, total),
            f"路径: {resolved}"]
    if heading:
        head.append(f"章节过滤: {heading} → {len(picked)}/{total} 片段")
    head.append("")

    parts, used, truncated = [], 0, False
    for c in picked:
        body = _strip_fm(str(c.get("text") or ""))
        hp = str(c.get("heading_path") or c.get("heading") or "")
        block = (f"\n<!-- {hp} -->\n{body}" if hp else f"\n{body}")
        if used + len(block) > budget:
            keep = max(0, budget - used)
            if keep > 200:
                parts.append(block[:keep])
            truncated = True
            break
        parts.append(block)
        used += len(block)

    if truncated:
        head.append(f"⚠ 已达 {budget} 字符预算，以下为截断内容"
                    f"（可用 heading= 只取某节，或调大 budget）")
    return [TextContent(type="text", text="\n".join(head) + "".join(parts))]


async def _handle_kb_outline(args: dict) -> Sequence[TextContent]:
    source = args.get("source", "")
    try:
        max_depth = max(1, min(6, int(args.get("max_depth", 3))))
    except (TypeError, ValueError):
        max_depth = 3

    resolved, cands = _resolve_source(source)
    if not resolved:
        if cands:
            return [TextContent(type="text", text="⚠ 匹配到多个文档，请给更完整的路径：\n" +
                                "\n".join(f"- {s}" for s in cands[:15]))]
        return [TextContent(type="text", text=f"未找到文档：{source}")]

    meta, chunks = _scroll_source(resolved)
    if not chunks:
        return [TextContent(type="text", text=f"文档 {resolved} 在库中无切片")]

    counts: dict[tuple, int] = {}
    order: list[tuple] = []
    for c in chunks:
        hp = str(c.get("heading_path") or c.get("heading") or "").strip()
        if not hp:
            continue
        parts = [x.strip() for x in hp.split(">") if x.strip()][:max_depth]
        for i in range(1, len(parts) + 1):
            key = tuple(parts[:i])
            if key not in counts:
                counts[key] = 0
                order.append(key)
            counts[key] += 1

    lines = [f"## {os.path.basename(resolved)} — 章节树", _doc_meta_line(meta, len(chunks)),
             f"路径: {resolved}", "",
             f"共 {len(chunks)} 片段 / {len(order)} 个章节节点（括号内为含子章节的片段数）", "```"]
    for key in order:
        lines.append("  " * (len(key) - 1) + f"- {key[-1]} ({counts[key]})")
    lines.append("```")
    lines.append("\n用 kb_get_doc(source=..., heading='章节名') 取该节全文。")
    return [TextContent(type="text", text="\n".join(lines))]


def _render_hits(items: list[dict], title: str, notes: list[str]) -> str:
    lines = [f"## {title}"]
    if notes:
        lines.append("说明: " + "；".join(notes))
    lines.append("")
    for i, it in enumerate(items, 1):
        p = it["payload"]
        kind = p.get("kind", "doc")
        tag = {"error_faq": "✅已登记FAQ", "defect_log": "✅缺陷登记"}.get(kind, kind)
        lines.append(f"[{i}] {tag} | 分数:{it['score']:.4f} | "
                     f"来源:{os.path.basename(p.get('source', ''))} | 章节:{p.get('heading', '')}")
        meta_bits = []
        if p.get("framework_version") and p["framework_version"] != "unknown":
            meta_bits.append(f"版本:{p['framework_version']}")
        if p.get("revision"):
            meta_bits.append(f"revision:{p['revision']}")
        if p.get("trust"):
            meta_bits.append(f"可信度:{p['trust']}")
        if meta_bits:
            lines.append("　" + " | ".join(meta_bits))
        lines.append(_strip_fm(str(p.get("text") or ""))[:1200])
        lines.append("---")
    return "\n".join(lines)


async def _handle_kb_error_lookup(args: dict) -> Sequence[TextContent]:
    text = (args.get("error_text") or "").strip()
    if len(text) < 6:
        return [TextContent(type="text", text=(
            "⚠ 请给出报错原文中**最有辨识度**的一段（≥6 字符）。"
            "只给 'Exception'/'Error' 这类通用词无法定位。"))]
    try:
        top_k = max(1, min(20, int(args.get("top_k", DEFAULT_TOPK))))
    except (TypeError, ValueError):
        top_k = DEFAULT_TOPK
    exact_only = bool(args.get("exact_only", False))

    items, meta = [], {}
    hit_scope = ""
    # ① 优先在已登记的 error_faq 里精确匹配（通道 C）
    for kind in (["error_faq"] if exact_only else ["error_faq", ""]):
        req = {"query": text, "mode": "exact", "top_k": top_k, "canonical_only": True}
        if kind:
            req["kind"] = kind
        items, meta = _search_structured(req)
        if items:
            hit_scope = "error_faq 精确命中" if kind else "全库精确命中（未登记）"
            break
    notes = []
    # ② 仍无 → 退回 hybrid 找语义相近的段落
    if not items and not exact_only:
        items, meta = _search_structured({"query": text, "mode": "hybrid",
                                          "top_k": top_k, "canonical_only": True})
        hit_scope = "hybrid 语义相近（**未登记**，需自行判断）"
        notes.append("库中无「错误串 FAQ」命中——查证后可用 kb_note 登记")
    if meta.get("notes"):
        notes += meta["notes"]
    if not items:
        return [TextContent(type="text", text=(
            f"❌ 知识库中未找到与「{text[:80]}」匹配的内容。\n"
            f"建议：① 只截取报错原文里最独特的一小段重试；"
            f"② 用 kb_note 把本次排查结论登记下来，下次就能命中。"))]
    return [TextContent(type="text", text=_render_hits(items, f"错误串检索（{hit_scope}）", notes))]


async def _handle_kb_note(args: dict) -> Sequence[TextContent]:
    topic = (args.get("topic") or "").strip()
    if not topic:
        return [TextContent(type="text", text="⚠ 需要 topic（一句话主题）")]
    evidence = (args.get("evidence") or "").strip() or \
        "（未提供证据：检索失败的 query / 报错原文 / 实测现象）"
    meta = dict(args.get("meta") or {})
    meta.update({"topic": topic, "evidence": evidence, "title": topic})
    try:
        path = kbt.write_doc("draft", meta, title=topic)
    except OSError as e:
        return [TextContent(type="text", text=f"❌ 草稿写入失败：{e}")]

    with open(path, encoding='utf-8') as f:
        text = f.read()
    issues = kbt.validate_text(text, "draft", meta)
    lines = [
        "## 草稿已落盘",
        f"- 路径：`{path}`",
        f"- 校验：{kbt.summarize(issues)}",
        "",
        "**下一步**：补完「初步结论/待办」后，把 `kind` 改成正式类型"
        "（如 `error_faq`/`defect_log`）并用 `kb_ingest(path=..., kind=..., dry_run=true)` 转正。",
    ]
    for i in issues:
        lines.append(f"- [{i['level']}] {i['rule']}：{i['detail']}")
    return [TextContent(type="text", text="\n".join(lines))]


# ---------- kb_ingest 支撑（P3-2 / P3-8） ----------

def _ingest_index_snapshot() -> list[dict]:
    """库内已入库文档的 (source, doc_hash, title, framework_version, revision) 快照"""
    qdrant = get_qdrant()
    seen: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME, limit=1000, offset=offset,
            with_payload=["source", "doc_hash", "title", "framework_version",
                          "revision", "kind"], with_vectors=False,
        )
        for pt in points:
            p = pt.payload or {}
            src = p.get("source")
            if not src:
                continue
            d = seen.setdefault(src, {})
            for k in ("doc_hash", "title", "framework_version", "revision", "kind"):
                if k not in d and p.get(k) not in (None, ""):
                    d[k] = p[k]
        if offset is None:
            break
    return [dict(v, source=s) for s, v in seen.items()]


def _ingest_report(text: str, kind: str, meta: dict, src_file: Optional[str],
                   snapshot: list[dict]) -> tuple[list[dict], dict]:
    """§9.4 校验报告：本地规则（kb_templates）+ 库里可判定的重复/冲突/待刷新"""
    issues = kbt.validate_text(text, kind, meta)
    doc_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    fm, _ = kbt._split_front_matter(text)
    eff = parse_meta_yaml(fm.strip().strip('-').strip()) if fm else {}
    eff = {**meta, **eff}
    title = str(eff.get("title") or "").strip()
    fw = str(eff.get("framework_version") or eff.get("framework") or "")
    rev = str(eff.get("revision") or "")
    info = {"doc_hash": doc_hash, "title": title, "framework_version": fw, "revision": rev}

    dup = [d for d in snapshot if d.get("doc_hash") == doc_hash]
    if dup:
        issues.append({"level": "warn", "rule": "重复内容",
                       "detail": "库中已有相同文本：" + "、".join(
                           os.path.basename(d["source"]) for d in dup[:3])})

    if title:
        other_ver = [d for d in snapshot
                     if str(d.get("title") or "").strip().lower() == title.lower()
                     and fw and str(d.get("framework_version") or "") not in ("", fw)]
        if other_ver:
            issues.append({"level": "info", "rule": "版本并存",
                           "detail": "同名文档已有其它版本：" + "、".join(
                               f"{os.path.basename(d['source'])}({d.get('framework_version')})"
                               for d in other_ver[:3]) + "——引用时须标注版本"})

    if src_file:
        prev = next((d for d in snapshot
                     if _norm_path(d["source"]).lower() == _norm_path(src_file).lower()), None)
        if prev and rev and prev.get("revision") and prev["revision"] != rev:
            issues.append({"level": "warn", "rule": "revision 变化 → 待刷新",
                           "detail": f"已入库 revision={prev['revision']}，本次 {rev}："
                                     "引用旧结论的事实卡/片段需重新核对"})
    return issues, info


def _plugin_kind_for(name: str) -> str:
    low = name.lower()
    if low in kbt.PLUGIN_PRODUCTS:
        return kbt.PLUGIN_PRODUCTS[low]
    if low.endswith(kbt.QLP_SUFFIXES):
        return "qlp_snippet"
    return ""


def _index_written_file(path: str) -> str:
    """写盘后立即向量化（复用服务端已加载的嵌入器，避免二次加载 e5-large）"""
    import index_docs
    emb = get_embedder()
    sparse = get_sparse_embedder()
    res = index_docs.index_single_file(
        path, embed_fn=emb.embed,
        sparse_embed_fn=(sparse.embed if sparse is not None else None), force=True)
    return f"{res['chunks']} 片段"


def _fmt_report(kind: str, target: str, issues: list[dict], extra: list[str] = None) -> str:
    lines = [f"## kb_ingest 校验报告（kind={kind}）", f"目标：`{target}`", kbt.summarize(issues)]
    if extra:
        lines += [""] + extra
    if issues:
        lines += ["", "| 级别 | 规则 | 明细 |", "|---|---|---|"]
        for i in issues:
            lines.append(f"| {i['level']} | {i['rule']} | {i['detail']} |")
    else:
        lines += ["", "未发现任何问题。"]
    lines += ["", "（dry_run 时不会落盘；确认后传 dry_run=false 真正入库）"]
    return "\n".join(lines)


async def _handle_kb_ingest(args: dict) -> Sequence[TextContent]:
    kind = (args.get("kind") or "").strip()
    payload = args.get("payload")
    path_arg = (args.get("path") or "").strip()
    meta = dict(args.get("meta") or {})
    dry_run = bool(args.get("dry_run", True))
    do_index = bool(args.get("index", True))

    valid_kinds = [k for k in kbt.kind_names() if k != "draft"]
    if kind not in valid_kinds:
        return [TextContent(type="text", text=(
            f"⚠ kind 必须是以下之一：{', '.join(valid_kinds)}"
            "（草稿用 kb_note；转正时改 kind 再走本工具）"))]
    if not payload and not path_arg:
        return [TextContent(type="text", text="⚠ 必须提供 payload（现写正文）或 path（已有文件/目录）之一")]

    if path_arg and os.path.isdir(path_arg):
        return await _ingest_plugin_dir(path_arg, kind, meta, dry_run, do_index)

    src_file = None
    title_hint = meta.get("title") or ""
    if payload:
        raw = payload
    else:
        src_file = os.path.abspath(path_arg)
        if not os.path.exists(src_file):
            return [TextContent(type="text", text=f"⚠ 文件不存在：{src_file}")]
        if os.path.splitext(src_file)[1].lower() != ".md":
            return [TextContent(type="text", text=(
                f"⚠ kb_ingest 只接受 .md（当前 {os.path.basename(src_file)}）。"
                "插件产物（yaml/json/qlp）请传其**所在目录**，工具按 P3-8 契约映射类型。"))]
        with open(src_file, encoding='utf-8', errors='replace') as f:
            raw = f.read()
        if not title_hint:
            title_hint = os.path.splitext(os.path.basename(src_file))[0]

    # 规范化：换头（kind/canonical/trust）+ 保留正文；产物一律写到 kb-inbox，
    # 不改动调用方给的原文件（不可逆地改用户文件更危险）
    text, merged = _normalize_ingest(raw, kind, meta, title_hint)
    target = kbt.target_path(kind, merged.get("title") or title_hint)

    issues, info = _ingest_report(text, kind, meta, src_file, _ingest_index_snapshot())
    extra = [f"- 内容哈希：`{info['doc_hash'][:16]}…`",
             f"- 解析器版本：{_parser_version()}"]
    if src_file:
        extra.append(f"- 来源文件：`{src_file}`（只读，不会被改动）")
    if dry_run:
        return [TextContent(type="text", text=_fmt_report(kind, target, issues, extra))]

    if [i for i in issues if i["level"] == "error"]:
        return [TextContent(type="text", text=(
            _fmt_report(kind, target, issues, extra) +
            "\n\n❌ 存在 error 级问题，已阻止入库（修正后重试）。"))]

    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        out = target
    except OSError as e:
        return [TextContent(type="text", text=f"❌ 写盘失败：{e}")]

    lines = [f"## ✅ 已入库（kind={kind}）", f"- 文件：`{out}`",
             f"- 校验：{kbt.summarize(issues)}"]
    for i in issues:
        lines.append(f"  - [{i['level']}] {i['rule']}：{i['detail']}")
    if do_index:
        try:
            lines.append(f"- 向量化：**{_index_written_file(out)}** 已写入并在检索中生效")
            _invalidate_caches()
        except Exception as e:
            lines.append(f"- ⚠ 向量化失败：{e}（文件已落盘，可稍后重试）")
    else:
        lines.append("- 已跳过向量化（index=false）")
    return [TextContent(type="text", text="\n".join(lines))]


async def _ingest_plugin_dir(dir_path: str, default_kind: str, meta: dict,
                             dry_run: bool, do_index: bool) -> Sequence[TextContent]:
    """P3-8 插件产物契约：按产物名映射 kind（设计 §9.2）"""
    root = os.path.abspath(dir_path)
    mapped, skipped = [], []
    for cur, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() not in _PLUGIN_EXTS:
                continue
            k = _plugin_kind_for(name)
            (mapped.append((os.path.join(cur, name), k)) if k else skipped.append(name))
    if not mapped:
        contract = "\n".join(f"- `{f}` → {k}" for f, k in kbt.PLUGIN_PRODUCTS.items())
        return [TextContent(type="text", text=(
            f"⚠ 目录 `{root}` 中未找到可识别的插件产物。\n契约（文件名 → 知识类型）：\n"
            f"{contract}\n- `*.qlp` → qlp_snippet\n\n"
            + (f"跳过的文件：{', '.join(skipped[:10])}" if skipped else "")))]

    snapshot = _ingest_index_snapshot()
    head = [f"## 插件产物批量入库（{len(mapped)} 个文件，dry_run={dry_run}）", ""]
    body, n_ok, n_err = [], 0, 0
    for fp, kind in mapped:
        title = os.path.basename(fp)
        with open(fp, encoding='utf-8', errors='replace') as f:
            raw = f.read()
        if raw.lstrip().startswith("---"):
            seed = raw
        elif kind == "qlp_snippet":
            seed = f"```qlp\n{raw.strip()}\n```"
        else:
            lang = os.path.splitext(fp)[1].lstrip(".")
            seed = f"```{lang}\n{raw.strip()}\n```"

        text, _merged = _normalize_ingest(seed, kind, {**meta, "title": title}, title)
        issues, _info = _ingest_report(text, kind, {**meta, "title": title}, None, snapshot)
        hard = [i for i in issues if i["level"] == "error"]
        n_err += 1 if hard else 0
        n_ok += 0 if hard else 1
        body.append(f"### {'❌' if hard else '✅'} {title} → {kind}")
        body.append(f"- 校验：{kbt.summarize(issues)}")
        for i in issues:
            body.append(f"  - [{i['level']}] {i['rule']}：{i['detail']}")
        if dry_run or hard:
            continue
        try:
            out = kbt.target_path(kind, title)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
            stat = _index_written_file(out) if do_index else "（跳过向量化）"
            body.append(f"- 落盘：`{out}` → {stat}")
        except Exception as e:
            n_ok -= 1
            n_err += 1
            body.append(f"- ❌ 入库失败：{e}")

    if not dry_run and n_ok:
        _invalidate_caches()
    head.append(f"结果：{n_ok} 个可入库 / {n_err} 个有 error 级问题")
    if skipped:
        head.append(f"跳过（不在契约内）：{', '.join(skipped[:10])}"
                    + ("…" if len(skipped) > 10 else ""))
    return [TextContent(type="text", text="\n".join(head + [""] + body))]


# ---------- kb_dupes（P3-6） ----------

def _iter_md_files(roots: list[str]):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for cur, _dirs, files in os.walk(root):
            for name in files:
                if os.path.splitext(name)[1].lower() == ".md":
                    yield os.path.join(cur, name)


def _canonical_score(path: str) -> tuple:
    """正本判据：① 路径含 Local_agent_kb ② 修改更新 ③ 体积更大"""
    exists = os.path.exists(path)
    return (1 if CANONICAL_HINT in _norm_path(path).lower() else 0,
            os.path.getmtime(path) if exists else 0,
            os.path.getsize(path) if exists else 0)


async def _handle_kb_dupes(args: dict) -> Sequence[TextContent]:
    mode = (args.get("mode") or "hash").strip().lower()
    try:
        limit = max(1, min(200, int(args.get("limit", 30))))
    except (TypeError, ValueError):
        limit = 30
    with_extra = bool(args.get("include_extra_dirs", True))

    roots = [DOCS_ROOT] + (DUPES_EXTRA_DIRS if with_extra else [])
    groups: dict[str, list[str]] = {}

    if mode in ("hash", "name"):
        for fp in _iter_md_files(roots):
            if mode == "name":
                key = os.path.basename(fp).lower()
            else:
                try:
                    with open(fp, 'rb') as f:
                        key = hashlib.sha256(f.read()).hexdigest()
                except OSError:
                    continue
            groups.setdefault(key, []).append(fp)
    else:  # content：库内文本哈希
        for d in _ingest_index_snapshot():
            if d.get("doc_hash"):
                groups.setdefault(d["doc_hash"], []).append(d["source"])

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    scope = "库内已入库文本" if mode == "content" else "；".join(roots)
    if not dupes:
        return [TextContent(type="text", text=f"✅ 未发现副本（模式:{mode}；扫描范围：{scope}）")]

    rows = sorted(dupes.items(), key=lambda kv: -len(kv[1]))[:limit]
    lines = [f"## 副本巡检（模式:{mode}）", f"扫描范围：{scope}",
             f"发现 {len(dupes)} 组副本，显示 {len(rows)} 组。"
             f"**按 D-5 只标记不删**——请人工确认后清理。", ""]
    for key, paths in rows:
        scored = sorted(paths, key=_canonical_score, reverse=True)
        keep, copies = scored[0], scored[1:]
        lines.append(f"### 组（{len(paths)} 份 | {mode}:{key[:12]}…）")
        lines.append(f"- ✅ **正本建议**：`{keep}`")
        for cp in copies:
            size = os.path.getsize(cp) if os.path.exists(cp) else 0
            lines.append(f"- ⚠ 副本：`{cp}`（{size} B）")
        lines += ["", "在副本的 front-matter 里加（然后重跑 kb_ingest 使其生效）：",
                  "```yaml", "canonical: false",
                  f"superseded_by: \"{os.path.basename(keep)}\"",
                  f"canonical_uid: \"{os.path.basename(keep)}\"", "```", ""]
    return [TextContent(type="text", text="\n".join(lines))]


# ---------- kb_caption（P4-3 图注治理） ----------

_CAPTION_TEXT_MARK = "[图注]"
_CAPTION_KEYS = ["source", "title", "heading", "heading_path", "image_refs",
                 "image_captions", "image_caption_source", "text", "doc_type"]
# 图注来源 → 是否「真图注」（人工/VLM/作者所写）而非自动兜底
_REAL_CAPTION_SOURCES = {"explicit", "alt"}

_CAPTION_SUMMARY_MSG = (
    "## kb_caption(mode=summary) 尚未实现\n\n"
    "多模态摘要（读图写结论式图注）属**二期 VLM backlog**，本期不提供——不做「假装读图」的伪摘要。\n\n"
    "本期补真图注的人工路径（写进源文件，可持久）：\n"
    "1. 在被引用的图片**下方 3 行内**写一行显式图注，例如：\n"
    "   `图注：ACR 层级 Bot → Lift → ACRLift → PRH1 → Gripper`\n"
    "   （也支持 `*Caption: …*` 与 `<!-- caption: … -->`）\n"
    "2. 也可以直接写进图片的 alt：`![ACR 层级 Bot → Lift → …](acr.png)`\n"
    "3. 然后 `kb_caption(source=…, mode=extract, dry_run=false, reindex=true)`，\n"
    "   让图注参与嵌入（解析器 2.2 起会以 `[图注] …` 行并入切片文本），检索即可命中（T9）。"
)


def _caption_candidates(limit: int = 20000) -> tuple[list, int, int]:
    """候选切片：**image_refs 非空**。

    P4-3① 修正设计 §8.4 的筛选条件——用 `has_image=true` 在修复前会命中 0 条
    （旧实现 MD 恒 False），现在虽已修复，但图注治理的语义就是「有图可注」，
    统一以 image_refs 为准，返回 (points, 候选数, 涉及文档数)。
    """
    qdrant = get_qdrant()
    flt = qm.Filter(must_not=[qm.IsEmptyCondition(is_empty=qm.PayloadField(key="image_refs"))])
    pts, offset = [], None
    while len(pts) < limit:
        batch, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME, scroll_filter=flt,
            limit=min(1000, limit - len(pts)), offset=offset,
            with_payload=_CAPTION_KEYS, with_vectors=False)
        pts.extend(batch)
        if offset is None:
            break
    return pts, len(pts), len({p.payload.get("source", "") for p in pts})


def _caption_of(payload: dict) -> tuple[str, str]:
    """按解析层同一套取向导，从 payload 推导图注（无需读源文件）。

    与 `document_parsers._image_caption` 共用逻辑：payload 里已有 heading/title/
    heading_path/image_refs，足以复现 heading / filename 两级兜底。
    """
    from document_parsers import _image_caption
    return _image_caption(
        payload.get("heading", ""), [], title=payload.get("title", ""),
        heading_path=payload.get("heading_path") or payload.get("heading", ""),
        refs=payload.get("image_refs") or [])


async def _handle_kb_caption(args: dict) -> Sequence[TextContent]:
    mode = (args.get("mode") or "extract").strip().lower()
    if mode not in ("extract", "summary"):
        return [TextContent(type="text", text=f"未知 mode：{mode}（支持 extract / summary）。")]
    if mode == "summary":
        return [TextContent(type="text", text=_CAPTION_SUMMARY_MSG)]

    dry_run = bool(args.get("dry_run", True))
    reindex = bool(args.get("reindex", False))
    top = int(args.get("top") or 0)
    try:
        show = max(1, min(200, int(args.get("limit", 20))))
    except (TypeError, ValueError):
        show = 20

    pts, n_pts, n_docs = _caption_candidates()

    # source 限定（唯一匹配优先，与 kb_get_doc 同口径）
    scope_note = ""
    src_arg = (args.get("source") or "").strip()
    if src_arg:
        resolved, cands = _resolve_source(src_arg)
        if not resolved:
            return [TextContent(type="text", text=(
                f"未找到匹配 `{src_arg}` 的已入库文档。候选："
                + ("、".join(os.path.basename(c) for c in cands[:5]) if cands else "无")))]
        pts = [p for p in pts if _norm_path(p.payload.get("source", "")) == _norm_path(resolved)]
        scope_note = f" ｜ source 限定 `{os.path.basename(resolved)}`"
        n_pts, n_docs = len(pts), len({p.payload.get("source", "") for p in pts})

    # top-N：按片段数取最高频的 N 篇（与 P4-1 的 Top20 人工窗口对齐）
    if top > 0:
        by_src: dict[str, list] = {}
        for p in pts:
            by_src.setdefault(p.payload.get("source", ""), []).append(p)
        keep = sorted(by_src.items(), key=lambda kv: -len(kv[1]))[:top]
        pts = [p for _s, group in keep for p in group]
        scope_note += f" ｜ 按片段数取 Top{top} 文档"
        n_pts, n_docs = len(pts), len(keep)

    has_cap, in_vector, todo = [], [], []
    marked_docs: set[str] = set()
    for p in pts:
        pl = p.payload or {}
        cap = (pl.get("image_captions") or "").strip()
        if cap:
            has_cap.append(p)
            if _CAPTION_TEXT_MARK in (pl.get("text") or ""):
                in_vector.append(p)
                marked_docs.add(pl.get("source", ""))
        else:
            todo.append(p)

    # 切片级「没带 [图注] 行」不等于该篇没进向量：长段落会被二次切分，图注行只落在其中一片，
    # 兄弟切片同样带 image_captions 但没有那行。真正要告警的是「整篇都没进向量」。
    cap_docs = {p.payload.get("source", "") for p in has_cap}
    stale_docs = cap_docs - marked_docs

    src_dist: dict[str, int] = {}
    for p in has_cap:
        src_dist[p.payload.get("image_caption_source") or "未标记"] = \
            src_dist.get(p.payload.get("image_caption_source") or "未标记", 0) + 1
    real_n = sum(v for k, v in src_dist.items() if k in _REAL_CAPTION_SOURCES)

    stubs: list[tuple[str, str, str]] = []
    for p in todo:
        cap, src = _caption_of(p.payload)
        if cap:
            stubs.append((os.path.basename(p.payload.get("source", "")), cap, src))

    lines = [
        f"## kb_caption（mode=extract{scope_note} ｜ {'dry_run' if dry_run else '写库'}）",
        "",
        f"- 候选口径：`image_refs` **非空**（P4-3①，不用 `has_image`）",
        f"- 候选切片：**{n_pts}** 条 / **{n_docs}** 篇",
        f"- 已有图注：**{len(has_cap)}** 条 / **{len(cap_docs)}** 篇；"
        f"**图注已进向量 {len(cap_docs) - len(stale_docs)} 篇**（整篇未进向量 **{len(stale_docs)}** 篇）",
        f"- 逐切片口径：其中 {len(in_vector)} 条切片自带 `[图注]` 行；其余属"
        f"「同篇兄弟切片携带图注、本片没有该行」的正常情形（长段落二次切分所致）",
        f"- 真图注（explicit/alt 标记）：**{real_n}** 条"
        + (f"；未标记 **{src_dist.get('未标记', 0)}** 条（解析器 2.2 之前写入，"
           f"重跑该篇才会补上来源标记）" if src_dist.get("未标记") else ""),
        f"- 本次可补：**{len(stubs)}** 条 / **{len({p.payload.get('source','') for p in todo})}** 篇"
        f"（阶梯：文档标题·章节路径 / 图片文件名）",
    ]
    if src_dist:
        lines.append("- 图注来源分布：" + "；".join(f"{k}={v}" for k, v in sorted(src_dist.items())))
    if stale_docs:
        lines.append("")
        lines.append(f"> ⚠ **{len(stale_docs)} 篇的图注尚未参与嵌入**（解析器 2.2 之前只写 payload）。"
                     f"带 `reindex=true` 重跑即可按篇增量重嵌使其可被检索命中（T9 前提）。"
                     f"样例：{'、'.join(os.path.basename(s) for s in list(stale_docs)[:3])}")

    if stubs:
        lines += ["", f"### 待补图注样例（共 {len(stubs)} 条，显示前 {min(show, len(stubs))} 条）"]
        for doc_name, cap, src in stubs[:show]:
            lines.append(f"- `{doc_name}` ｜{src}｜ {cap}")
    else:
        lines += ["", "全部候选切片均已有图注——存量抽取已收敛，剩余价值在**人工真图注**"
                      "（`kb_caption(mode=summary)` 属二期 VLM）。"]

    if dry_run:
        lines += ["", "_dry_run=true：未写库。确认后传 `dry_run=false`；"
                      "要让图注参与嵌入再传 `reindex=true`。_"]
        return [TextContent(type="text", text="\n".join(lines))]

    # ---------- 写回 ----------
    qdrant = get_qdrant()
    written = 0
    by_payload: dict[tuple[str, str], list] = {}
    for p in todo:
        cap, src = _caption_of(p.payload)
        if cap:
            by_payload.setdefault((cap, src), []).append(p.id)
    for (cap, src), ids in by_payload.items():
        qdrant.set_payload(collection_name=COLLECTION_NAME,
                           payload={"image_captions": cap, "image_caption_source": src},
                           points=ids)
        written += len(ids)

    affected = sorted({p.payload.get("source", "") for p in todo if p.payload.get("source")})
    lines += ["", "### 写库结果",
              f"- `set_payload` 写回图注：**{written}** 条（{len(by_payload)} 组）",
              f"- 涉及文档：**{len(affected)}** 篇"]

    if reindex:
        # 覆盖范围 = 「图注尚未进入向量」的候选文档：
        #   ① 本次新写回图注的（todo）；② 旧解析器只写了 payload、图注未参与嵌入的（has_cap 里的 stale）。
        # 只重嵌 ① 会漏掉存量主体（实测 1,084 篇），T9 对存量不生效——这就是必须并上 ② 的原因。
        new_cap = {(p.payload.get("source") or "") for p in todo if _caption_of(p.payload)[0]}
        stale = {(p.payload.get("source") or "") for p in has_cap
                 if _CAPTION_TEXT_MARK not in (p.payload.get("text") or "")}
        targets = {s for s in (new_cap | stale) if s}
        need = sorted(s for s in targets if os.path.exists(s))
        lines.append(f"- 增量重嵌（reindex=true）：待重嵌 **{len(targets)}** 篇"
                     f"（本次新写 {len(new_cap)} 篇 / 存量旧图注未参与嵌入 {len(stale)} 篇，去重后 "
                     f"{len(targets)} 篇），命中磁盘 **{len(need)}** 篇")
        ok, fail = 0, []
        for s in need:
            try:
                _index_written_file(s)
                ok += 1
            except Exception as e:
                fail.append(f"{os.path.basename(s)}: {e}")
        _invalidate_caches()
        lines.append(f"- 重嵌成功 {ok} 篇" + (f"，失败 {len(fail)} 篇：{'；'.join(fail[:3])}" if fail else ""))
        lines.append("> 图注已随解析器 2.2 的 `[图注] …` 行参与嵌入，检索可直接命中。")
        if len(need) > 200:
            lines.append(f"> 💡 待重嵌 **{len(need)}** 篇（含图文档全量 ≈86% 的全库切片），逐篇慢于整体："
                         f"直接 `index_docs.py md-source --reindex-all` 一次过完通常更省事"
                         f"（30–60 分钟，顺带刷新所有 2.1 遗留切片）。可先小批 `top=5` 验证再决定。")
    else:
        lines += ["", f"> 未传 `reindex=true`：图注已写 payload（检索返回里会显示为「图注（弱，自动生成）」），"
                      f"但**尚未进入向量**——要让图注文字可被检索命中，需带 `reindex=true` 重跑"
                      f"（或等下一次全库重跑）。"]

    lines += ["", "> 持久性：payload 写入即刻可用；**重跑入库会以源文件为准**。"
                  "要长期保留，请把图注写进源文件的图片下方（`图注：…`）或 alt。"]
    return [TextContent(type="text", text="\n".join(lines))]


async def main():
    logger.info("服务启动（模型按需加载，空闲自动释放）")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
