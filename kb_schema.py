#!/usr/bin/env python3
"""
KB 元数据规范 v2 共享层（设计 §5 / 可行性报告 G-A~G-D）

- front-matter v2 解析（PyYAML，支持嵌套 applies_to）
- PDF/DOCX sidecar 元数据（<文档名>.meta.yaml，G-B）
- 元数据 → payload 字段映射与默认值兜底（R1：降权不过滤）
- version_rank / doc_date 数值化（G-C）
- search_text 规范化（精确串检索通道底座，G-F）
"""
import os
import re
import datetime

try:
    import yaml
except ImportError:  # 容错：未装 PyYAML 时退回朴素行解析
    yaml = None

# ==================== 规范常量（设计 §5.1/§8.1） ====================

TRUST_LEVELS = ("measured", "tutorial", "pending")
REQUIRED_META_KEYS = ("title", "applies_to.framework", "doc_date", "trust", "canonical")
SOURCE_ORIGINS = ("kb", "plugin", "web", "decompile")

DEFAULTS = {
    "kind": "doc",
    "framework_version": "unknown",
    "model_scope": "general",
    "trust": "pending",
    "doc_version": "",
    "canonical_uid": "",
    "ingested_by": "manual",
    "revision": "",
}

_SIDE_CACHE: dict[str, dict] = {}


# ==================== front-matter / sidecar 解析（G-A / G-B） ====================

def _naive_yaml(text: str) -> dict:
    """PyYAML 缺失时的兜底：仅支持扁平 key: value（旧行为）"""
    meta = {}
    for line in text.splitlines():
        if ':' in line and not line.lstrip().startswith('#'):
            k, v = line.split(':', 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta


def parse_meta_yaml(text: str) -> dict:
    """解析 YAML 元数据块；嵌套结构（applies_to.framework）原样保留为 dict。"""
    if not text or not text.strip():
        return {}
    if yaml is None:
        return _naive_yaml(text)
    try:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return _naive_yaml(text)


def load_sidecar_meta(file_path: str) -> dict:
    """读取 <文档路径>.meta.yaml（如 foo.pdf.meta.yaml），与 front-matter 同构。结果带缓存。"""
    if file_path in _SIDE_CACHE:
        return _SIDE_CACHE[file_path]
    meta: dict = {}
    side = file_path + ".meta.yaml"
    if os.path.exists(side):
        try:
            with open(side, encoding='utf-8', errors='replace') as f:
                meta = parse_meta_yaml(f.read())
        except OSError:
            meta = {}
    _SIDE_CACHE[file_path] = meta
    return meta


def get_nested(meta: dict, dotted_key: str, default=""):
    """取嵌套键值，如 applies_to.framework"""
    cur = meta
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
        if cur is None:
            return default
    return cur if cur is not None else default


def _is_set(v) -> bool:
    """「字段已声明」判定：canonical: false 与 doc_version: "" 语义不同——
    前者是明确的「这是副本」，后者才是没写。"""
    return v is not None and v != "" and v != [] and v != {}


def lint_meta(meta: dict, is_model_level: bool = False) -> list[str]:
    """返回缺失必填项清单（设计 §8.1；applies_to.model 在模型级结论中必填）"""
    missing = [k for k in REQUIRED_META_KEYS if not _is_set(get_nested(meta, k))]
    if is_model_level and not _is_set(get_nested(meta, "applies_to.model")):
        missing.append("applies_to.model")
    trust = get_nested(meta, "trust")
    if trust and trust not in TRUST_LEVELS:
        missing.append(f"trust(illegal:{trust})")
    origin = get_nested(meta, "source_origin")
    if origin and origin not in SOURCE_ORIGINS:
        missing.append(f"source_origin(illegal:{origin})")
    return missing


# ==================== 数值化（G-C） ====================

def version_to_rank(version: str) -> int:
    """2.4.0.0 -> 2_004_000_000 级数可比较；无法解析返回 -1"""
    if not version:
        return -1
    parts = re.findall(r'\d+', str(version))
    if not parts:
        return -1
    rank = 0
    for i, p in enumerate(parts[:4]):
        try:
            rank += int(p) * (1000 ** (3 - i))
        except ValueError:
            return -1
    return rank


def date_to_int(date_str: str, mtime: float = None) -> int:
    """'2026-09-27' -> 20260927；缺失回退 mtime 并视为推断（调用方标 date_inferred）"""
    if date_str:
        m = re.search(r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})', str(date_str))
        if m:
            return int(f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}")
    if mtime:
        return int(datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%d"))
    return 0


# ==================== 精确检索文本规范化（G-F，设计 §5.2 search_text） ====================

_QUOTE_RE = re.compile(r'[\"\'\u201c\u201d\u2018\u2019`]|\\+')
_WS_RE = re.compile(r'\s+')


def normalize_for_search(text: str) -> str:
    """去引号/统一空白/保留原始大小写与下划线（错误串归一化的基础）"""
    if not text:
        return ''
    text = _QUOTE_RE.sub(' ', text)
    return _WS_RE.sub(' ', text).strip()


# ==================== 元数据 → payload 映射（设计 §5.2） ====================

def meta_to_payload(meta: dict, *, mtime: float = None, is_model_level: bool = False) -> dict:
    """front-matter / sidecar 元数据展平为 payload 扩展字段。缺失按 R1 兜底。"""
    applies_to = meta.get("applies_to") if isinstance(meta.get("applies_to"), dict) else {}

    framework = str(applies_to.get("framework") or get_nested(meta, "applies_to.framework") or DEFAULTS["framework_version"])
    model = str(applies_to.get("model") or "general")
    doc_date_str = str(meta.get("doc_date") or "")
    doc_date = date_to_int(doc_date_str, mtime)
    trust = str(meta.get("trust") or DEFAULTS["trust"])
    canonical = meta.get("canonical")

    payload = {
        "kind": str(meta.get("kind") or DEFAULTS["kind"]),
        "doc_version": str(meta.get("doc_version") or DEFAULTS["doc_version"]),
        "doc_date": doc_date,
        "framework_version": framework,
        "model_scope": model,
        "trust": trust if trust in TRUST_LEVELS else DEFAULTS["trust"],
        "version_rank": version_to_rank(framework),
        "canonical_uid": str(meta.get("canonical_uid") or DEFAULTS["canonical_uid"]),
        "ingested_by": str(meta.get("ingested_by") or DEFAULTS["ingested_by"]),
        "source_origin": str(meta.get("source_origin") or ""),
        # P3：revision 用于插件产物与事实卡的「待刷新」判定（设计 §11）
        "revision": str(meta.get("revision") or DEFAULTS["revision"]),
    }
    if not doc_date_str and mtime:
        payload["date_inferred"] = True
    # P3：supersedes 记录本文件取代的旧文档（§5.1）；列表用 MatchAny 过滤
    supersedes = meta.get("supersedes")
    if isinstance(supersedes, str):
        supersedes = [supersedes] if supersedes else []
    if supersedes:
        payload["supersedes"] = [str(s) for s in supersedes]
    if canonical is True:
        payload["is_canonical"] = True
    elif canonical is False:
        payload["is_canonical"] = False
        if meta.get("superseded_by"):
            payload["superseded_by"] = str(meta["superseded_by"])
    # is_model_level 仅影响 lint，不改变存储
    return payload


# ==================== 图片链接（G-H） ====================

# ![alt](path "title") —— path 不含空格与括号；捕获 alt 与 path
IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+["\'][^"\']*["\'])?\s*\)')
# HTML 图片标签（实测 2,099 处 / 272 文件 / 占全库文本 1.66%，src 全部无法直接本地解析）
HTML_IMG_RE = re.compile(r'<img\s[^>]*?>', re.IGNORECASE)
HTML_SRC_RE = re.compile(r'src=["\']([^"\']+)["\']', re.IGNORECASE)
IMAGE_PLACEHOLDER = "[图]"

# 站点相对图片（如 /docfetch.php?id=...）补域名后作为外链透出
SITE_BASE = os.getenv("KB_SITE_BASE", "https://store.sim3d.com").rstrip("/")
# assets 索引根（`images/xxx.jpg` 这类 src 需按 basename 反查；默认 <项目根>/md-source）
ASSETS_ROOT = os.getenv("KB_ASSETS_ROOT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "md-source")


def strip_images(text: str) -> tuple[str, list[str]]:
    """剥离 markdown 图片链接与 HTML 图片标签：返回 (替换为 [图] 的文本, 原始路径/URL 列表)

    MD 链接占全库文本约 16.2%，HTML `<img>` 再占 1.66%——两者都是纯噪声 token，
    必须都出嵌入文本；路径原样收进 refs 由返回层解析（不参与嵌入）。
    """
    refs = [m.group(2) for m in IMAGE_RE.finditer(text)]
    cleaned = IMAGE_RE.sub(IMAGE_PLACEHOLDER, text)
    for tag in HTML_IMG_RE.findall(cleaned):
        m = HTML_SRC_RE.search(tag)
        if m:
            refs.append(m.group(1))
    cleaned = HTML_IMG_RE.sub(IMAGE_PLACEHOLDER, cleaned)
    return cleaned, refs


_ASSET_INDEX: dict[str, list[str]] | None = None


def _assets_lookup(ref: str, source_file: str) -> str | None:
    """`images/xxx.jpg` 这类相对 src 的反查：basename 唯一命中直接用；
    多个同名候选时，用「源文件名的 slug 变体出现在候选路径里」择优；仍不确定返回 None。"""
    global _ASSET_INDEX
    if _ASSET_INDEX is None:
        idx: dict[str, list[str]] = {}
        if os.path.isdir(ASSETS_ROOT):
            for dirpath, _, files in os.walk(ASSETS_ROOT):
                for f in files:
                    idx.setdefault(f.lower(), []).append(os.path.join(dirpath, f))
        _ASSET_INDEX = idx
    cands = _ASSET_INDEX.get(os.path.basename(ref).lower(), [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    stem = os.path.splitext(os.path.basename(source_file))[0].strip().lower()
    variants = {stem, stem.replace(' ', '_'), stem.replace(' ', '-'),
                stem.replace(' ', ''), stem.replace('_', ''), stem.replace('-', '')}
    for c in cands:
        low = c.lower().replace('\\', '/')
        if any(v and v in low for v in variants):
            return c
    return None


def resolve_image_paths(source_file: str, refs: list[str]) -> list[str]:
    """相对路径 → 绝对路径（返回层用，不让 Agent 手算 ../../）

    实测三类图片链接（全库 11,555 条）与对应处理：
    - **markdown 相对链接（8,444 条，必经 unquote）**：7,157 条含 `%20` 等编码，
      不解码仅 1,287 条可达、解码后 8,444 条全部可达；
    - **http(s) 外链（1,012 条）** 与 **站点相对（549 条 `/docfetch.php?...`）**：
      本地读不到，前者原样返回、后者补 `SITE_BASE` 域名后返回，由调用方标注为外链；
    - **HTML `images/xxx.jpg`（1,550 条）**：按 basename 反查 assets 索引，
      能唯一确定的给绝对路径，否则返回原拼路径（调用方会标为「路径不存在」，不会误导）。
    """
    import os.path as _p
    from urllib.parse import unquote
    base = _p.dirname(source_file)
    out = []
    for r in refs:
        low = r.lower()
        if low.startswith(('http://', 'https://')):
            out.append(r)
        elif r.startswith('/'):
            out.append(SITE_BASE + r)
        else:
            ap = _p.normpath(_p.join(base, unquote(r)))
            out.append(ap if _p.exists(ap) else (_assets_lookup(r, source_file) or ap))
    return out
