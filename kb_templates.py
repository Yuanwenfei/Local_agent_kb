#!/usr/bin/env python3
"""
P3 知识类型模板与入库支撑层（设计 §8.2 / §9.1–9.4）。

- 模板文件在 `templates/` 下，是 front-matter v2 头与固定小节的**唯一骨架来源**：
  kb_note / kb_ingest 用它渲染，保证 8 类知识都自带必填元数据与结构。
- 提供草稿落盘（kb_note）与规范化入库（kb_ingest）所需的路径、渲染、校验逻辑。
"""
import os
import re
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

# kb_note 草稿与 kb_ingest 产物的落盘根（可用 KB_INBOX_DIR 覆盖）
INBOX_DIR = os.getenv("KB_INBOX_DIR") or os.path.join(PROJECT_ROOT, "kb-inbox")
DRAFTS_DIR = os.path.join(INBOX_DIR, "drafts")

# kind → (模板文件, 中文名, 默认 trust, 默认 source_origin, 默认 ingested_by, 是否模型级)
KINDS = {
    "model_fact_card": ("model_fact_card.md", "模型事实卡", "measured", "plugin", "em3d_export", True),
    "part_signature": ("part_signature.md", "部件签名表", "measured", "plugin", "em3d_export", False),
    "version_matrix": ("version_matrix.md", "版本差异矩阵", "tutorial", "kb", "manual", False),
    "error_faq": ("error_faq.md", "错误串 FAQ", "measured", "plugin", "qlp_export", False),
    "defect_log": ("defect_log.md", "缺陷登记", "measured", "plugin", "em3d_export", False),
    "enum_examples": ("enum_examples.md", "枚举示例", "tutorial", "kb", "manual", False),
    "qlp_snippet": ("qlp_snippet.md", "QLP 可跑片段", "measured", "plugin", "qlp_export", False),
    "asset_pointer": ("asset_pointer.md", "资产路径登记", "tutorial", "decompile", "manual", False),
    "draft": ("draft.md", "草稿（kb_note）", "pending", "kb", "kb_note", False),
}

# 插件产物 → kind 映射（设计 §9.2；P3-8 契约）
PLUGIN_PRODUCTS = {
    "model_digest.md": "model_fact_card",
    "model_digest.yaml": "model_fact_card",
    "model_graph.json": "model_fact_card",
    "capabilities.json": "part_signature",
    "model_health.md": "defect_log",
    "crash_site.json": "error_faq",
    "qlp_trace.json": "error_faq",
}
QLP_SUFFIXES = (".qlp",)

# 模板里要求填写的哨兵：渲染后仍在，说明该小节没写内容
_UNFILLED_RE = re.compile(r'待填[:：]')
_HEADING_RE = re.compile(r'^##\s+(.+)$', re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r'\{\{(\w+)\}\}')

# 文件名非法字符（Windows）+ 空白
_ILLEGAL_RE = re.compile(r'[\\/:*?"<>|\s]+')


def kind_names() -> list[str]:
    return list(KINDS)


def kind_label(kind: str) -> str:
    return KINDS.get(kind, ("", kind, "", "", "", False))[1]


def is_model_level(kind: str) -> bool:
    return bool(KINDS.get(kind, ("", "", "", "", "", False))[5])


def template_path(kind: str) -> str:
    entry = KINDS.get(kind)
    return os.path.join(TEMPLATES_DIR, entry[0]) if entry else ""


def load_template(kind: str) -> str:
    """读取模板原文；kind 未知返回空串（调用方决定是否报错）"""
    path = template_path(kind)
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding='utf-8') as f:
        return f.read()


def slugify(text: str, max_len: int = 60) -> str:
    """生成安全的文件名片段：保留中英文数字，压缩空白与非法字符"""
    s = _ILLEGAL_RE.sub('-', (text or "").strip()).strip('-')
    s = re.sub(r'-{2,}', '-', s)
    return (s[:max_len] or "untitled").strip('-')


def _yaml_scalar(value) -> str:
    """转成 YAML 安全标量（一律加引号，内部引号转义）"""
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


def default_fields(kind: str, meta: dict | None = None, *, title: str = "") -> dict:
    """按 kind 的默认值补全模板占位符（D-4：插件产物 measured / 人工 tutorial / 待验 pending）"""
    entry = KINDS.get(kind)
    _, label, trust, origin, ingested_by, _ml = entry if entry else ("", kind, "tutorial", "kb", "manual", False)
    meta = dict(meta or {})
    # 允许 meta 直接给 applies_to 嵌套结构
    applies = meta.get("applies_to") if isinstance(meta.get("applies_to"), dict) else {}
    fields = {
        "title": meta.get("title") or title or f"{label}-{date.today().isoformat()}",
        "product": meta.get("product") or "emulate3d",
        "framework": (applies.get("framework") or meta.get("framework")
                      or meta.get("framework_version") or "unknown"),
        "model": applies.get("model") or meta.get("model") or meta.get("model_scope") or "general",
        "doc_version": meta.get("doc_version") or "",
        "doc_date": meta.get("doc_date") or date.today().isoformat(),
        "trust": meta.get("trust") or trust,
        "source_origin": meta.get("source_origin") or origin,
        "ingested_by": meta.get("ingested_by") or ingested_by,
        "revision": meta.get("revision") or "",
        # 正文类占位符（由调用方覆盖，未给则留空，渲染后即为"空小节"）
        "model_path": meta.get("model_path") or "",
        "symbol": meta.get("symbol") or "",
        "error_text": meta.get("error_text") or "",
        "example": meta.get("example") or "",
        "code": meta.get("code") or "",
        "host": meta.get("host") or "",
        "asset_path": meta.get("asset_path") or "",
        "fixed": meta.get("fixed") or "待确认",
        "topic": meta.get("topic") or "",
        "evidence": meta.get("evidence") or "",
    }
    # canonical / superseded_by：模板里写死的布尔行由 render 单独替换
    if "canonical" in meta and meta["canonical"] is not None:
        fields["canonical"] = meta["canonical"]
    else:
        fields["canonical"] = (kind != "draft")   # 入库即正本；草稿默认副本语义（不对外检索）
    fields["superseded_by"] = meta.get("superseded_by") or ""
    fields["canonical_uid"] = meta.get("canonical_uid") or ""
    # 调用方显式传入的其它键直接覆盖（如 error_text/code）
    fields.update({k: v for k, v in meta.items() if k in fields and v})
    return fields


def render(kind: str, meta: dict | None = None, body: str | None = None,
           *, title: str = "") -> str:
    """渲染模板。

    body=None → 用模板自带的骨架正文（带"待填"哨兵，便于人补写）；
    body 给出 → 保留模板的 front-matter v2 头，正文替换为给定内容。
    """
    tpl = load_template(kind)
    if not tpl:
        raise ValueError(f"未知知识类型 {kind!r}（可选：{', '.join(kind_names())}）")
    fields = default_fields(kind, meta, title=title)

    # supersedes 列表特殊处理（模板里写的是 `supersedes: []`）
    supersedes = (meta or {}).get("supersedes") or []
    if isinstance(supersedes, str):
        supersedes = [supersedes]
    if supersedes:
        block = "supersedes:\n" + "\n".join(f'  - {_yaml_scalar(s)}' for s in supersedes)
        tpl = tpl.replace("supersedes: []", block)

    # 模板里 canonical 是写死的布尔行：副本（P3-6）必须能翻成 false，
    # 并可就地追加 superseded_by / canonical_uid 指向正本
    canon = bool(fields.get("canonical"))
    extra_lines = []
    if fields.get("superseded_by"):
        extra_lines.append("superseded_by: " + _yaml_scalar(fields["superseded_by"]))
    if fields.get("canonical_uid"):
        extra_lines.append("canonical_uid: " + _yaml_scalar(fields["canonical_uid"]))
    repl = "canonical: " + ("true" if canon else "false")
    if extra_lines:
        repl += "\n" + "\n".join(extra_lines)
    base_line = "canonical: true" if "canonical: true" in tpl else "canonical: false"
    tpl = tpl.replace(base_line, repl, 1)

    if body is not None:
        tpl = _split_front_matter(tpl)[0] + body.strip() + "\n"

    def _sub(m):
        key = m.group(1)
        val = fields.get(key, "")
        return _yaml_scalar(val) if _in_front_matter(tpl, m.start()) else str(val)

    return _PLACEHOLDER_RE.sub(_sub, tpl)


_FM_SPLIT_RE = re.compile(r'^(---\s*\n.*?\n---\s*\n)', re.DOTALL)


def _split_front_matter(text: str) -> tuple[str, str]:
    """返回 (front-matter 块（含定界符）, 正文)"""
    m = _FM_SPLIT_RE.match(text)
    if not m:
        return "", text
    return m.group(1), text[m.end():]


def _in_front_matter(text: str, pos: int) -> bool:
    fm, _ = _split_front_matter(text)
    return pos < len(fm)


# ==================== 落盘 ====================

def target_path(kind: str, title: str) -> str:
    """入库目标路径：kb-inbox/<kind>/<slug>.md"""
    folder = DRAFTS_DIR if kind == "draft" else os.path.join(INBOX_DIR, kind)
    return os.path.join(folder, slugify(title) + ".md")


def write_doc(kind: str, meta: dict | None = None, body: str | None = None,
              *, title: str = "", path: str | None = None) -> str:
    """渲染并写盘；返回写入路径。目录不存在时自动创建。"""
    text = render(kind, meta, body, title=title)
    out = path or target_path(kind, title or (meta or {}).get("title") or "")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return out


# ==================== 入库校验（设计 §9.4，本地可判定的部分） ====================

MIN_CHUNK_CHARS = 20  # 沿用解析器阈值
_MOJIBAKE_RE = re.compile(r'[\ufffd]|Ã[\x80-\xbf]|â€')


def validate_text(text: str, kind: str, meta: dict | None = None) -> list[dict]:
    """返回校验项列表：{level: error|warn|info, rule, detail}。

    仅做不看库就能判的检查（缺必填元数据、空/过短、编码乱码、未填小节）；
    重复 hash 与版本冲突需要查库，由 kb_ingest 处理器补充（见 _handle_kb_ingest）。
    """
    from kb_schema import lint_meta, parse_meta_yaml

    issues: list[dict] = []

    if kind not in KINDS:
        issues.append({"level": "error", "rule": "kind",
                       "detail": f"未知类型 {kind!r}，可选：{', '.join(kind_names())}"})

    fm, body = _split_front_matter(text)
    if not fm:
        issues.append({"level": "error", "rule": "缺 front-matter",
                       "detail": "入库文本必须带 --- 包裹的元数据头（用 kb_note/kb_ingest 生成）"})
        meta_eff = dict(meta or {})
    else:
        # 以**渲染后的 front-matter 为准**：它才是真正会进 payload 的那份元数据
        meta_eff = parse_meta_yaml(fm.strip().strip('-').strip())
        for k, v in (meta or {}).items():
            if not meta_eff.get(k):
                meta_eff[k] = v

    missing = lint_meta(meta_eff, is_model_level=is_model_level(kind))
    if missing:
        issues.append({"level": "error", "rule": "缺必填元数据",
                       "detail": "、".join(missing)})
    if is_model_level(kind) and not meta_eff.get("revision"):
        issues.append({"level": "warn", "rule": "缺 revision",
                       "detail": "模型级结论应带 revision，否则无法判断是否「待刷新」"})

    stripped = _PLACEHOLDER_RE.sub('', body).strip()
    if not stripped:
        issues.append({"level": "error", "rule": "无正文", "detail": "正文为空"})
    elif len(stripped) < MIN_CHUNK_CHARS:
        issues.append({"level": "error", "rule": "正文过短",
                       "detail": f"{len(stripped)} 字符 < {MIN_CHUNK_CHARS}，切片会被解析器丢弃"})

    unfilled = _UNFILLED_RE.findall(body)
    if unfilled:
        issues.append({"level": "warn", "rule": "未填小节",
                       "detail": f"{len(unfilled)} 处「待填」哨兵仍在（可入库，但检索价值低）"})

    # 模板小节是否齐（info：草稿转正时正文常还是旧模板的结构，提示补写）
    tpl_headings = _HEADING_RE.findall(load_template(kind)) if kind in KINDS else []
    if tpl_headings and stripped:
        have = {h.strip() for h in _HEADING_RE.findall(body)}
        lack = [h for h in tpl_headings if h.strip() not in have]
        if lack:
            issues.append({"level": "info", "rule": "模板缺节",
                           "detail": "该类型模板要求的小节未出现在正文：" + "、".join(lack[:6])})

    # 只查正文，避免 front-matter 里的中文被误判
    if _MOJIBAKE_RE.search(body):
        issues.append({"level": "error", "rule": "编码/乱码",
                       "detail": "检测到替换字符或典型乱码序列（疑似 GBK/UTF-8 误读）"})

    return issues


def summarize(issues: list[dict]) -> str:
    """校验项 → 单行摘要"""
    if not issues:
        return "✅ 校验通过"
    errs = sum(1 for i in issues if i["level"] == "error")
    warns = sum(1 for i in issues if i["level"] == "warn")
    return f"{'❌' if errs else '⚠'} {errs} 项错误 / {warns} 项警告"
