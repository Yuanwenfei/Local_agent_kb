#!/usr/bin/env python3
"""
PDF / Word(DOCX) 文档解析器
- parse_pdf(file_path)  -> list[chunk]
- parse_docx(file_path) -> list[chunk]
- parse_document(file_path) -> 按扩展名分发

辅助:
- normalize_text         统一文本规范化
- detect_doc_type        文档类型识别（路径 + 内容 fallback）
- detect_product         产品识别（路径 + 内容 fallback）
- detect_header_footer   PDF 跨页重复检测（去页眉/页脚）
"""
import os
import sys
import re
import hashlib
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

# 元数据规范 v2 共享层（front-matter/sidecar/图片剥离，见 kb_schema.py）
# 嵌入式 Python（python312._pth）不会把脚本目录放进 sys.path，同目录模块必须自行引导，
# 否则被别的入口点 import 时同样会 ModuleNotFoundError。
_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _LOCAL_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_DIR)

from kb_schema import (
    parse_meta_yaml, meta_to_payload, strip_images, load_sidecar_meta,
    normalize_for_search,
)

# 重依赖延迟导入
def _import_fitz():
    import fitz  # PyMuPDF
    return fitz

def _import_docx():
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn
    return docx, Table, Paragraph, qn


# ==================== 文本规范化 ====================
def normalize_text(text: str) -> str:
    """
    统一规范化：连字符断词合并、NFKC、行内空白压缩（保留行首缩进）、空行压缩

    行首缩进会被保留，避免破坏代码块结构；行中间多余的空格/Tab 仍会被
    压缩为单空格，行尾空白会被去除。
    """
    if not text:
        return ''

    # 1. 连字符断词合并：con-\nfiguration -> configuration
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # 2. NFKC 归一化（全角 ASCII -> 半角；兼容字符替换）
    text = unicodedata.normalize('NFKC', text)

    # 3. 逐行处理：保留行首缩进、压缩行中间多余空白、去除行尾空白
    cleaned_lines = []
    for line in text.split('\n'):
        m = re.match(r'^([ \t]*)', line)
        leading = m.group(1) if m else ''
        rest = line[len(leading):]
        rest = re.sub(r'[ \t]+', ' ', rest).rstrip()
        cleaned_lines.append(leading + rest)
    text = '\n'.join(cleaned_lines)

    # 4. 3+ 空行 -> 2 空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 5. 去除开头/结尾的空行，但不剥离首行的缩进
    text = text.lstrip('\n').rstrip()
    return text


# ==================== 文档类型 / 产品（路径 + 内容 fallback） ====================
def detect_doc_type(file_path: str, text_preview: str = "") -> str:
    fp = file_path.lower()
    if "api" in fp or "reference" in fp:
        return "api_reference"
    if "tutorial" in fp or "guide" in fp or "howto" in fp:
        return "tutorial"
    if "design" in fp or "spec" in fp:
        return "design_spec"
    if "release" in fp or "changelog" in fp:
        return "release_note"

    if text_preview:
        tp = text_preview[:300].lower()
        if any(k in tp for k in ("api reference", "api 参考", "namespace ", "class ")):
            return "api_reference"
        if any(k in tp for k in ("tutorial", "教程", "step by step", "快速入门")):
            return "tutorial"
        if any(k in tp for k in ("design specification", "设计规范", "design spec")):
            return "design_spec"
        if any(k in tp for k in ("release note", "更新日志", "changelog")):
            return "release_note"

    return "user_manual"


def detect_product(file_path: str, text_preview: str = "") -> str:
    fp = file_path.lower()
    if "emulate" in fp or "e3d" in fp:
        return "emulate3d"
    if "wms" in fp:
        return "wms"
    if "wcs" in fp:
        return "wcs"
    if "agv" in fp or "amr" in fp:
        return "agv"
    if "asrs" in fp:
        return "asrs"

    if text_preview:
        tp = text_preview[:300].lower()
        if "emulate3d" in tp or "emulate 3d" in tp or "demo3d" in tp:
            return "emulate3d"
        if "wms" in tp:
            return "wms"
        if "wcs" in tp:
            return "wcs"
        if "agv" in tp or "amr" in tp:
            return "agv"
        if "asrs" in tp:
            return "asrs"

    return "general"


# ==================== 通用工具 ====================
_CLASS_RE = re.compile(r'class\s+(\w+)')
_METHOD_RE = re.compile(
    r'(?:public|private|protected|static|internal|virtual|override|async)?\s*'
    r'(?:[\w<>,\s]+)\s+(\w+)\s*\('
)


def _detect_class_method(text: str) -> tuple[str, str]:
    cm = _CLASS_RE.search(text)
    mm = _METHOD_RE.search(text)
    return (cm.group(1) if cm else ""), (mm.group(1) if mm else "")


def _file_content_hash(file_path: str) -> str:
    """计算文件内容的 MD5（前 8 位），与 index_state.json 保持一致"""
    h = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:8]


def _safe_mtime(file_path: str) -> float:
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return None


# ==================== PDF：页眉/页脚检测 ====================
def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    if a == b:
        return True
    if abs(len(a) - len(b)) / max(len(a), len(b)) > 0.5:
        return False
    sm = SequenceMatcher(None, a, b)
    # real_quick_ratio / quick_ratio 均为 ratio() 的数学上界（O(n+m)），
    # 上界已低于阈值时可安全提前否决，判定结果与直接调用 ratio() 完全一致。
    # detect_header_footer 是 O(页码^2) 次比对，此提前否决可大幅缩短耗时。
    if sm.real_quick_ratio() < threshold:
        return False
    if sm.quick_ratio() < threshold:
        return False
    return sm.ratio() >= threshold


def detect_header_footer(
    pages: list[str],
    head_lines: int = 5,
    foot_lines: int = 3,
    page_ratio: float = 0.7,
    sim_threshold: float = 0.8,
) -> list[str]:
    """
    跨页重复检测，去除页眉/页脚。
    - 顶部最多 head_lines 行、底部最多 foot_lines 行作为候选区
    - 在 >= page_ratio 比例的页中，同位置出现相似度 >= sim_threshold 的行 -> 视为页眉/页脚
    - 纯数字行（页码）固定位置 -> 直接移除
    """
    n_pages = len(pages)
    if n_pages < 3:
        return pages

    threshold_count = max(2, int(n_pages * page_ratio))
    page_lines = [p.splitlines() for p in pages]

    head_remove = [set() for _ in range(n_pages)]
    foot_remove = [set() for _ in range(n_pages)]

    # 顶部 head_lines 行
    for pos in range(head_lines):
        for p_idx in range(n_pages):
            if pos >= len(page_lines[p_idx]):
                continue
            line = page_lines[p_idx][pos].strip()
            if not line or line.startswith('[图片占位符'):
                continue
            if re.fullmatch(r'\d+', line):
                head_remove[p_idx].add(pos)
                continue
            cnt = 1
            for q_idx in range(n_pages):
                if q_idx == p_idx:
                    continue
                if pos < len(page_lines[q_idx]):
                    other = page_lines[q_idx][pos].strip()
                    if _similar(line, other, sim_threshold):
                        cnt += 1
            if cnt >= threshold_count:
                head_remove[p_idx].add(pos)

    # 底部 foot_lines 行（负偏移）
    for neg in range(1, foot_lines + 1):
        for p_idx in range(n_pages):
            n = len(page_lines[p_idx])
            if neg > n:
                continue
            pos = n - neg
            line = page_lines[p_idx][pos].strip()
            if not line or line.startswith('[图片占位符'):
                continue
            if re.fullmatch(r'\d+', line):
                foot_remove[p_idx].add(pos)
                continue
            cnt = 1
            for q_idx in range(n_pages):
                if q_idx == p_idx:
                    continue
                qlen = len(page_lines[q_idx])
                if neg > qlen:
                    continue
                q_pos = qlen - neg
                other = page_lines[q_idx][q_pos].strip()
                if _similar(line, other, sim_threshold):
                    cnt += 1
            if cnt >= threshold_count:
                foot_remove[p_idx].add(pos)

    cleaned = []
    total_removed = 0
    for p_idx, lines in enumerate(page_lines):
        remove = head_remove[p_idx] | foot_remove[p_idx]
        total_removed += len(remove)
        new_lines = [ln for i, ln in enumerate(lines) if i not in remove]
        cleaned.append('\n'.join(new_lines))

    if total_removed > 0:
        removed_lines = []
        for p_idx in range(n_pages):
            for pos in sorted(head_remove[p_idx] | foot_remove[p_idx]):
                if pos < len(page_lines[p_idx]):
                    removed_lines.append(f"  P{p_idx + 1}L{pos + 1}: {page_lines[p_idx][pos].strip()[:80]}")
        if removed_lines:
            print(f"[KB-PDF] 检测到页眉/页脚，移除 {total_removed} 行:", file=sys.stderr)
            for rl in removed_lines[:20]:  # 最多展示 20 行
                print(rl, file=sys.stderr)
            if len(removed_lines) > 20:
                print(f"  ... 等共 {len(removed_lines)} 行", file=sys.stderr)

    return cleaned


# ==================== PDF：表格转 Markdown ====================
def _rows_to_markdown(rows: list[list]) -> str:
    if not rows:
        return ''
    max_cols = max((len(r) for r in rows), default=0)
    if max_cols == 0:
        return ''
    md_rows = []
    for row in rows:
        cells = []
        for i in range(max_cols):
            c = row[i] if i < len(row) else ''
            c_str = ('' if c is None else str(c)).strip().replace('\n', ' ').replace('|', '\\|')
            cells.append(c_str)
        if any(cells):
            md_rows.append('| ' + ' | '.join(cells) + ' |')
    if not md_rows:
        return ''
    sep = '| ' + ' | '.join(['---'] * max_cols) + ' |'
    if len(md_rows) > 1:
        md_rows.insert(1, sep)
    else:
        md_rows.append(sep)
    return '\n'.join(md_rows)


def _rect_overlaps(rect, block, threshold: float = 0.5) -> bool:
    """block 在 rect 内的面积比例 >= threshold"""
    bx0, by0, bx1, by1 = block[0], block[1], block[2], block[3]
    rx0, ry0, rx1, ry1 = rect[0], rect[1], rect[2], rect[3]
    ox = max(0.0, min(bx1, rx1) - max(bx0, rx0))
    oy = max(0.0, min(by1, ry1) - max(by0, ry0))
    area = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return (ox * oy) / area >= threshold


def _extract_pdf_page(page) -> tuple[str, bool]:
    """提取单页文本（阅读顺序），表格 -> Markdown；返回 (text, has_image)"""
    page_idx = page.number  # 0-based
    has_image = len(page.get_images()) > 0

    # 表格检测
    table_items = []  # [(y_top, markdown, bbox)]
    try:
        finder = page.find_tables()
        for tab in finder.tables:
            try:
                rows = tab.extract()
            except Exception:
                continue
            md = _rows_to_markdown(rows)
            if md.strip():
                bbox = tuple(tab.bbox)
                table_items.append((bbox[1], md, bbox))
    except Exception:
        pass

    table_rects = [bbox for _, _, bbox in table_items]

    # 文本块（排除表格区域）
    blocks = page.get_text("blocks")
    items = []  # [(y_top, text)]
    for b in blocks:
        if len(b) < 7 or b[6] != 0:
            continue
        txt = (b[4] or '').strip()
        if not txt:
            continue
        if any(_rect_overlaps(tr, b) for tr in table_rects):
            continue
        items.append((b[1], txt))

    for y, md, _ in table_items:
        items.append((y, md))

    items.sort(key=lambda x: x[0])

    page_text = '\n\n'.join(it[1] for it in items if it[1])

    return page_text, has_image


# ==================== 切块公共逻辑（各格式共用） ====================
# 目标块大小：超过则按自然边界二次切分。
# 旧实现是 "text[:1500]" 硬截断，实测让 33% 的正文既不进向量也读不到。
CHUNK_MAX_CHARS = int(os.getenv("KB_CHUNK_MAX_CHARS", "1200"))

# 标题识别：兼容站点实际使用的 "##Title##"（# 后无空格）与标准 "## Title ##"
_HEAD_RE = re.compile(r'^(#{1,6})[ \t]*(.+?)[ \t]*#*[ \t]*$')
_HEAD_MAX_LEN = 120

# Hugo 短代码 {{% notice warning %}} / {{% /notice %}} / {{% children %}}
_SHORTCODE_RE = re.compile(r'\{\{%[^%]*%\}\}')
# YAML front-matter
_FRONT_MATTER_RE = re.compile(r'\A---\r?\n(.*?)\r?\n---\r?\n', re.S)


def match_heading(line: str):
    """识别标题行，返回 (level, 标题文本)；不是标题返回 None。

    容忍 "##Title##" 这种 # 后无空格的写法——站点两套文档都在用，
    旧的 `^(#{1,4})\\s+(.+)` 完全匹配不到，会把整篇当正文。
    """
    m = _HEAD_RE.match(line)
    if not m:
        return None
    hashes, text = m.group(1), m.group(2).strip()
    if not text or len(text) > _HEAD_MAX_LEN:
        return None
    if len(hashes) == 1 and len(line) > 1 and line[1] not in ' \t':
        return None  # "#hashtag" 之类，不当作标题
    return len(hashes), text


def parse_front_matter(text: str):
    """拆出 YAML front-matter，返回 (metadata dict, 剩余正文)。

    v2（G-A）：改用 PyYAML 解析，支持嵌套结构（applies_to.framework 等）；
    解析失败退回朴素行解析，保证健壮性。
    """
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    return parse_meta_yaml(m.group(1)), text[m.end():]


def is_toc_text(text: str) -> bool:
    """目录页判定：非空行中 >40% 是 markdown 链接列表项（- [x](y) 或含 ]( ）。"""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    linkish = sum(1 for ln in lines if ln.lstrip().startswith(('- [', '* [')) or '](' in ln)
    return linkish / len(lines) > 0.4


# 显式图注标记（P4-3③ 人工补真图注的规范写法）：`图注：xxx` / `*Caption: xxx*` / `<!-- caption: xxx -->`
_EXPLICIT_CAPTION_RE = re.compile(
    r'^\s*(?:[-*>]\s*)?(?:图\s*注|注\s*释|注|caption|fig(?:ure)?)\s*[:：]\s*(.+?)\s*\**\s*$',
    re.IGNORECASE)
_HTML_CAPTION_RE = re.compile(r'<!--\s*caption\s*[:：]\s*(.*?)\s*-->', re.IGNORECASE)
# 这些小节标题不含信息量，单独拿来当图注等于没有
_GENERIC_HEADS = {"", "概述", "概览", "overview", "introduction", "简介", "前言"}


def _image_caption(heading: str, raw_lines: list[str], *, title: str = "",
                   heading_path: str = "", refs: list[str] = None) -> tuple[str, str]:
    """图注抽取向导（P4-3②）：返回 (caption, source)。

    优先级（前两类是**真图注**，后三类是**弱图注**兜底）：
      1. `explicit` 显式标记——图片上/下 3 行内的 `图注：xxx` 或 `<!-- caption: xxx -->`；
      2. `alt` 作者写的 alt（质量最高的存量来源，全库 387 条）；
      3. `context` 图片上方紧邻的一句正文（≥8 字）；
      4. `heading` 文档标题 + 章节路径（跳过「概述」这类无信息量标题）；
      5. `filename` 图片文件名（去扩展名/分隔符）。
    全都不成立时返回 ("", "")，交由 kb_caption 人工补齐。
    """
    # 1) 显式图注：人工/规范写法优先，且优先看图片**下方**（约定位置）
    img_idx = next((i for i, ln in enumerate(raw_lines) if '![' in ln or '<img' in ln), None)
    if img_idx is not None:
        for ln in raw_lines[img_idx:img_idx + 4]:
            m = _EXPLICIT_CAPTION_RE.match(ln) or _HTML_CAPTION_RE.search(ln)
            if m and m.group(1).strip():
                return m.group(1).strip()[:200], "explicit"

    # 2) alt（作者亲手写的，质量最高）
    alts = [a.strip() for a in re.findall(r'!\[([^\]]*)\]\(', "\n".join(raw_lines)) if a.strip()]
    if alts:
        return "；".join(alts[:3])[:200], "alt"

    # 3) 图片上方紧邻的一句正文
    if img_idx is not None:
        for ln in reversed(raw_lines[:img_idx]):
            s = ln.strip().strip('#').strip().strip('*').strip()
            if s and '![' not in s and '<img' not in s and len(s) >= 8:
                cap = f"{heading}: {s}" if heading and heading not in _GENERIC_HEADS else s
                return cap[:200], "context"

    # 4) 文档标题 + 章节路径（图片名与标题是最稳的结构化信号）
    parts = [p.strip() for p in (heading_path or heading or "").split(">")
             if p.strip() and p.strip().lower() not in _GENERIC_HEADS]
    base = " > ".join(parts[-2:])
    if title and base and not base.startswith(title):
        base = f"{title} · {base}"
    elif not base:
        base = (title or "").strip()

    # 5) 图片文件名兜底（`Conveyor Indicator Arrows.png` 这类名字本身就是语义）
    names = []
    for r in (refs or [])[:2]:
        # 图片路径常带 %20 等转义（md 里空格被编码），先还原再取语义
        from urllib.parse import unquote
        n = os.path.splitext(os.path.basename(unquote(str(r))))[0]
        n = re.sub(r'[_-]+', ' ', n).strip()
        if n and not re.fullmatch(r'(?:image|img|photo|pic|screenshot|untitled)?\s*\d*', n, re.I):
            names.append(n)
    if names:
        tag = "（图：" + "；".join(names) + "）"
        return ((base + tag) if base else "图：" + "；".join(names))[:200], "filename"
    if base:
        return base[:200], "heading"
    return "", ""


def _weak_image_caption(heading: str, raw_lines: list[str]) -> str:
    """兼容旧签名（只取图注文字，不关心来源）。"""
    return _image_caption(heading, raw_lines)[0]


def strip_shortcodes(text: str) -> str:
    """去掉 Hugo 短代码标记（保留其中的正文）。"""
    return _SHORTCODE_RE.sub('', text)


def split_long_text(text: str, max_chars: int = None) -> list:
    """超长文本按空行 / 行边界二次切分，不在代码块内部断开。"""
    max_chars = max_chars or CHUNK_MAX_CHARS
    if len(text) <= max_chars:
        return [text]

    blocks, buf, in_code = [], [], False
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            in_code = not in_code
        if not in_code and not line.strip() and buf and sum(len(b) + 1 for b in buf) > max_chars * 0.5:
            blocks.append('\n'.join(buf))
            buf = []
            continue
        buf.append(line)
    if buf:
        blocks.append('\n'.join(buf))

    out, cur, cur_len = [], [], 0
    for b in blocks:
        if len(b) > max_chars:                     # 单块仍超长 -> 按行累加
            for line in b.split('\n'):
                if len(line) > max_chars:          # 单行超长（无换行长行）-> 硬切
                    # 必须切：e5 最大 512 token（约 2000 字符），超出的部分
                    # 不会被编码，等于静默丢失
                    if cur:
                        out.append('\n'.join(cur))
                        cur, cur_len = [], 0
                    out.extend(line[k:k + max_chars]
                               for k in range(0, len(line), max_chars))
                    continue
                if cur and cur_len + len(line) + 1 > max_chars:
                    out.append('\n'.join(cur))
                    cur, cur_len = [], 0
                cur.append(line)
                cur_len += len(line) + 1
            continue
        if cur and cur_len + len(b) + 2 > max_chars:
            out.append('\n'.join(cur))
            cur, cur_len = [], 0
        cur.append(b)
        cur_len += len(b) + 2
    if cur:
        out.append('\n'.join(cur))

    pieces = [c.strip() for c in out if c.strip()]
    return pieces or [text[:max_chars]]


def finalize_chunks(text: str, *, heading: str, source: str, title: str,
                    doc_type: str, product: str, file_hash: str,
                    has_image: bool = False, chunk_index: int = 0,
                    extra: dict = None) -> list[dict]:
    """统一出口：拼嵌入文本 + 超长二次切分 + 补齐 payload。

    - 嵌入文本前置 "标题 — 章节"，让标题类查询能直接命中（实测 MRR 0.627 -> 0.842）
    - 超长内容拆成多个 chunk（chunk_part 标记序号），不再硬截断丢正文
    """
    text = strip_shortcodes(text or '').strip()
    if not text:
        return []

    prefix = f"{title} — {heading}" if (title and heading) else (title or '')
    chunks = []
    for part, piece in enumerate(split_long_text(text)):
        class_name, method_name = _detect_class_method(piece)
        chunk = {
            "text": f"{prefix}\n{piece}" if prefix else piece,
            "heading": heading,
            "source": source,
            "title": title,
            "doc_type": doc_type,
            "product": product,
            "chunk_index": chunk_index,
            "chunk_part": part,
            "has_image": has_image,
            "class_name": class_name,
            "method_name": method_name,
            "file_hash": file_hash,
            # 精确串检索通道底座（G-F）：与嵌入文本同源的规范化正文
            "search_text": normalize_for_search(piece),
        }
        if extra:
            chunk.update(extra)
        chunks.append(chunk)
    return chunks


def parse_pdf(file_path: str, meta: dict = None) -> list[dict]:
    if meta is None:
        meta = load_sidecar_meta(file_path)  # G-B：默认自动加载 sidecar
    fitz = _import_fitz()

    pages_text: list[str] = []
    pages_has_image: list[bool] = []

    with fitz.open(file_path) as doc:
        for page in doc:
            text, has_image = _extract_pdf_page(page)
            pages_text.append(text)
            pages_has_image.append(has_image)

    if not pages_text:
        return []

    # 页眉/页脚过滤
    pages_text = detect_header_footer(pages_text)
    # 文本规范化
    pages_text = [normalize_text(p) for p in pages_text]
    # 图片占位符（过滤后追加，避免被误判为页眉/页脚）
    pages_text = [
        (t + f'\n\n[图片占位符: 第{i + 1}页]').strip() if pages_has_image[i] else t
        for i, t in enumerate(pages_text)
    ]

    # 元数据
    preview = '\n'.join(pages_text[:3])[:600]
    doc_type = detect_doc_type(file_path, preview)
    product = detect_product(file_path, preview)
    title = Path(file_path).stem
    file_hash = _file_content_hash(file_path)

    chunks: list[dict] = []
    for i, (text, has_image) in enumerate(zip(pages_text, pages_has_image)):
        text = text.strip()
        if len(text) < 20:
            continue
        chunks.extend(finalize_chunks(
            text,
            heading=f"第{i + 1}页",
            source=file_path,
            title=title,
            doc_type=doc_type,
            product=product,
            file_hash=file_hash,
            has_image=has_image,
            chunk_index=i,
            extra={"page_number": i + 1,
                   **meta_to_payload(meta or {}, mtime=_safe_mtime(file_path))},
        ))
    return chunks


# ==================== Word (DOCX) 解析 ====================
def _iter_block_items(doc):
    """按文档顺序产出 ('paragraph', Paragraph) / ('table', Table)"""
    _, Table, Paragraph, qn = _import_docx()
    parent_elm = doc.element.body
    for child in parent_elm.iterchildren():
        tag = child.tag
        if tag == qn('w:p'):
            yield 'paragraph', Paragraph(child, doc)
        elif tag == qn('w:tbl'):
            yield 'table', Table(child, doc)


def _docx_paragraph_has_image(para) -> bool:
    _, _, _, qn = _import_docx()
    try:
        for _ in para._p.iter(qn('w:drawing')):
            return True
        for _ in para._p.iter(qn('w:pict')):
            return True
    except Exception:
        pass
    return False


def _docx_table_to_markdown(table) -> str:
    rows_data = []
    n_cols = 0
    for row in table.rows:
        cells = []
        for cell in row.cells:
            t = cell.text.strip().replace('\n', ' ').replace('|', '\\|')
            cells.append(t)
        n_cols = max(n_cols, len(cells))
        rows_data.append(cells)
    if not rows_data or n_cols == 0:
        return ''
    md_rows = []
    for cells in rows_data:
        if len(cells) < n_cols:
            cells = cells + [''] * (n_cols - len(cells))
        md_rows.append('| ' + ' | '.join(cells) + ' |')
    sep = '| ' + ' | '.join(['---'] * n_cols) + ' |'
    if len(md_rows) > 1:
        md_rows.insert(1, sep)
    else:
        md_rows.append(sep)
    return '\n'.join(md_rows)


def parse_docx(file_path: str, meta: dict = None) -> list[dict]:
    if meta is None:
        meta = load_sidecar_meta(file_path)  # G-B
    docx_mod, _, _, _ = _import_docx()
    doc = docx_mod.Document(file_path)

    chunks_raw = []
    current = {"heading": "概述", "level": 0, "lines": [], "has_image": False}

    for kind, item in _iter_block_items(doc):
        if kind == 'paragraph':
            style_name = ''
            try:
                style_name = item.style.name if item.style else ''
            except Exception:
                pass
            heading_match = re.match(r'Heading\s+(\d+)', style_name or '')
            if heading_match:
                level = int(heading_match.group(1))
                if level <= 3:
                    if current["lines"]:
                        chunks_raw.append(current)
                    head_text = item.text.strip() or f"Heading {level}"
                    current = {
                        "heading": head_text,
                        "level": level,
                        "lines": ['#' * level + ' ' + head_text],
                        "has_image": False,
                    }
                    if _docx_paragraph_has_image(item):
                        current["has_image"] = True
                    continue

            text = item.text
            if text and text.strip():
                current["lines"].append(text)
            if _docx_paragraph_has_image(item):
                current["has_image"] = True
                current["lines"].append("[图片占位符]")
        elif kind == 'table':
            md = _docx_table_to_markdown(item)
            if md:
                current["lines"].append('')
                current["lines"].append(md)
                current["lines"].append('')

    if current["lines"]:
        chunks_raw.append(current)

    if not chunks_raw:
        return []

    # 元数据 fallback 用的预览
    preview = ''
    for c in chunks_raw:
        preview = '\n'.join(c["lines"])[:600]
        if preview.strip():
            break

    title = Path(file_path).stem
    doc_type = detect_doc_type(file_path, preview)
    product = detect_product(file_path, preview)
    file_hash = _file_content_hash(file_path)

    chunks: list[dict] = []
    for i, c in enumerate(chunks_raw):
        text = normalize_text('\n'.join(c["lines"])).strip()
        if len(text) < 20:
            continue
        chunks.extend(finalize_chunks(
            text,
            heading=c["heading"],
            source=file_path,
            title=title,
            doc_type=doc_type,
            product=product,
            file_hash=file_hash,
            has_image=c["has_image"],
            chunk_index=i,
            extra=meta_to_payload(meta or {}, mtime=_safe_mtime(file_path)),
        ))
    return chunks


# ==================== 纯文本 (TXT) 解析 ====================
def parse_txt(file_path: str, meta: dict = None) -> list[dict]:
    """
    解析 .txt：保留缩进（含代码块），按空行段落切分，单块上限 1500 字符。

    元数据：路径优先，回落到首 600 字符内容启发式。
    """
    if meta is None:
        meta = load_sidecar_meta(file_path)  # G-B
    # 容错读取（少数日志可能带非 UTF-8 字节）
    raw = Path(file_path).read_text(encoding='utf-8', errors='replace')
    text = normalize_text(raw)
    if not text:
        return []

    # 段落切分：2+ 空行（超长块由 finalize_chunks 二次切分）
    blocks = re.split(r'\n{2,}', text)

    preview = text[:600]
    doc_type = detect_doc_type(file_path, preview)
    product = detect_product(file_path, preview)
    title = Path(file_path).stem
    file_hash = _file_content_hash(file_path)

    chunks: list[dict] = []
    for i, content in enumerate(blocks):
        content = content.strip()
        if len(content) < 20:
            continue
        # heading：首个非空行的前 60 字符
        heading = ''
        for ln in content.splitlines():
            if ln.strip():
                heading = ln.strip()[:60]
                break
        heading = heading or f"段{i + 1}"
        chunks.extend(finalize_chunks(
            content,
            heading=heading,
            source=file_path,
            title=title,
            doc_type=doc_type,
            product=product,
            file_hash=file_hash,
            chunk_index=i,
            extra=meta_to_payload(meta or {}, mtime=_safe_mtime(file_path)),
        ))
    return chunks


# ==================== Markdown (MD) 解析 ====================
def parse_md(file_path: str, meta: dict = None) -> list[dict]:
    """
    解析 .md：按 H1-H6 标题层级切分，代码块保护，短片段合并。

    v2 增强（设计 §5/§8.4 + 可行性报告 G-A/G-H）：
    - front-matter 走 PyYAML 嵌套解析；元数据按 §5.2 展平进 payload
    - 记录 heading_path（章节全路径）与 is_toc（目录页降权用）
    - 图片链接剥离出嵌入文本（替换为 [图] 占位），原始路径收进 image_refs；
      has_image 对 MD 正确置位（修复旧实现恒 False 的缺陷）；
    - 图注（P4-3②/设计 §8.4）：`_image_caption` 抽取向导 + 来源标记
      （explicit/alt/context/heading/filename），并以 `[图注] …` 行**参与嵌入**，
      使「图注文字可命中」（T9）成立
    """
    text = Path(file_path).read_text(encoding='utf-8', errors='replace')
    if not text.strip():
        return []

    front_matter, body = parse_front_matter(text)
    lines = body.splitlines()
    chunks_raw = []
    current = {"heading": "概述", "level": 0, "lines": [], "heading_path": ""}
    in_code_block = False
    head_stack: list[str] = []  # 章节层级栈，用于 heading_path

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        heading_match = None if in_code_block else match_heading(line)
        if heading_match:
            if current["lines"]:
                chunks_raw.append(current)
            level, head_text = heading_match
            # 标题本身可能带图（如 `###Line ![Line Entity](...png)`）——若不剥离，
            # 链接会经 heading/heading_path 拼回 text 前缀，正文剥离就白做了
            # （实测全库 12 处残留全出自此路径，见 tools/_diag_images.py）
            head_text = strip_images(head_text)[0].strip()
            head_stack = head_stack[:level - 1] + [head_text]
            current = {
                "heading": head_text,
                "level": level,
                "lines": [line],
                "heading_path": " > ".join(head_stack),
            }
        else:
            current["lines"].append(line)

    if current["lines"]:
        chunks_raw.append(current)

    # 合并短片段：标题块过短就并入前一块（阈值从 50 提到 120，
    # 减少"只有一行标题"的空块；首个块没有前驱，保留）
    merged = []
    for c in chunks_raw:
        content = "\n".join(c["lines"]).strip()
        if len(content) < 120 and c["level"] > 0 and merged:
            merged[-1]["lines"].extend(c["lines"])
        elif len(content) >= 20:
            merged.append(c)

    # 收尾：只有标题行、没有任何正文的块并入下一块，消除"空标题块"
    def _is_heading_only(c) -> bool:
        return not any(ln.strip() and not match_heading(ln) for ln in c["lines"])

    cleaned, pending = [], []
    for c in merged:
        if _is_heading_only(c):
            pending.extend(c["lines"])
            continue
        c = dict(c)
        if pending:
            c["lines"] = pending + c["lines"]
            pending = []
        cleaned.append(c)
    if pending:
        if cleaned:
            cleaned[-1]["lines"].extend(pending)
        else:
            cleaned.append({"heading": "概述", "level": 0, "lines": pending,
                            "heading_path": ""})
    merged = cleaned

    # 注意：这里仍用含 front-matter 的原文做类型识别，保持与旧实现一致的结果，
    # 避免本次修复顺带改变 doc_type / product 分类（实测会漂移 19 个文件）
    doc_type = detect_doc_type(file_path, text[:600])
    product = detect_product(file_path, text[:600])
    file_hash = _file_content_hash(file_path)

    # front-matter 元数据提升为 payload 字段（§5.2 展平 + 旧字段兼容）
    fm_title = str(front_matter.get("title") or "").strip()
    title = fm_title or Path(file_path).name.replace(".md", "")
    meta_payload = meta_to_payload(front_matter, mtime=_safe_mtime(file_path))
    extra = dict(meta_payload)
    for src_key, dst_key in (("source-id", "source_id"),
                             ("source-url", "source_url"),
                             ("section", "section"),
                             ("fetched", "fetched")):
        val = front_matter.get(src_key)
        if val:
            extra[dst_key] = str(val)

    results: list[dict] = []
    for i, c in enumerate(merged):
        raw_lines = c["lines"]
        raw_content = "\n".join(raw_lines)

        # G-H：剥离图片链接出嵌入文本，收进 image_refs
        stripped, image_refs = strip_images(raw_content)

        content = normalize_text(stripped).strip()
        if len(content) < 20:
            continue

        chunk_extra = dict(extra)
        chunk_extra["heading_path"] = c.get("heading_path") or c["heading"]
        if is_toc_text(raw_content):
            chunk_extra["is_toc"] = True
        if image_refs:
            chunk_extra["image_refs"] = image_refs
            caption, cap_source = _image_caption(
                c["heading"], raw_lines, title=title,
                heading_path=c.get("heading_path") or c["heading"], refs=image_refs)
            if caption:
                chunk_extra["image_captions"] = caption
                chunk_extra["image_caption_source"] = cap_source
                # 设计 §8.4：图注必须**参与嵌入**，否则「图注文字可命中」（T9）无从谈起
                content = f"{content}\n[图注] {caption}".strip()

        results.extend(finalize_chunks(
            content,
            heading=c["heading"],
            source=file_path,
            title=title,
            doc_type=doc_type,
            product=product,
            file_hash=file_hash,
            has_image=bool(image_refs),
            chunk_index=i,
            extra=chunk_extra,
        ))
    return results


# ==================== 统一入口（MD / PDF / Word / TXT） ====================
def parse_document(file_path: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == '.md':
        return parse_md(file_path)  # MD 元数据走文档内 front-matter
    # PDF/DOCX/TXT 无 front-matter 载体：sidecar <文档名>.meta.yaml（G-B），
    # 由各解析器默认自动加载
    if ext == '.pdf':
        return parse_pdf(file_path)
    if ext == '.docx':
        return parse_docx(file_path)
    if ext == '.txt':
        return parse_txt(file_path)
    raise ValueError(f"不支持的文件类型: {ext}")
