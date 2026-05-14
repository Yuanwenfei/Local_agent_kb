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


# ==================== PDF：页眉/页脚检测 ====================
def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    if a == b:
        return True
    if abs(len(a) - len(b)) / max(len(a), len(b)) > 0.5:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


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


def parse_pdf(file_path: str) -> list[dict]:
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
        class_name, method_name = _detect_class_method(text)
        chunks.append({
            "text": text[:1500],
            "heading": f"第{i + 1}页",
            "source": file_path,
            "title": title,
            "doc_type": doc_type,
            "product": product,
            "chunk_index": i,
            "page_number": i + 1,
            "has_image": has_image,
            "class_name": class_name,
            "method_name": method_name,
            "file_hash": file_hash,
        })
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


def parse_docx(file_path: str) -> list[dict]:
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
        class_name, method_name = _detect_class_method(text)
        chunks.append({
            "text": text[:1500],
            "heading": c["heading"],
            "source": file_path,
            "title": title,
            "doc_type": doc_type,
            "product": product,
            "chunk_index": i,
            "has_image": c["has_image"],
            "class_name": class_name,
            "method_name": method_name,
            "file_hash": file_hash,
        })
    return chunks


# ==================== 纯文本 (TXT) 解析 ====================
_TXT_CHUNK_MAX = 1500


def _split_long_block(block: str, max_size: int) -> list[str]:
    """超过 max_size 的段落按行累加切分，保证不在行中间断开"""
    if len(block) <= max_size:
        return [block]
    out = []
    current: list[str] = []
    current_len = 0
    for line in block.split('\n'):
        line_len = len(line) + 1  # +1 for '\n'
        if current and current_len + line_len > max_size:
            out.append('\n'.join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        out.append('\n'.join(current))
    return out


def parse_txt(file_path: str) -> list[dict]:
    """
    解析 .txt：保留缩进（含代码块），按空行段落切分，单块上限 1500 字符。

    元数据：路径优先，回落到首 600 字符内容启发式。
    """
    # 容错读取（少数日志可能带非 UTF-8 字节）
    raw = Path(file_path).read_text(encoding='utf-8', errors='replace')
    text = normalize_text(raw)
    if not text:
        return []

    # 段落切分：2+ 空行
    blocks = re.split(r'\n{2,}', text)
    capped: list[str] = []
    for block in blocks:
        block = block.strip('\n')
        if not block.strip():
            continue
        capped.extend(_split_long_block(block, _TXT_CHUNK_MAX))

    if not capped:
        return []

    preview = text[:600]
    doc_type = detect_doc_type(file_path, preview)
    product = detect_product(file_path, preview)
    title = Path(file_path).stem
    file_hash = _file_content_hash(file_path)

    chunks: list[dict] = []
    for i, content in enumerate(capped):
        content_stripped = content.strip()
        if len(content_stripped) < 20:
            continue
        # heading：首个非空行的前 60 字符
        heading = ''
        for ln in content.splitlines():
            if ln.strip():
                heading = ln.strip()[:60]
                break
        if not heading:
            heading = f"段{i + 1}"
        class_name, method_name = _detect_class_method(content)
        chunks.append({
            "text": content,
            "heading": heading,
            "source": file_path,
            "title": title,
            "doc_type": doc_type,
            "product": product,
            "chunk_index": i,
            "has_image": False,
            "class_name": class_name,
            "method_name": method_name,
            "file_hash": file_hash,
        })
    return chunks


# ==================== Markdown (MD) 解析 ====================
def parse_md(file_path: str) -> list[dict]:
    """
    解析 .md：按 H1-H4 标题层级切分，代码块保护，短片段合并。
    复用现有 split_md 逻辑，增加 normalize_text 和内容 fallback 分类。
    """
    text = Path(file_path).read_text(encoding='utf-8', errors='replace')
    if not text.strip():
        return []

    lines = text.splitlines()
    chunks_raw = []
    current = {"heading": "概述", "level": 0, "lines": []}
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        heading_match = re.match(r'^(#{1,4})\s+(.+)', line)
        if heading_match and not in_code_block:
            if current["lines"]:
                chunks_raw.append(current)
            level = len(heading_match.group(1))
            current = {
                "heading": heading_match.group(2),
                "level": level,
                "lines": [line]
            }
        else:
            current["lines"].append(line)

    if current["lines"]:
        chunks_raw.append(current)

    # 合并短片段
    merged = []
    for c in chunks_raw:
        content = "\n".join(c["lines"]).strip()
        if len(content) < 50 and c["level"] > 0 and merged:
            merged[-1]["lines"].extend(c["lines"])
        elif len(content) >= 20:
            merged.append(c)

    file_name = Path(file_path).name
    doc_type = detect_doc_type(file_path, text[:600])
    product = detect_product(file_path, text[:600])
    file_hash = _file_content_hash(file_path)

    results: list[dict] = []
    for i, c in enumerate(merged):
        content = normalize_text("\n".join(c["lines"])).strip()
        if len(content) < 20:
            continue
        class_name, method_name = _detect_class_method(content)
        results.append({
            "text": content[:1500],
            "heading": c["heading"],
            "source": file_path,
            "title": file_name.replace(".md", ""),
            "doc_type": doc_type,
            "product": product,
            "chunk_index": i,
            "class_name": class_name,
            "method_name": method_name,
            "file_hash": file_hash,
        })
    return results


# ==================== 统一入口（MD / PDF / Word / TXT） ====================
def parse_document(file_path: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == '.md':
        return parse_md(file_path)
    if ext == '.pdf':
        return parse_pdf(file_path)
    if ext == '.docx':
        return parse_docx(file_path)
    if ext == '.txt':
        return parse_txt(file_path)
    raise ValueError(f"不支持的文件类型: {ext}")
