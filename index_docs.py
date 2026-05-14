#!/usr/bin/env python3
"""
批量索引 Markdown / PDF / Word / TXT 文档到 Qdrant（支持增量更新 + rich 进度条）
用法: python index_docs.py /path/to/docs/folder
支持扩展名: .md / .pdf / .docx / .txt
"""
import os
import sys
import uuid
import hashlib
import json
import time
import math
from pathlib import Path

# 禁用 HuggingFace 网络检查，强制使用本地模型
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding

# PDF/Word/TXT/MD 解析器（同目录模块，统一入口）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from document_parsers import parse_document

# ==================== GPU 环境配置 ====================
_nvdiapath = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'nvidia')
if os.path.exists(_nvdiapath):
    for _pkg in os.listdir(_nvdiapath):
        _bindir = os.path.join(_nvdiapath, _pkg, 'bin')
        if os.path.exists(_bindir) and _bindir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _bindir + os.pathsep + os.environ.get('PATH', '')

# Windows 下强制 UTF-8 输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==================== rich 控制台 ====================
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console(force_terminal=True)

# ==================== 配置 ====================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("KB_COLLECTION", "emulate3d_docs")
EMBED_MODEL = "intfloat/multilingual-e5-large"
VECTOR_SIZE = 1024
BATCH_SIZE = 100
EMBED_BATCH = 32

# 支持的文档扩展名
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".docx", ".txt"}

# 脚本所在目录作为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
INDEX_STATE_FILE = os.path.join(PROJECT_ROOT, "index_state.json")

# ==================== 初始化 ====================
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ==================== 增量更新状态管理 ====================
def _is_abs_path(path: str) -> bool:
    """判断是否为绝对路径（兼容 Windows/Linux）"""
    return (len(path) > 1 and path[1] == ':') or path.startswith('/')


def load_index_state() -> dict:
    if os.path.exists(INDEX_STATE_FILE):
        try:
            with open(INDEX_STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # 防护：清理异常中断留下的脏状态（chunks < 0 表示未完成）
            cleaned = {}
            dirty = 0
            migrated = 0
            for k, v in state.items():
                if not isinstance(v, dict) or not isinstance(v.get("chunks"), int) or v["chunks"] < 0:
                    dirty += 1
                    continue
                # 兼容旧版：key 是绝对路径，迁移为 rel_path（统一正斜杠）
                if _is_abs_path(k) and "rel_path" in v:
                    cleaned[v["rel_path"].replace('\\', '/')] = v
                    migrated += 1
                else:
                    cleaned[k.replace('\\', '/')] = v
            if migrated:
                console.print(f"[yellow]⚠️ 发现 {migrated} 条旧版绝对路径记录，已自动迁移为相对路径[/yellow]")
            if dirty:
                console.print(f"[yellow]⚠️ 发现 {dirty} 条异常索引记录（可能上次中断导致），将强制重新索引[/yellow]")
            return cleaned
        except Exception:
            pass
    return {}

def save_index_state(state: dict):
    with open(INDEX_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def get_file_content_hash(file_path: str) -> str:
    """计算文件内容的 MD5 hash"""
    h = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def scan_files(folder_path: str, state: dict):
    """
    扫描文件（.md / .pdf / .docx / .txt），对比状态，返回:
    added: 新增文件列表
    modified: 修改文件列表
    deleted: 删除文件路径列表
    unchanged_count: 未变更文件数
    current_files: 当前所有文件信息字典
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文档目录不存在: {folder_path}")

    all_files = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith("~$")  # 排除 Word 临时锁文件
    ]
    current_files = {}

    for doc_file in all_files:
        full_path = str(doc_file.resolve())
        rel_path = str(doc_file.relative_to(folder)).replace('\\', '/')
        try:
            stat = doc_file.stat()
            content_hash = get_file_content_hash(full_path)
            current_files[rel_path] = {
                "full_path": full_path,
                "hash": content_hash,
                "mtime": stat.st_mtime,
                "rel_path": rel_path,
                "chunks": 0  # 占位，索引完成后更新
            }
        except Exception as e:
            console.print(f"[yellow]警告: 无法读取 {doc_file}: {e}[/yellow]")

    added = []
    modified = []
    unchanged = 0

    for rel_path, info in current_files.items():
        if rel_path not in state:
            added.append(info)
        elif state[rel_path]["hash"] != info["hash"]:
            modified.append(info)
        else:
            unchanged += 1

    deleted = [p for p in state if p not in current_files]

    return added, modified, deleted, unchanged, current_files

def delete_file_points(file_path: str) -> bool:
    """删除某个文件对应的所有 points"""
    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=file_path))]
            )
        )
        return True
    except Exception as e:
        console.print(f"[red]删除旧数据失败: {file_path} - {e}[/red]")
        return False

# ==================== 切分逻辑（detect 函数已统一从 document_parsers 导入） ====================

def parse_file(file_path: str) -> list[dict]:
    """统一入口：委托给 document_parsers.parse_document()"""
    return parse_document(file_path)

def _validate_vector(vec: list, preview: str = ""):
    """校验单个向量：维度、全零、NaN/Inf"""
    if len(vec) != VECTOR_SIZE:
        raise ValueError(f"向量维度异常: {len(vec)} != {VECTOR_SIZE}")
    if all(abs(x) < 1e-9 for x in vec):
        raise ValueError(f"生成零向量，embedding 模型异常。预览: {preview[:80]}")
    if any(math.isnan(x) or math.isinf(x) for x in vec):
        raise ValueError(f"向量含 NaN/Inf，embedding 模型异常。预览: {preview[:80]}")


def _model_health_check(embedder) -> list:
    """模型热身 + 健康检查：确认能输出非零向量"""
    test_texts = ["Hello world", "class RackConfigurator", "仓储自动化"]
    vectors = list(embedder.embed(test_texts))
    for txt, vec in zip(test_texts, vectors):
        v = vec.tolist()
        _validate_vector(v, txt)
    return vectors


def _smoke_test(qdrant, embedder):
    """索引完成后冒烟测试：确认查询能返回有效分数"""
    console.print("[bold cyan]🔍 执行入库后冒烟测试...[/bold cyan]")
    test_queries = ["conveyor belt", "IBatchEditable", "RackConfigurator"]
    for q in test_queries:
        vec = list(embedder.embed([q]))[0].tolist()
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=1,
            with_payload=False,
            score_threshold=0.0
        )
        if not results:
            raise RuntimeError(f"冒烟测试失败：查询 '{q}' 无返回，向量可能全零或未成功入库")
        score = results[0].score
        if score < 1e-6:
            raise RuntimeError(f"冒烟测试失败：查询 '{q}' 的最高相似度为 {score:.6f}，向量数据异常")
        console.print(f"  [green]✓[/green] '{q}' -> top_score={score:.4f}")
    console.print("[bold green]✅ 冒烟测试通过[/bold green]")


def ensure_collection():
    try:
        qdrant.get_collection(COLLECTION_NAME)
        return True
    except Exception:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        return False

# ==================== 主索引函数（rich 进度条版） ====================
def index_folder(folder_path: str):
    ensure_collection()
    state = load_index_state()
    
    # 1. 扫描文件
    console.rule("[bold blue]📁 扫描文件")
    with console.status("[bold green]正在扫描文档 (.md / .pdf / .docx / .txt)..."):
        try:
            added, modified, deleted, unchanged, current_files = scan_files(folder_path, state)
        except FileNotFoundError as e:
            console.print(f"[bold red]错误: {e}[/bold red]")
            return
    
    total_process = len(added) + len(modified) + len(deleted)
    
    # 删除保护：如果大量文件缺失，暂停并提示
    if deleted:
        total_state = len(state)
        if total_state > 0 and len(deleted) / total_state > 0.8:
            console.print(Panel(
                f"[bold red]⚠️ 检测到 {len(deleted)}/{total_state} ({len(deleted)/total_state*100:.0f}%) 的已索引文件在当前目录中缺失！\n\n"
                f"这可能是文档目录被移动、删除或配置错误导致的。[/bold red]\n"
                f"[yellow]为防止误删 Qdrant 数据，已跳过本次清理操作。[/yellow]\n\n"
                f"如需强制重新索引，请删除 {INDEX_STATE_FILE} 后重试。",
                title="数据保护",
                border_style="red"
            ))
            deleted = []
            total_process = len(added) + len(modified)
    
    # 显示扫描结果
    summary = Table.grid(padding=(0, 2))
    summary.add_row(
        f"[green]➕ 新增: {len(added)}[/green]",
        f"[yellow]✏️ 修改: {len(modified)}[/yellow]",
        f"[red]🗑️ 删除: {len(deleted)}[/red]",
        f"[dim]➖ 未变: {unchanged}[/dim]",
        f"[bold]📄 总计: {len(current_files)}[/bold]"
    )
    console.print(Panel(summary, title="扫描结果", border_style="blue"))
    
    if total_process == 0:
        console.print("[bold green]✅ 所有文件已是最新，无需更新！[/bold green]")
        save_index_state(state)
        return
    
    # 2. 处理删除
    if deleted:
        console.rule("[bold red]🗑️ 清理已删除文件")
        for rel_path in deleted:
            console.print(f"  [red]删除[/red] {Path(rel_path).name}")
            # 使用 state 中记录的旧 full_path 删除 Qdrant 数据（兼容旧版绝对路径）
            old_full_path = state[rel_path].get("full_path", rel_path)
            delete_file_points(old_full_path)
            del state[rel_path]
    
    # 3. 处理修改（先删旧数据）
    files_to_index = added + modified
    
    if modified:
        console.rule("[bold yellow]✏️ 清理修改文件的旧数据")
        for info in modified:
            rel_path = info['rel_path']
            console.print(f"  [yellow]清理[/yellow] {Path(rel_path).name}")
            old_full_path = state.get(rel_path, {}).get("full_path", info['full_path'])
            delete_file_points(old_full_path)
    
    # 4. 索引入库（带 rich 进度条）
    if not files_to_index:
        console.print("[bold green]✅ 更新完成！[/bold green]")
        save_index_state(state)
        return
    
    console.rule("[bold green]🚀 开始索引入库")
    
    # 加载模型 + 健康检查
    with console.status("[bold green]正在加载 Embedding 模型..."):
        embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE_DIR)
    _model_health_check(embedder)
    
    points = []
    embed_buffer = []
    total_chunks = 0
    total_processed_chunks = 0
    start_time = time.time()
    
    # 统计总 chunk 数（用于进度条）
    with console.status("[dim]预计算 chunk 数量..."):
        total_expected_chunks = 0
        for info in files_to_index:
            try:
                chunks = parse_file(info['full_path'])
                total_expected_chunks += len(chunks)
            except Exception:
                pass
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("[bold]{task.fields[speed]}", justify="right"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        file_task = progress.add_task("[cyan]文件处理", total=len(files_to_index), speed="")
        chunk_task = progress.add_task("[green]片段入库", total=total_expected_chunks, speed="")
        
        def _flush_embed():
            nonlocal points, embed_buffer, total_chunks, total_processed_chunks
            if not embed_buffer:
                return
            texts = [c["text"] for c in embed_buffer]
            vectors = list(embedder.embed(texts))
            for chunk, vec in zip(embed_buffer, vectors):
                vec_list = vec.tolist()
                _validate_vector(vec_list, chunk.get("text", ""))
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec_list,
                    payload=chunk
                ))
                total_chunks += 1
                total_processed_chunks += 1
                progress.advance(chunk_task)
            embed_buffer.clear()
        
        def _flush_points(force=False):
            nonlocal points
            while len(points) >= BATCH_SIZE or (force and points):
                batch = points[:BATCH_SIZE] if len(points) >= BATCH_SIZE else points
                qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
                points = points[len(batch):]
        
        for info in files_to_index:
            full_path = info['full_path']
            try:
                chunks = parse_file(full_path)

                for chunk in chunks:
                    embed_buffer.append(chunk)
                    if len(embed_buffer) >= EMBED_BATCH:
                        _flush_embed()
                        _flush_points()

                # 更新状态
                state[rel_path] = {
                    "hash": info["hash"],
                    "mtime": info["mtime"],
                    "rel_path": rel_path,
                    "full_path": full_path,
                    "chunks": len(chunks)
                }

                # 更新文件进度
                elapsed = time.time() - start_time
                speed = f"{total_processed_chunks / elapsed:.1f} chunks/s" if elapsed > 0 else ""
                progress.update(file_task, advance=1, speed=speed)

            except Exception as e:
                progress.console.print(f"[red]错误: {full_path} - {e}[/red]")
                progress.advance(file_task)
        
        _flush_embed()
        _flush_points(force=True)
    
    # 入库完成后冒烟测试（验证向量真的有效）
    _smoke_test(qdrant, embedder)

    save_index_state(state)
    
    # 最终统计
    elapsed = time.time() - start_time
    result_table = Table(title="索引完成", box=box.ROUNDED)
    result_table.add_column("指标", style="cyan")
    result_table.add_column("数值", style="green", justify="right")
    result_table.add_row("处理文件", str(len(files_to_index)))
    result_table.add_row("入库片段", str(total_chunks))
    result_table.add_row("耗时", f"{elapsed:.1f} 秒")
    result_table.add_row("平均速度", f"{total_chunks / elapsed:.1f} chunks/s" if elapsed > 0 else "N/A")
    result_table.add_row("当前总片段", str(qdrant.get_collection(COLLECTION_NAME).points_count))
    console.print(result_table)
    console.print(f"[bold green]✅ 索引完成！状态已保存到 {INDEX_STATE_FILE}[/bold green]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[bold red]用法: python index_docs.py /path/to/docs/folder[/bold red]")
        console.print("[dim]支持扩展名: .md / .pdf / .docx / .txt[/dim]")
        sys.exit(1)
    index_folder(sys.argv[1])
