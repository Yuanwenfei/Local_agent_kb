#!/usr/bin/env python3
"""
存量 payload 元数据回填（G-D）：不重新嵌入，只更新 payload 字段。

适用场景：
- 只修了 sidecar/front-matter（元数据），不想重嵌向量时；
- --reindex-all 全量重嵌之外的轻量修补通道。

用法:
  python312\\python.exe tools\\backfill_payload.py <文件路径> [<文件路径>...] [--dry-run]
  python312\\python.exe tools\\backfill_payload.py --all-in-state [--dry-run]

注意：只回填元数据字段（kind/doc_version/doc_date/framework_version/model_scope/
trust/version_rank/canonical_uid/is_canonical/ingested_by/source_origin/date_inferred/
superseded_by），不触碰 text/vector/image 相关字段。
"""
import os
import sys
import argparse

os.environ.setdefault('HF_HUB_OFFLINE', '1')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from kb_schema import meta_to_payload, parse_meta_yaml, load_sidecar_meta

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("KB_COLLECTION", "emulate3d_docs")
INDEX_STATE_FILE = os.path.join(PROJECT_ROOT, "index_state.json")

META_FIELDS = ("kind", "doc_version", "doc_date", "framework_version", "model_scope",
               "trust", "version_rank", "canonical_uid", "is_canonical", "ingested_by",
               "source_origin", "date_inferred", "superseded_by")


def meta_for_file(file_path: str) -> dict:
    """front-matter（MD）或 sidecar（PDF/DOCX/TXT）→ 元数据 payload"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".md":
        from document_parsers import parse_front_matter
        with open(file_path, encoding='utf-8', errors='replace') as f:
            text = f.read()
        meta, _ = parse_front_matter(text)
    else:
        meta = load_sidecar_meta(file_path)
    return meta_to_payload(meta, mtime=_safe_mtime(file_path))


def _safe_mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def backfill(qdrant, file_path: str, dry_run: bool, sync: bool = False) -> int:
    # payload 里的 `source` 是**绝对路径**，过滤必须用同一形式，否则静默 0 命中
    file_path = os.path.abspath(file_path)
    payload = {k: v for k, v in meta_for_file(file_path).items() if k in META_FIELDS}
    filt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=file_path))])
    if dry_run:
        count = qdrant.count(COLLECTION_NAME, count_filter=filt).count
        print(f"  [dry] {os.path.basename(file_path)}: 将更新 {count} 点 -> {payload}")
        return count
    qdrant.set_payload(COLLECTION_NAME, payload=payload, points=filt)
    # --sync：`set_payload` 是合并语义，旧键不会被清掉。前端的元数据键如果在新 payload
    # 里已不存在（典型：`date_inferred` 在补了真实 doc_date 后应失效），必须显式删除，
    # 否则会留下自相矛盾的状态（有权威日期却仍标"推断"）。
    stale = [k for k in META_FIELDS if k not in payload]
    if sync and stale:
        qdrant.delete_payload(COLLECTION_NAME, keys=stale, points=filt)
    count = qdrant.count(COLLECTION_NAME, count_filter=filt).count
    extra = f"（清理过期键 {','.join(stale)}）" if (sync and stale) else ""
    print(f"  [ok]  {os.path.basename(file_path)}: 已更新 {count} 点{extra}")
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="要回填的文件路径")
    ap.add_argument("--all-in-state", action="store_true", help="回填 index_state 中全部现存文件")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sync", action="store_true",
                    help="同时删除新 payload 里已不存在的元数据键（典型：date_inferred）")
    args = ap.parse_args()

    files = list(args.files)
    if args.all_in_state:
        import json
        with open(INDEX_STATE_FILE, encoding='utf-8') as f:
            state = json.load(f)
        for info in state.values():
            if os.path.exists(info["full_path"]):
                files.append(info["full_path"])
    if not files:
        ap.error("未指定文件（或用 --all-in-state）")

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    total = 0
    for fp in files:
        if not os.path.exists(fp):
            print(f"  [skip] 文件不存在: {fp}")
            continue
        total += backfill(qdrant, fp, args.dry_run, args.sync)
    print(f"完成：共 {total} 点{'（dry-run，未写入）' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
