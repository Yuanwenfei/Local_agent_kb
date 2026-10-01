#!/usr/bin/env python3
"""
Qdrant payload 索引创建（幂等，可重复执行）

用法: python312\\python.exe tools\\ensure_schema.py

为设计 §5.2 新增字段建立 payload 索引；已存在的索引自动跳过。
text 索引（search_text）是精确串检索通道的底座（G-F）。
"""
import os
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType, TextIndexParams, TokenizerType

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("KB_COLLECTION", "emulate3d_docs")

KEYWORD_FIELDS = [
    "kind", "framework_version", "model_scope", "trust", "canonical_uid",
    "doc_version", "doc_type", "product", "heading_path", "source",
    "section",  # 检索工具的 section 过滤依赖该索引
    "revision",  # P3-8：插件产物 revision 变化 → 事实卡「待刷新」
    "superseded_by",  # P3-6：副本指向正本
    "ingested_by",  # P3：manual / em3d_export / qlp_export / kb_note
]
INT_FIELDS = ["doc_date", "version_rank"]
BOOL_FIELDS = ["is_canonical", "is_toc", "has_image"]


def main():
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    existing = set(qdrant.get_collection(COLLECTION_NAME).payload_schema or {})

    def create(field_name, field_schema):
        if field_name in existing:
            print(f"  [skip] {field_name} 已存在")
            return
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"  [ok]   {field_name}")
        except Exception as e:
            print(f"  [fail] {field_name}: {e}")

    print("== keyword 索引 ==")
    for f in KEYWORD_FIELDS:
        create(f, PayloadSchemaType.KEYWORD)

    print("== int 索引 ==")
    for f in INT_FIELDS:
        create(f, PayloadSchemaType.INTEGER)

    print("== bool 索引 ==")
    for f in BOOL_FIELDS:
        create(f, PayloadSchemaType.BOOL)

    print("== text 索引（search_text，多语种分词） ==")
    create("search_text", TextIndexParams(
        type=PayloadSchemaType.TEXT,
        tokenizer=TokenizerType.MULTILINGUAL,
        lowercase=True,
    ))

    print("完成")


if __name__ == "__main__":
    main()
