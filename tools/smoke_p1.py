#!/usr/bin/env python3
"""P1 冒烟验收：治理工具 + 检索返回体图片块（T1/T8/T12 快速自检）

用法:
  python tools/smoke_p1.py                       # 治理工具（不加载模型，秒级）
  python tools/smoke_p1.py --search "AGV 层级配置"  # 额外验证 search 返回体（需加载模型，约 30-60s）
"""
import os
import sys
import asyncio
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('ORT_LOGGING_LEVEL', 'ERROR')


def _text(res) -> str:
    return "\n".join(c.text for c in res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", metavar="QUERY", help="额外跑一次检索，打印返回体（验证图片路径块）")
    ap.add_argument("--limit", type=int, default=5, help="清单显示条数")
    args = ap.parse_args()

    import kb_mcp_server as srv

    print("=" * 70)
    print("[1] kb_list_docs（T8 前置：文档清单 + 元数据）")
    print("=" * 70)
    print(_text(asyncio.run(srv._handle_kb_list_docs({"limit": args.limit}))))

    print()
    print("=" * 70)
    print("[2] kb_meta_lint（T1：缺元数据 / 未回填 / canonical 冲突）")
    print("=" * 70)
    print(_text(asyncio.run(srv._handle_kb_meta_lint({"top_docs": 20, "limit": args.limit}))))

    if args.search:
        print()
        print("=" * 70)
        print(f"[3] search_tech_kb('{args.search}') 返回体（T12：图片绝对路径块）")
        print("=" * 70)
        body = _text(asyncio.run(srv._handle_search({"query": args.search, "top_k": 3,
                                                     "score_threshold": 0.5})))
        print(body)
        has_block = "📎" in body
        has_abs = ":" in body and "\\" in body
        print()
        print(f"[检查] 含图片提示块: {has_block} | 提示块含绝对路径: {has_abs}")


if __name__ == "__main__":
    main()
