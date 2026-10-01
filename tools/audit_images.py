#!/usr/bin/env python3
"""图片链接审计（解析层，无需 Qdrant）：统计剥离是否彻底、路径能否解析

用法:
  python312\\python.exe tools\\audit_images.py                    # 审计 md-source
  python312\\python.exe tools\\audit_images.py <语料目录>
  python312\\python.exe tools\\audit_images.py --leaks 20         # 顺带列出残留/未解析样例

用途：
- 入库前后核对「嵌入文本是否还有图片噪声」；
- 语料更新后核对新增图片的可解析率（判断是否需要补 assets 或改写作规范）。
"""
import os
import sys
import argparse
import collections

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kb_schema import strip_images, resolve_image_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=os.path.join(PROJECT_ROOT, "md-source"))
    ap.add_argument("--leaks", type=int, default=0, help="列出前 N 条残留/未解析样例")
    args = ap.parse_args()

    files = refs_total = local_ok = ext_url = unresolved = leak_files = 0
    unresolved_samples = []
    for dp, _, fs in os.walk(args.folder):
        for f in fs:
            if not f.lower().endswith('.md'):
                continue
            p = os.path.join(dp, f)
            try:
                text = open(p, encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            files += 1
            cleaned, refs = strip_images(text)
            if '![' in cleaned or '<img' in cleaned:
                leak_files += 1
                if args.leaks:
                    print(f"[残留] {os.path.relpath(p, args.folder)}: "
                          f"{[l for l in cleaned.splitlines() if '![' in l or '<img' in l][:1]}")
            refs_total += len(refs)
            for r, ap_ in zip(refs, resolve_image_paths(p, refs)):
                if ap_.lower().startswith(('http://', 'https://')):
                    ext_url += 1
                elif os.path.exists(ap_):
                    local_ok += 1
                else:
                    unresolved += 1
                    if args.leaks and len(unresolved_samples) < args.leaks:
                        unresolved_samples.append((os.path.relpath(p, args.folder), r))

    print(f"扫描 MD 文件: {files}")
    print(f"图片引用总数: {refs_total}")
    print(f"  本地可达: {local_ok} ({local_ok / max(refs_total, 1) * 100:.1f}%)")
    print(f"  外链/站内 URL: {ext_url}")
    print(f"  未解析（语料内无对应文件）: {unresolved}")
    print(f"剥离后仍有图片标记的文件: {leak_files}")
    for s in unresolved_samples:
        print("  [未解析]", s)


if __name__ == "__main__":
    main()
