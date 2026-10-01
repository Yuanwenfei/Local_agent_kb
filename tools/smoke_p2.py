#!/usr/bin/env python3
"""P2 检索层冒烟：三通道 / 过滤扩展 / 后处理 / 返回体 v2 / kb_stats（含 T11 计时）

用法:
  python312\\python.exe tools\\smoke_p2.py                # 全部用例
  python312\\python.exe tools\\smoke_p2.py --case exact   # 只跑某个用例

说明：集合若尚未重建（缺 sparse 配置），hybrid 会自动降级并在返回体里提示，
      这不是失败；要验证 sparse 通道需先重建集合并全量重跑（见开发计划 §12）。
"""
import os
import sys
import time
import asyncio
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('ORT_LOGGING_LEVEL', 'ERROR')

import kb_mcp_server as srv


def _text(res) -> str:
    return "\n".join(c.text for c in res)


CASES = {
    "hybrid": {"query": "RackConfigurator GenerateSlots 用法", "top_k": 3},
    "semantic": {"query": "AGV 电池充电策略", "mode": "semantic", "top_k": 3},
    "exact": {"query": "gripper component must exist as a child of", "mode": "exact", "top_k": 3},
    "exact2": {"query": "does not have an AMR Controller as child", "mode": "exact", "top_k": 3},
    "exact3": {"query": "rack pick location", "mode": "exact", "top_k": 3},
    "exact4": {"query": "set [rackJob] after rack pick targets", "mode": "exact", "top_k": 3},
    "filter+toc": {"query": "conveyor belt 运行参数", "top_k": 3, "section": "Basics"},
    "drift": {"query": "AMRController.Bot 类型", "version": "9.9.9.9", "top_k": 3},
    "trust": {"query": "ACR 层级结构", "trust": ["measured"], "top_k": 3},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="只跑指定用例")
    args = ap.parse_args()

    names = [args.case] if args.case else list(CASES)

    for name in names:
        if name not in CASES:
            print(f"未知用例: {name}，可选: {list(CASES)}")
            continue
        t0 = time.time()
        body = _text(asyncio.run(srv._handle_search(CASES[name])))
        dt = time.time() - t0
        print("=" * 78)
        print(f"[{name}] {CASES[name]}  耗时 {dt * 1000:.0f} ms")
        print("=" * 78)
        print(body[:2200])
        print()

    t0 = time.time()
    stats = _text(asyncio.run(srv._handle_kb_stats({})))
    print("=" * 78)
    print(f"[kb_stats] 默认（count 聚合）耗时 {(time.time() - t0) * 1000:.0f} ms")
    print("=" * 78)
    print(stats)
    print()

    t0 = time.time()
    _text(asyncio.run(srv._handle_kb_stats({"detail": True})))
    print(f"[kb_stats detail=true] 全扫耗时 {(time.time() - t0) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
