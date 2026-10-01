#!/usr/bin/env python3
"""P2 检索层验收（走真实检索入口 _search_structured，非旁路查询）

用法:
  python312\\python.exe regression\\eval_p2.py                 # hybrid 全量回归 + 断言
  python312\\python.exe regression\\eval_p2.py --mode semantic
  python312\\python.exe regression\\eval_p2.py --compare baseline

覆盖断言：
  T3  目录页不进 Top3、单文档 ≤2、MRR@5/Recall@5 不劣于 P0 基线
  T4  Top5 分数梯度 ≥0.03（P0 基线 0.0083）
  T2  版本过滤：指定 version 时结果要么版本一致，要么显式 drift 告警
  T11 单次检索耗时 + kb_stats(detail=false) 不全扫
结果写入 regression/results/p2_<时间戳>.json
"""
import os
import sys
import json
import time
import argparse
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('ORT_LOGGING_LEVEL', 'ERROR')

QUERIES_FILE = os.path.join(PROJECT_ROOT, "regression", "queries.jsonl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "regression", "results")
TOP_K = 5


def load_queries():
    out = []
    with open(QUERIES_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run(mode: str):
    import kb_mcp_server as srv

    queries = load_queries()
    per_query, times = [], []

    for q in queries:
        args = {"query": q["query"], "top_k": TOP_K, "mode": mode}
        t0 = time.time()
        items, meta = srv._search_structured(args)
        times.append(time.time() - t0)

        expect_any = q.get("expect_any", [])
        expect_no_hit = q.get("expect_no_hit", False)

        ranked, first_rel = [], None
        for rank, it in enumerate(items, 1):
            text = (it["payload"].get("text") or "")
            rel = any(k.lower() in text.lower() for k in expect_any) if expect_any else False
            if rel and first_rel is None:
                first_rel = rank
            ranked.append({
                "rank": rank,
                "score": round(it["score"], 5),
                "source": it["payload"].get("source", ""),
                "heading": it["payload"].get("heading", ""),
                "relevant": rel,
                "is_toc": bool(it["payload"].get("is_toc")),
                "channels": it["channels"],
                "trust": it["payload"].get("trust", ""),
                "version": it["payload"].get("framework_version", ""),
                "exactness": it["exactness"],
            })

        scores = [h["score"] for h in ranked]
        sources = [h["source"] for h in ranked]
        per_query.append({
            "id": q["id"], "query": q["query"], "note": q.get("note", ""),
            "expect_no_hit": expect_no_hit,
            "top5": ranked,
            "mrr5": round(1.0 / first_rel, 4) if first_rel else 0.0,
            "recall5": 1.0 if first_rel else 0.0,
            "gradient": round(max(scores) - min(scores), 4) if len(scores) > 1 else 0.0,
            "top1_toc": ranked[0]["is_toc"] if ranked else False,
            "top3_has_toc": any(h["is_toc"] for h in ranked[:3]),
            "max_per_source": max(sources.count(s) for s in set(sources)) if sources else 0,
            "n_results": len(ranked),
            "low_conf": bool(meta.get("low_conf")),
            "drift": bool(meta.get("drift")),
            "notes": meta.get("notes", []),
        })

    return per_query, times


def _real_framework_version() -> str:
    """取库内真实存在、且最常见的 framework_version（用于 T2 的"版本存在"分支）。

    2026-10-01：原实现把该分支硬编码成 `version="unknown"`——当时全库只有 3 篇自研文档
    是 `unknown`，过滤能命中；P4-1 补齐元数据后**没有任何文档是 unknown 了**，该断言随前提
    一起失效（行为本身正确：无匹配 → 放宽 + drift 告警）。所以改为动态取真实版本。
    """
    from collections import Counter
    import kb_mcp_server as srv

    pts, _off = srv.get_qdrant().scroll(
        collection_name=srv.COLLECTION_NAME, limit=2000, offset=None,
        with_payload=["framework_version"], with_vectors=False)
    c = Counter(p.payload.get("framework_version") for p in pts)
    c.pop(None, None)
    c.pop("", None)
    c.pop("unknown", None)
    return c.most_common(1)[0][0] if c else ""


def t2_version_check(mode: str):
    """T2：指定一个不存在的版本号 → 验证 drift 告警；再指定真实存在的版本 → 只返回该版本、无 drift。"""
    import kb_mcp_server as srv

    items_bad, meta_bad = srv._search_structured(
        {"query": "AMRController.Bot 类型", "version": "9.9.9.9", "top_k": 3, "mode": mode})
    versions_bad = {it["payload"].get("framework_version") for it in items_bad}

    real = _real_framework_version()
    items_ok, meta_ok = srv._search_structured(
        {"query": "AMRController.Bot 类型", "version": real, "top_k": 3, "mode": mode})
    versions_ok = {it["payload"].get("framework_version") for it in items_ok}

    return {
        "drift_flagged": bool(meta_bad.get("drift")),
        "drift_versions": sorted(v for v in versions_bad if v),
        "exact_version_passed": real,
        "exact_version_filtered": bool(items_ok) and versions_ok == {real} and not meta_ok.get("drift"),
        "n_exact_filtered": len(items_ok),
    }


def t_recency_check():
    """T14：时效衰减口径（2026-10-01 A 方案）——只对「结论类」kind 生效。

    教程/手册/API 参考恒定 1.0（过时由版本字段表达）；结论类 6 个月线性降到 0.85；
    `date_inferred`（日期不可信）一律豁免。
    """
    import kb_mcp_server as srv

    today = datetime.date.today()

    def d(days_ago):
        return int((today - datetime.timedelta(days=days_ago)).strftime("%Y%m%d"))

    cases = {
        "教程（tutorial）不衰减": ({"kind": "doc", "doc_type": "tutorial",
                                    "trust": "tutorial", "doc_date": d(400)}, 1.0),
        "官方手册（measured）不衰减": ({"kind": "doc", "doc_type": "user_manual",
                                        "trust": "measured", "doc_date": d(400)}, 1.0),
        "结论类新条目不衰减": ({"kind": "error_faq", "doc_date": d(0)}, 1.0),
        "结论类线性中点（91 天）": ({"kind": "note", "doc_date": d(91)}, 0.9252),
        "结论类半年后触底（183 天）": ({"kind": "defect_log", "doc_date": d(183)}, 0.85),
        "结论类更久仍在下限（400 天）": ({"kind": "version_matrix", "doc_date": d(400)}, 0.85),
        "结论类但日期为推断→豁免": ({"kind": "model_fact_card", "doc_date": d(400),
                                     "date_inferred": True}, 1.0),
        "默认 kind=doc 无日期": ({"kind": "doc"}, 1.0),
    }
    out, all_ok = {}, True
    for name, (payload, expect) in cases.items():
        got, note = srv._recency_factor(payload)
        ok = abs(got - expect) < 0.005
        out[name] = {"factor": round(got, 4), "expect": expect, "ok": ok, "note": note}
        all_ok = all_ok and ok

    # 端到端：真跑一遍 _postprocess，确认"时效×…"标注只出现在结论类旧条目上
    class _Hit:
        def __init__(self, payload):
            self.payload = payload
            self.id = "probe"
            self.score = 0.5

    def _ranked(payload):
        h = _Hit(payload)
        return [(h, 0.5, ["dense"], 0.0)]

    tut = srv._postprocess(_ranked({"kind": "doc", "doc_type": "tutorial", "doc_date": d(400)}), 2)[0]
    old = srv._postprocess(_ranked({"kind": "defect_log", "doc_date": d(200)}), 2)[0]
    e2e = {"教程 reasons": tut["reasons"], "结论类旧条目 reasons": old["reasons"]}
    e2e_ok = (not any("时效" in r for r in tut["reasons"])
              and any("时效" in r for r in old["reasons"])
              and old["score"] < tut["score"])

    return {"pass": all_ok and e2e_ok, "cases": out, "e2e": e2e}


def t11_perf():
    import asyncio
    import kb_mcp_server as srv

    t0 = time.time()
    asyncio.run(srv._handle_kb_stats({}))
    fast = time.time() - t0

    t0 = time.time()
    asyncio.run(srv._handle_kb_stats({"detail": True}))
    slow = time.time() - t0
    return {"kb_stats_default_s": round(fast, 3), "kb_stats_detail_s": round(slow, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "semantic", "exact"])
    ap.add_argument("--compare", metavar="NAME", help="与 regression/results/<NAME>.json 的 aggregate 对比")
    args = ap.parse_args()

    print(f"P2 验收：mode={args.mode}，查询 {len(load_queries())} 条（首次会加载模型，约数十秒）", file=sys.stderr)

    per_query, times = run(args.mode)
    t2 = t2_version_check(args.mode)
    trec = t_recency_check()
    t11 = t11_perf()

    judged = [q for q in per_query if not q["expect_no_hit"]]
    n = len(judged) or 1
    agg = {
        "mode": args.mode,
        "mrr5": round(sum(q["mrr5"] for q in judged) / n, 4),
        "recall5": round(sum(q["recall5"] for q in judged) / n, 4),
        "avg_gradient": round(sum(q["gradient"] for q in judged) / n, 4),
        "top1_toc_count": sum(1 for q in per_query if q["top1_toc"]),
        "top3_toc_count": sum(1 for q in per_query if q["top3_has_toc"]),
        "max_per_source_violations": sum(1 for q in per_query if q["max_per_source"] > 2),
        "low_conf_count": sum(1 for q in per_query if q["low_conf"]),
        "drift_count": sum(1 for q in per_query if q["drift"]),
        "avg_latency_ms": round(sum(times) / len(times) * 1000, 1),
        "median_latency_ms": round(sorted(times)[len(times) // 2] * 1000, 1),
        "no_hit_probes": [q["id"] for q in per_query if q["expect_no_hit"] and q["n_results"] == 0],
        "unexpected_hits_probes": [q["id"] for q in per_query if q["expect_no_hit"] and q["n_results"] > 0],
        "t2": t2, "t11": t11, "t_recency": trec,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"p2_{stamp}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({"timestamp": stamp, "aggregate": agg, "per_query": per_query},
                  f, ensure_ascii=False, indent=2)

    # ---- 断言输出 ----
    print("\n## P2 验收指标")
    print("| 指标 | 值 |")
    print("|---|---|")
    for k in ("mode", "mrr5", "recall5", "avg_gradient", "top1_toc_count", "top3_toc_count",
              "max_per_source_violations", "low_conf_count", "drift_count",
              "avg_latency_ms", "median_latency_ms"):
        print(f"| {k} | {agg[k]} |")

    print("\n## 断言")
    checks = [
        ("T4 分数梯度 ≥0.03", agg["avg_gradient"] >= 0.03, f"实测 {agg['avg_gradient']}"),
        ("T3 无单文档超过 2 条", agg["max_per_source_violations"] == 0,
         f"违规 {agg['max_per_source_violations']} 条查询"),
        ("T2 version-drift 告警生效", t2["drift_flagged"] is True, str(t2)),
        ("T2 精确版本过滤生效", t2["exact_version_filtered"] is not False, str(t2)),
        ("T11 kb_stats 默认不全扫", t11["kb_stats_default_s"] <= max(t11["kb_stats_detail_s"], 0.5),
         f"默认 {t11['kb_stats_default_s']}s / detail {t11['kb_stats_detail_s']}s"),
        ("T14 时效衰减只作用于结论类 kind", trec["pass"],
         "；".join(f"{k}={v['factor']}" for k, v in trec["cases"].items())),
    ]
    for name, ok, detail in checks:
        print(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")

    print(f"\n结果已保存: {out_file}")

    if args.compare:
        base_file = os.path.join(RESULTS_DIR, f"{args.compare}.json")
        with open(base_file, encoding='utf-8') as f:
            base = json.load(f)["aggregate"]
        print("\n## 与 P0 基线对比")
        print("| 指标 | P0 基线 | P2 本次 | 变化 |")
        print("|---|---|---|---|")
        for k in ("mrr5", "recall5", "avg_gradient", "top1_toc_count"):
            if k in base and k in agg:
                print(f"| {k} | {base[k]} | {agg[k]} | {agg[k] - base[k]:+.4f} |")


if __name__ == "__main__":
    main()
