#!/usr/bin/env python3
"""
KB 检索回归度量脚本（P0 基线 / 各阶段门禁）

用法:
  python regression/eval.py                     # 跑全部查询，结果存 regression/results/<时间戳>.json
  python regression/eval.py --compare baseline  # 与 baseline 结果对比出报告

度量指标（对每条查询，top_k=5，不设 score_threshold 以观察完整分数分布）:
  - MRR@5 / Recall@5      : 按 expect_any 关键词命中判定相关性（缺口探针用 expect_no_hit）
  - 分数梯度              : Top5 分数 max-min（基线病灶：0.855-0.891 无梯度）
  - Top1 是否目录页        : 链接列表占主导判定（TOC 探针 q15 及一般查询）
  - 单文档占比            : Top5 中同一 source 的最大占比（基线病灶：被单文档占满）
另记录 score_threshold=0.65（生产默认值）下的命中情况。
"""
import os
import sys
import json
import argparse
import datetime

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('ORT_LOGGING_LEVEL', 'ERROR')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 图片链接剥离的验收指标（T12）用同一套正则/路径解析，避免与解析层判定不一致
from kb_schema import IMAGE_RE, resolve_image_paths

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("KB_COLLECTION", "emulate3d_docs")
EMBED_MODEL = "intfloat/multilingual-e5-large"
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
USE_GPU = os.getenv("KB_USE_GPU", "0") == "1"
ONNX_PROVIDERS = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                  if USE_GPU else ["CPUExecutionProvider"])

QUERIES_FILE = os.path.join(PROJECT_ROOT, "regression", "queries.jsonl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "regression", "results")
DEFAULT_THRESHOLD = 0.65  # 生产默认，单独记录
TOP_K = 5


def is_toc_chunk(text: str) -> bool:
    """目录页判定：非空行中 >40% 是 markdown 链接列表项（- [x](y)）"""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    linkish = sum(1 for ln in lines if ln.lstrip().startswith(('- [', '* [')) or '](' in ln)
    return linkish / len(lines) > 0.4


def load_queries():
    queries = []
    with open(QUERIES_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def run(queries):
    from qdrant_client import QdrantClient
    from fastembed import TextEmbedding

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE_DIR,
                             providers=ONNX_PROVIDERS)

    per_query = []
    for q in queries:
        vec = list(embedder.embed([q["query"]]))[0].tolist()
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=TOP_K,
            with_payload=True,
        )
        hits65 = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=TOP_K,
            with_payload=True,
            score_threshold=DEFAULT_THRESHOLD,
        )

        expect_any = q.get("expect_any", [])
        expect_no_hit = q.get("expect_no_hit", False)

        ranked = []
        first_rel_rank = None
        img_chars = 0
        text_chars = 0
        img_refs_total = 0
        img_paths_ok = 0
        img_paths_missing = 0
        img_external = 0
        md_img_flag = 0          # top5 中 has_image=True 的片段数（MD 修复前恒 0）
        for rank, hit in enumerate(hits, 1):
            text = hit.payload.get("text", "") or ""
            rel = any(k.lower() in text.lower() for k in expect_any) if expect_any else False
            if rel and first_rel_rank is None:
                first_rel_rank = rank
            # T12：嵌入文本里的图片链接残留（重嵌后应趋近 0，基线约 16.2%）
            text_chars += len(text)
            img_chars += sum(len(m.group(0)) for m in IMAGE_RE.finditer(text))
            refs = hit.payload.get("image_refs") or []
            img_refs_total += len(refs)
            if hit.payload.get("has_image"):
                md_img_flag += 1
            for ap in resolve_image_paths(hit.payload.get("source", ""), refs):
                if ap.lower().startswith(('http://', 'https://')):
                    img_external += 1          # 外链本地不可读，不计入可达率分母
                elif os.path.exists(ap):
                    img_paths_ok += 1
                else:
                    img_paths_missing += 1
            ranked.append({
                "rank": rank,
                "score": round(hit.score, 4),
                "source": hit.payload.get("source", ""),
                "title": hit.payload.get("title", ""),
                "heading": hit.payload.get("heading", ""),
                "relevant": rel,
                "is_toc": is_toc_chunk(text),
                "image_refs": len(refs),
            })

        scores = [h["score"] for h in ranked]
        sources = [h["source"] for h in ranked]
        max_doc_ratio = (max(sources.count(s) for s in set(sources)) / len(sources)) if sources else 0

        per_query.append({
            "id": q["id"],
            "query": q["query"],
            "note": q.get("note", ""),
            "toc_probe": q.get("toc_probe", False),
            "expect_no_hit": expect_no_hit,
            "top5": ranked,
            "mrr5": round(1.0 / first_rel_rank, 4) if first_rel_rank else 0.0,
            "recall5": 1.0 if first_rel_rank else 0.0,
            "gradient": round((max(scores) - min(scores)), 4) if scores else 0.0,
            "top1_toc": ranked[0]["is_toc"] if ranked else False,
            "max_doc_ratio": round(max_doc_ratio, 2),
            "hit_at_default_threshold": len(hits65) > 0,
            # T12 图片指标
            "img_link_char_ratio": round(img_chars / text_chars, 4) if text_chars else 0.0,
            "image_refs_total": img_refs_total,
            "image_paths_ok": img_paths_ok,
            "image_paths_missing": img_paths_missing,
            "image_external": img_external,
            "has_image_chunks": md_img_flag,
        })

    embedder = None
    return per_query


def aggregate(per_query):
    judged = [q for q in per_query if not q["expect_no_hit"]]
    n = len(judged) or 1
    avg_gradient = sum(q["gradient"] for q in judged) / n
    refs_total = sum(q["image_refs_total"] for q in per_query)
    return {
        "mrr5": round(sum(q["mrr5"] for q in judged) / n, 4),
        "recall5": round(sum(q["recall5"] for q in judged) / n, 4),
        "avg_gradient": round(avg_gradient, 4),
        "top1_toc_count": sum(1 for q in per_query if q["top1_toc"]),
        "dominated_queries": sum(1 for q in judged if q["max_doc_ratio"] > 0.6),
        "zero_hit_default_threshold": sum(1 for q in per_query if not q["hit_at_default_threshold"]),
        "no_hit_probes": [q["id"] for q in per_query if q["expect_no_hit"] and not q["recall5"]],
        "unexpected_hits_probes": [q["id"] for q in per_query if q["expect_no_hit"] and q["recall5"]],
        # T12：图片剥离验收（img_link_char_ratio 应 ≈0；image_refs_total>0 说明剥离链路生效）
        "img_link_char_ratio_avg": round(
            sum(q["img_link_char_ratio"] for q in per_query) / (len(per_query) or 1), 4),
        "image_refs_total": refs_total,
        "image_paths_ok": sum(q["image_paths_ok"] for q in per_query),
        "image_paths_missing": sum(q["image_paths_missing"] for q in per_query),
        "image_external": sum(q["image_external"] for q in per_query),
        "queries_with_image_refs": sum(1 for q in per_query if q["image_refs_total"] > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", metavar="NAME", help="与 regression/results/<NAME>.json 对比并输出报告")
    args = ap.parse_args()

    queries = load_queries()
    print(f"加载查询 {len(queries)} 条，开始度量（模型加载约需数十秒）...", file=sys.stderr)
    per_query = run(queries)
    agg = aggregate(per_query)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"{stamp}.json")
    payload = {"timestamp": stamp, "collection": COLLECTION_NAME, "top_k": TOP_K,
               "aggregate": agg, "per_query": per_query}
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_file}", file=sys.stderr)

    # 同时更新 latest.json 方便 --compare baseline
    latest = os.path.join(RESULTS_DIR, "latest.json")
    with open(latest, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(agg, ensure_ascii=False, indent=2))

    if args.compare:
        base_file = os.path.join(RESULTS_DIR, f"{args.compare}.json")
        with open(base_file, encoding='utf-8') as f:
            base = json.load(f)
        b, a = base["aggregate"], agg
        print("\n## 与基线对比")
        print("| 指标 | 基线 | 本次 | 变化 |")
        print("|---|---|---|---|")
        for key in ("mrr5", "recall5", "avg_gradient", "top1_toc_count",
                    "dominated_queries", "zero_hit_default_threshold"):
            delta = a[key] - b[key]
            print(f"| {key} | {b[key]} | {a[key]} | {delta:+.3f} |")
        print(f"| no_hit_probes | {b['no_hit_probes']} | {a['no_hit_probes']} | |")
        print(f"| unexpected_hits_probes | {b['unexpected_hits_probes']} | {a['unexpected_hits_probes']} | |")
        for key in ("img_link_char_ratio_avg", "image_refs_total", "image_paths_ok",
                    "image_paths_missing", "image_external", "queries_with_image_refs"):
            if key in b or key in a:
                bs, as_ = b.get(key, "-"), a.get(key, "-")
                delta = f"{as_ - bs:+.4f}" if isinstance(bs, (int, float)) and isinstance(as_, (int, float)) else ""
                print(f"| {key} | {bs} | {as_} | {delta} |")


if __name__ == "__main__":
    main()
