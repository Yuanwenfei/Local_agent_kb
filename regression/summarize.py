#!/usr/bin/env python3
"""回归结果速览：python regression/summarize.py [results/xxx.json]（默认 latest.json）"""
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "latest.json"
    path = arg if os.path.isabs(arg) else os.path.join(PROJECT_ROOT, "regression", "results", arg)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)

    print(f"== {os.path.basename(path)}  aggregate: {json.dumps(d['aggregate'], ensure_ascii=False)}")
    for q in d["per_query"]:
        top1 = q["top5"][0] if q["top5"] else {}
        tag = "NOHIT-EXPECTED" if q["expect_no_hit"] else ""
        print(f"{q['id']} mrr={q['mrr5']:.2f} grad={q['gradient']:.4f} "
              f"toc={int(q['top1_toc'])} doc%={q['max_doc_ratio']:.2f} | {q['query'][:44]} {tag}")
        show = q["top5"][:3] if (q["mrr5"] == 0 and not q["expect_no_hit"]) else q["top5"][:1]
        for h in show:
            src = os.path.basename(h["source"])
            print(f"    top{h['rank']} {h['score']:.3f} rel={int(h['relevant'])} "
                  f"{src[:44]} | {h['title'][:26]} | {h['heading'][:26]}")


if __name__ == "__main__":
    main()
