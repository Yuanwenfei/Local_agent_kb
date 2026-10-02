#!/usr/bin/env python3
"""中英术语表挖掘 / 体检（只扫 md-source，不连 Qdrant、不加载模型）

背景：kb_mcp_server.CN_EN_TERMS 是手写的「中文概念 → 英文标识符」表，纯中文提问时
服务端用它转写 sparse 通道（见 _keyword_query）。手敲有两个必然缺陷，本工具各治一个：

  1. **漏词**：语料里高频、表里没有的英文术语 → 默认模式列候选（按文件频次排序）
  2. **死词**：表里写的英文词在语料中根本不存在 → `--check` 反向体检，这类映射
     转写出去只会让 sparse 空跑，纯浪费
  3. **异形词**：表里写 `CurveConveyor`、语料实际是 `CurveConveyors` → `--forms`
     列出每个映射在语料里的真实形态与篇数（BM42 是子词哈希，单复数错位会掉 idf 权重）

候选只解决"缺哪些词"，中文标签仍须人工判断（脚本不做机器翻译，避免把
`Transfer` 翻成"迁移"这种和领域用法相反的词）。

用法:
  python312\\python.exe tools\\mine_cn_terms.py                    # 列缺失候选
  python312\\python.exe tools\\mine_cn_terms.py --top 80 --min-df 5
  python312\\python.exe tools\\mine_cn_terms.py --check            # 体检：死映射
  python312\\python.exe tools\\mine_cn_terms.py --forms            # 体检：形态是否对得上语料
  python312\\python.exe tools\\mine_cn_terms.py --json out.json    # 机读，便于比对
"""
import argparse
import ast
import io
import json
import os
import re
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_ROOT = os.path.join(PROJECT_ROOT, "md-source")
SERVER_PY = os.path.join(PROJECT_ROOT, "kb_mcp_server.py")

# Windows PowerShell 默认 GBK 代码页，输出 ⚠ / → 会直接 UnicodeEncodeError 崩掉。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# 候选来源：标题、文件名、代码片段（这三处是"领域术语"最密集的地方）
HEAD_RE = re.compile(r"^#{1,4}\s+(.*)$")
CODE_RE = re.compile(r"`([^`\n]{2,60})`")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.]*")

# 标识符判定：驼峰 / 带点下划线 / 全大写缩写。命中这三类的信息密度远高于普通单词。
def is_identifier(tok: str) -> bool:
    core = tok.strip("._")
    if len(core) < 3 or not core[0].isalpha():
        return False
    if any(sep in core for sep in "._") and any(c.isalpha() for c in core):
        return True
    if core.isupper():
        return True
    return len(core) > 3 and any(c.isupper() for c in core[1:]) and core[0].isupper()


STOP = {
    "the", "and", "for", "with", "this", "that", "from", "will", "can", "you",
    "your", "how", "what", "when", "are", "not", "use", "using", "all", "one",
    "true", "false", "null", "none", "string", "int32", "boolean", "object",
    "example", "note", "step", "click", "page", "tab", "box", "see", "also",
    "https", "http", "com", "www", "md", "png", "jpg", "html", "em3d",
    "tip", "wip", "ok", "id", "fig", "app", "e.g", "i.e",
}

# 噪声形态：文件扩展名 / 域名后缀 / 拉丁缩写。它们高频但不是可检索的领域术语。
NOISE_EXT = {
    "png", "jpg", "jpeg", "gif", "svg", "bmp", "webp", "md", "pdf", "php",
    "html", "xml", "json", "txt", "csv", "ini", "dll", "exe", "zip",
    "snapshot", "cs", "py", "xaml", "config",
}
DOMAINS = {"com", "net", "org", "cn", "edu", "gov", "io", "xyz"}
LATIN = {"ie", "eg", "etc", "vs", "etal", "nb", "ps", "cf"}


def is_noise(tok: str) -> bool:
    """front-matter 键、图片名、URL、拉丁缩写 → 不是术语候选。"""
    t = tok.strip("._-;:()[]{}\"'`").lower()
    if not t or t in LATIN or t.replace(".", "") in LATIN:
        return True
    segs = [s for s in t.split(".") if s]
    if len(segs) > 1 and (segs[-1] in NOISE_EXT or segs[-1] in DOMAINS):
        return True
    return False


def load_terms(path: str) -> dict:
    """用 AST 静态读出 CN_EN_TERMS，避免 import kb_mcp_server 触发模型加载。"""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "CN_EN_TERMS":
                    return {k.value: v.value for k, v in zip(node.value.keys, node.value.values)}
    raise SystemExit(f"未在 {path} 找到 CN_EN_TERMS 定义")


def iter_md(root: str):
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".md"):
                yield os.path.join(dirpath, fn)


def scan(root: str):
    """返回 (标识符/短语 → 出现的文件集合, [(相对路径, 原文, 小写原文)])。

    小写 blob 用于 --check 的子串判定，原文用于 --forms 的形态回显；
    全库 7.7MB，整批留在内存比反复读盘便宜。
    """
    df: dict = defaultdict(set)
    blobs = []
    for path in iter_md(root):
        try:
            text = io.open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        rel = os.path.relpath(path, os.path.dirname(root)).replace("\\", "/")
        blobs.append((rel, text, text.lower()))

        stem = re.sub(r"^\d+[\s._-]*", "", os.path.splitext(os.path.basename(path))[0])
        sources = [stem]
        sources += [m.group(1) for m in HEAD_RE.finditer(text)]
        sources += [m.group(1) for m in CODE_RE.finditer(text)]
        # 正文里的驼峰词（类名常只出现在散文里，不在标题中）
        hit = {t for t in WORD_RE.findall(text) if is_identifier(t) and not is_noise(t)}
        for s in sources:
            hit.update(t for t in WORD_RE.findall(s) if is_identifier(t) and not is_noise(t))

        for tok in hit:
            tok = tok.strip("._-")            # 合并 `Emulate3D.` / `Emulate3D` 这类尾部标点变体
            if tok and tok.lower() not in STOP and not is_noise(tok):
                df[tok].add(rel)

        # 短语候选：标题/文件名里的连续首字母大写词组（如 "Motor Speed"、"Photo Eye"）
        for s in sources:
            words = re.findall(r"[A-Z][a-zA-Z]*", re.sub(r"[`*]", " ", s))
            for n in (2, 3):
                for i in range(len(words) - n + 1):
                    gram = words[i:i + n]
                    if gram[0].islower() or any(w.lower() in STOP for w in gram):
                        continue
                    df[" ".join(gram)].add(rel)
    return df, blobs


def check_table(terms: dict, blobs):
    """反向体检：表内每个英文词在语料里是否真的存在（词边界匹配）。

    值允许是空格分隔的几个词（同义形态或短语），**逐词判定**：sparse 拿到的
    就是词袋，所以只要此中任一词全库 0 篇，这部分转写就是空跑。files 取最小值，
    让「死映射」判定仍然只看最弱的一个词。
    """
    rows = []
    for cn, en in sorted(terms.items(), key=lambda kv: kv[1]):
        per = []
        for tok in en.split():
            pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(tok.lower()) + r"(?![A-Za-z0-9_])")
            per.append(sum(1 for _rel, _raw, low in blobs if pat.search(low)))
        rows.append({"cn": cn, "en": en, "files": min(per) or 0,
                     "per_token": dict(zip(en.split(), per))})
    return rows


def realized_forms(value: str, blobs):
    """查单个英文词在语料里的真实形态（按词干前缀归并单复数、同类派生词）。

    返回 [(形态, 篇数)]，按篇数降序。调用方按词传入（多词值已拆开）。
    """
    if " " in value:                       # 防御：调用方应已拆词
        value = value.split()[0]
    stem = re.sub(r"y$", "", re.sub(r"s$", "", value, flags=re.I), flags=re.I)
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(stem) + r"[A-Za-z0-9_]*")
    per_form = defaultdict(set)
    for rel, raw, _low in blobs:
        for f in set(pat.findall(raw)):
            per_form[f].add(rel)
    return sorted(((f, len(s)) for f, s in per_form.items()), key=lambda x: (-x[1], x[0]))[:8]


def main():
    ap = argparse.ArgumentParser(description="中英术语表挖掘/体检")
    ap.add_argument("--top", type=int, default=50, help="候选条数（默认 50）")
    ap.add_argument("--min-df", type=int, default=3, help="候选最低文件频次（默认 3）")
    ap.add_argument("--check", action="store_true", help="体检：表内英文词在语料中是否存在")
    ap.add_argument("--forms", action="store_true", help="体检：表内写法与语料真实形态是否一致")
    ap.add_argument("--max-df-pct", type=float, default=50.0,
                    help="高于此语料百分比的候选视为样板噪声（front-matter 键、页脚链接等，默认 50）")
    ap.add_argument("--json", metavar="PATH", help="结果写入 JSON 文件")
    ap.add_argument("--grep", metavar="STEM", nargs="+",
                    help="查任意词干在语料里的真实形态（新增映射前先确认英文该写哪个）")
    ap.add_argument("--terms", metavar="PATH", default=SERVER_PY, help="CN_EN_TERMS 所在文件")
    args = ap.parse_args()

    terms = load_terms(args.terms)
    covered = {t.lower() for v in terms.values() for t in (v, v.lower())}
    covered |= {v.lower() for v in terms.values()}

    print(f"扫描 {DOCS_ROOT} …（表内现有映射 {len(terms)} 条）", file=sys.stderr)
    df, blobs = scan(DOCS_ROOT)
    print(f"共 {len(blobs)} 篇 md，候选术语 {len(df)} 个", file=sys.stderr)

    if args.grep:
        print("\n## 词干形态查询")
        for stem in args.grep:
            forms = realized_forms(stem, blobs)
            print(f"  {stem}: " + "、".join(f"{f}({n})" for f, n in forms))
        return 0

    if args.forms:
        print("\n## 映射形态核对（表内写法 vs 语料真实形态）")
        bad = []
        for cn, en in sorted(terms.items(), key=lambda kv: kv[0]):
            toks = en.split()
            parts, missing = [], []
            for tok in toks:
                forms = realized_forms(tok, blobs)
                if not any(f.lower() == tok.lower() and n > 0 for f, n in forms):
                    missing.append(tok)
                parts.append(tok + "：" + ("、".join(f"{f}({n})" for f, n in forms) or "—"))
            shown = "；".join(parts)
            mark = "" if not missing else "  ⚠ 表内写法不存在：" + "/".join(missing)
            if missing:
                bad.append((cn, en, shown))
            print(f"  {cn} → {en:<20} 语料：{shown}{mark}")
        print(f"\n小结：{len(terms)} 条映射，写法对不上语料的 {len(bad)} 条")
        for cn, en, shown in bad:
            print(f"  待改：{cn} → {en}（语料实际：{shown}）")
        if args.json:
            json.dump({cn: {t: realized_forms(t, blobs) for t in en.split()}
                       for cn, en in terms.items()},
                      io.open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"已写入 {args.json}")
        return 1 if bad else 0

    if args.check:
        rows = check_table(terms, blobs)
        dead = [r for r in rows if r["files"] == 0]
        rare = [r for r in rows if 0 < r["files"] <= 2]
        print("\n## 死映射（英文目标词在语料中 0 命中 → 转写后 sparse 必然空跑）")
        for r in dead:
            extra = "" if " " not in r["en"] else "（逐词：" + "、".join(
                f"{t}={n}" for t, n in r["per_token"].items()) + "）"
            print(f"  {r['cn']} → {r['en']}{extra}")
        print("\n## 稀有映射（≤2 篇命中，确认是否拼错/该换成真实标识符）")
        for r in rare:
            extra = "" if " " not in r["en"] else "（逐词：" + "、".join(
                f"{t}={n}" for t, n in r["per_token"].items()) + "）"
            print(f"  {r['cn']} → {r['en']}  ({r['files']} 篇){extra}")
        print(f"\n小结：{len(rows)} 条映射，死 {len(dead)} 条，稀有 {len(rare)} 条")
        if args.json:
            json.dump(rows, io.open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"已写入 {args.json}")
        return 1 if dead else 0

    # 候选：表内未覆盖 + 达频次 + 不是全库样板；标识符优先（信息量高于普通词组）
    max_df = int(len(blobs) * args.max_df_pct / 100.0)
    cands = []
    for tok, files in df.items():
        if len(files) < args.min_df or len(files) > max_df:
            continue
        if tok.lower() in covered:
            continue
        cands.append((tok, len(files), is_identifier(tok), sorted(files)[:2]))
    cands.sort(key=lambda x: (-x[2], -x[1], x[0]))
    cands = cands[: args.top]

    print(f"\n## 高频但表内缺失的术语候选（前 {len(cands)} 条；已排除出现在 >{args.max_df_pct:.0f}% 文件的样板词）")
    print("| 术语 | 篇数 | 类型 | 样例文件 |")
    print("|---|---|---|---|")
    for tok, n, ident, ex in cands:
        print(f"| `{tok}` | {n} | {'标识符' if ident else '词组'} | {'；'.join(os.path.basename(e) for e in ex)} |")
    print("\n提示：中文标签请人工定，脚本不做机器翻译（领域译名常与通用译名相反，如 Transfer=转移/积放）。")
    if args.json:
        json.dump([{"term": t, "df": n, "identifier": i, "examples": e} for t, n, i, e in cands],
                  io.open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
