#!/usr/bin/env python3
"""
站点外链图片落地（P4 补图通道）——把 md-source 里指向 store.sim3d.com 的**外链图**
下载到本地 assets 并改写为相对路径，使 MCP 返图（top1 缩略图）能真正看到这些图。

背景（2026-10-01 实测）：
- 全库 959 个片段 / 324 篇文档含外链图，去重 1,473 个 URL，其中 **1,435 个是
  `store.sim3d.com/docfetch.php?id=…&file=images/Image0.png`**（爬取时被写成占位形式，
  匿名请求 404），另有 36 个 YouTube（非图片）；
- 站点有三个**纯 HTTP、无需登录**的接口（已实测）：
  - `helpconsole.php?action=view&format=raw&j=<project>&p=<page>` → 该页**原始 markdown**，
    内含真实相对图片名 `images/<page>_image_<k>_Image0.png`（权威来源，替代臆测推导）；
  - `docfetch.php?id=<project>/<page>&file=<相对路径>` → 图片本体（HTTP 200）；
  - `helpindex.php / helpcontents.php` → 站点目录索引。
- 抽样 20 篇：raw 内图片引用 95 条、可下载 91 条（**96%**）。

用法:
  python312\\python.exe tools\\fetch_site_images.py --all --dry-run    # 预演：只统计+报告
  python312\\python.exe tools\\fetch_site_images.py --all --apply      # 真下：下载 + 改写 md（自动备份）
  python312\\python.exe tools\\fetch_site_images.py <文件> --apply
  python312\\python.exe tools\\fetch_site_images.py --all --restore    # 从备份还原 md（图片文件保留）

备份：backup\\site-images-20261001\\<相对路径>（仅 md 源文件；下载的图片本身不备份）
元数据：顺带把 raw 的 `date`/`lastmod` 落到 backup\\site_raw_meta.json（供 P4-1 回填真实文档日期）
"""
import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from urllib.parse import quote, unquote

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_SOURCE = os.path.join(PROJECT_ROOT, "md-source")
BACKUP_ROOT = os.path.join(PROJECT_ROOT, "backup", "site-images-20261001")
RAW_META_FILE = os.path.join(PROJECT_ROOT, "backup", "site_raw_meta.json")

SITE = "https://store.sim3d.com"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TIMEOUT = 30

sys.path.insert(0, PROJECT_ROOT)
from kb_schema import resolve_image_paths  # noqa: E402
from document_parsers import parse_front_matter  # noqa: E402

# raw 里的图片引用：`src="images/x"` / `href=...` / `![](images/x)`
REF_RE = re.compile(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']|!\[[^\]]*\]\(([^)]+)\)')
FM_RE = re.compile(r'\A---[ \t]*\r?\n(?P<body>.*?)\r?\n---[ \t]*(?:\r?\n)?', re.S)
IMG_EXT_RE = re.compile(r'\.(png|jpe?g|gif|bmp|webp|svg)$', re.I)
# 我们 md 里那种占位外链：docfetch.php?id=<j>/<page>_image_<k>&file=images/Image0.png
PLACEHOLDER_RE = re.compile(
    r'https?://store\.sim3d\.com/docfetch\.php\?id=[^&"\'\s]*?_image_(?P<k>\d+)&(?:amp;)?file=[^"\'\s\)]*')


def http_get(url: str, binary=False, retries=2):
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=TIMEOUT) as r:
                return r.status, (r.read() if binary else r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            return e.code, (b"" if binary else "")
        except Exception:
            if i == retries:
                return 0, (b"" if binary else "")
    return 0, (b"" if binary else "")


def rail(*parts) -> str:
    return os.path.join(*parts).replace("\\", "/")


def vault_of(path: str) -> str:
    rel = os.path.relpath(path, MD_SOURCE).split(os.sep)
    return rel[0] if len(rel) > 1 else ""


def raw_refs(text: str) -> list[str]:
    out = []
    for a, b in REF_RE.findall(text):
        r = (a or b).strip().strip('"').strip("'")
        if not r or r.lower().startswith(("http://", "https://", "data:", "#", "mailto:")):
            continue
        if not IMG_EXT_RE.search(r):
            continue
        r = r.lstrip("./")
        if r not in out:
            out.append(r)
    return out


def real_doc_date(raw_meta: dict) -> str:
    """raw 的 `date`(YYYY-MM-DD) 优先；缺失时解析 `lastmod`(DD/MM/YYYY)"""
    d = str(raw_meta.get("date") or "").strip()
    m = re.match(r'^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$', d)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    lm = str(raw_meta.get("lastmod") or "").strip()
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})', lm)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        # `24/10/2022` 证明站点是「日/月/年」；日≤12 时无法区分，跳过避免写错
        if a > 12 or b > 12:
            day, mon = (a, b) if a > 12 else (b, a)
            return f"{m.group(3)}-{mon:02d}-{day:02d}"
    return ""


def set_doc_date(text: str, iso: str) -> tuple[str, bool]:
    """把 front-matter 里的 doc_date 换成真实日期（无则插入）"""
    m = FM_RE.match(text)
    if not m or not iso:
        return text, False
    head, tail = m.group(0), text[m.end():]
    if re.search(r'^doc_date\s*:', head, re.M):
        if re.search(rf'^doc_date\s*:\s*{re.escape(iso)}\s*$', head, re.M):
            return text, False
        head = re.sub(r'^doc_date\s*:.*$', f'doc_date: {iso}', head, count=1, flags=re.M)
    else:
        eol = "\r\n" if "\r\n" in head else "\n"
        head = head.replace(f"---{eol}", f"---{eol}doc_date: {iso}{eol}", 1) \
            if eol in head else head
    return head + tail, True


def project_page(meta: dict) -> tuple[str, str]:
    """从 front-matter 推 (项目, 页面)：
    - `source-url`（`https://store.sim3d.com/<project>/<page>`）最完整，优先；
    - 退化用 `source-id`；`Emulate3D-Tutorials` 语料的 source-id 不带项目前缀，
      其 source-url 形如 `/tutorials/<page>`，所以仍以 source-url 为准。"""
    su = str(meta.get("source-url") or "").strip()
    if su:
        m = re.match(r'https?://[^/]+/(.+)$', su)
        if m:
            parts = [x for x in m.group(1).split("/") if x]
            if len(parts) >= 2:
                return parts[0], "/".join(parts[1:])
    sid = str(meta.get("source-id") or "")
    return (sid.split("/", 1) if "/" in sid else ("", ""))


def process(path: str, apply: bool, force: bool, meta_out: dict, set_date: bool = False) -> dict:
    text = open(path, encoding="utf-8", errors="replace").read()
    meta, _ = parse_front_matter(text)
    j, p = project_page(meta)
    if not j or not p:
        return {"skip": "无 source-url/source-id（非站点文档）"}
    sid = f"{j}/{p}"

    st, body = http_get(f"{SITE}/helpconsole.php?action=view&format=raw&lang="
                        f"&j={j}&p={quote(p, safe='')}")
    if st != 200 or not body:
        return {"skip": f"raw 接口 HTTP {st}"}
    refs = raw_refs(body)

    rel_meta = os.path.relpath(path, PROJECT_ROOT)
    rm = parse_front_matter(body)[0]
    raw_meta = {k: str(rm.get(k, "")) for k in
                ("date", "lastmod", "creatordisplayname", "lastmodifierdisplayname")}
    if raw_meta.get("date") or raw_meta.get("lastmod"):
        meta_out[rel_meta] = raw_meta

    if not refs and not set_date:
        return {"skip": "raw 内无图片引用"}

    vault = vault_of(path)
    assets_dir = os.path.join(MD_SOURCE, vault, "assets", p.replace("/", "_"), "images")
    md_dir = os.path.dirname(path)

    downloaded, failed, local_of = 0, [], {}
    for ref_raw in refs:
        ref = unquote(ref_raw)   # raw 里可能已 URL 编码（如 `Animation%202.gif`），先解码再请求
        name = os.path.basename(ref)
        local = os.path.join(assets_dir, name)
        url = f"{SITE}/docfetch.php?id={quote(sid, safe='/')}&file={quote(ref, safe='/')}"
        if os.path.exists(local) and not force:
            local_of[ref_raw] = local     # 键保留 raw 原始写法，便于在 md 里原地匹配
            continue
        st2, data = http_get(url, binary=True)
        if st2 == 200 and data:
            if apply:                       # 预演只验证可达，不落盘
                os.makedirs(assets_dir, exist_ok=True)
                with open(local, "wb") as f:
                    f.write(data)
            downloaded += 1
            local_of[ref_raw] = local
        else:
            failed.append((ref_raw, st2))

    # ---------- 改写 md ----------
    # local_of: raw 里的相对引用（形如 images/x.png）→ 本地绝对路径
    def rel_link(ref: str) -> str:
        return rail(os.path.relpath(local_of[ref], md_dir))

    # `_image_<k>` → raw 真实引用（供 R1 映射）；命名对不上时按出现顺序兜底
    by_k: dict[str, str] = {}
    for ref in local_of:
        m = re.search(r'_image_(\d+)(?:_Image\d+)?\.[A-Za-z]+$', os.path.basename(ref))
        if m:
            by_k.setdefault(m.group(1), ref)
    order = list(local_of)

    new_text = text
    # R1：占位外链 `docfetch.php?id=…_image_<k>&file=images/Image0.png` → 本地相对路径
    def r1(m):
        k = m.group("k")
        ref = by_k.get(k)
        if not ref:  # 命名对不上 → 按 `_image_<k>` 的第 k 张兜底
            idx = int(k) - 1
            ref = order[idx] if 0 <= idx < len(order) else None
        return rel_link(ref) if ref else m.group(0)

    n_r1 = len(PLACEHOLDER_RE.findall(new_text))
    new_text = PLACEHOLDER_RE.sub(r1, new_text)
    replaced_r1 = sum(1 for ref in local_of if rel_link(ref) in new_text)

    # R2：raw 里的相对引用（`images/<name>`，含 src= 与 markdown 两种写法）→ 本地相对路径
    n_r2 = 0
    for ref in list(local_of):
        if ref not in new_text:
            continue
        link = rel_link(ref)
        before = new_text
        new_text = (new_text.replace(f'"{ref}"', f'"{link}"')
                            .replace(f"'{ref}'", f"'{link}'")
                            .replace(f'({ref})', f'({link})'))
        if new_text != before:
            n_r2 += 1

    left = len(PLACEHOLDER_RE.findall(new_text))

    # R3（可选）：用 raw 的真实日期替换 doc_date
    date_changed, iso = False, ""
    if set_date:
        iso = real_doc_date(raw_meta)
        if iso:
            new_text, date_changed = set_doc_date(new_text, iso)

    changed = new_text != text
    if changed and apply:
        backup = os.path.join(BACKUP_ROOT, os.path.relpath(path, PROJECT_ROOT))
        if not os.path.exists(backup):
            os.makedirs(os.path.dirname(backup), exist_ok=True)
            shutil.copy2(path, backup)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new_text)

    return {"ok": True, "imgs": len(refs), "downloaded": downloaded, "failed": failed,
            "r1": n_r1, "r1_replaced": replaced_r1, "r2": n_r2, "left": left,
            "changed": changed, "date_changed": date_changed, "iso": iso}


def collect(patterns, all_flag) -> list[str]:
    files = []
    if patterns:
        for p in patterns:
            ap = os.path.abspath(p)
            if os.path.isfile(ap):
                files.append(ap)
            elif os.path.isdir(ap):
                for root, dirs, fs in os.walk(ap):
                    dirs[:] = [d for d in dirs if d.lower() != "assets"]
                    files += [os.path.join(root, f) for f in fs if f.endswith(".md")]
    if all_flag:
        qdrant = None
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from kb_mcp_server import get_qdrant, COLLECTION_NAME
            qdrant, coll = get_qdrant(), COLLECTION_NAME
        except Exception as e:
            print(f"  [warn] 连不上 Qdrant（{e}），改为全量扫描 md-source")
        if qdrant is None:
            for root, dirs, fs in os.walk(MD_SOURCE):
                dirs[:] = [d for d in dirs if d.lower() != "assets"]
                files += [os.path.join(root, f) for f in fs if f.endswith(".md")]
        else:
            pts, off = [], None
            while True:
                batch, off = qdrant.scroll(collection_name=coll, limit=3000, offset=off,
                                           with_payload=["source", "image_refs"],
                                           with_vectors=False)
                pts.extend(batch)
                if off is None:
                    break
            for pt in pts:
                src = pt.payload.get("source", "")
                refs = pt.payload.get("image_refs") or []
                if not refs:
                    continue
                paths = resolve_image_paths(src, refs)
                if any(x.lower().startswith(("http://", "https://")) for x in paths):
                    files.append(src)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description="站点外链图片落地（下载 + 改写 md）")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true", help="扫描所有含外链图的文档")
    ap.add_argument("--dry-run", action="store_true", help="预演，不落盘（默认）")
    ap.add_argument("--apply", action="store_true", help="真下 + 改写（自动备份 md）")
    ap.add_argument("--force", action="store_true", help="已存在的本地图片也重下")
    ap.add_argument("--set-date", action="store_true",
                    help="顺带用 raw 的真实日期替换 front-matter 的 doc_date")
    ap.add_argument("--jobs", type=int, default=6, help="并发文档数（默认 6）")
    ap.add_argument("--restore", action="store_true", help="从备份还原 md 文件")
    args = ap.parse_args()

    if not args.files and not args.all:
        ap.error("请给文件路径，或使用 --all")

    files = collect(args.files, args.all)
    if not files:
        ap.error("没找到含外链图的文档")

    if args.restore:
        n = 0
        for path in files:
            backup = os.path.join(BACKUP_ROOT, os.path.relpath(path, PROJECT_ROOT))
            if os.path.exists(backup):
                shutil.copy2(backup, path)
                n += 1
        print(f"已从备份还原 {n} 个 md 文件")
        return

    apply = bool(args.apply) and not args.dry_run
    meta_out: dict = {}
    rows, skipped = [], []
    tot_img = tot_dl = tot_fail = 0

    def run_one(path):
        try:
            return path, process(path, apply, args.force, meta_out, args.set_date)
        except Exception as e:
            return path, {"skip": f"异常 {e}"}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, args.jobs)) as ex:
        for path, r in ex.map(run_one, files):
            if r.get("skip"):
                skipped.append((os.path.basename(path), r["skip"]))
                continue
            rows.append((os.path.relpath(path, PROJECT_ROOT), r))
            tot_img += r["imgs"]
            tot_dl += r["downloaded"]
            tot_fail += len(r["failed"])

    mode = "真下+改写" if apply else "预演（不落盘）"
    print(f"## fetch_site_images（{mode}"
          f"{'｜含 doc_date 回填' if args.set_date else ''}）")
    print(f"\n- 文档：**{len(rows)}** 篇处理成功 / {len(skipped)} 篇跳过")
    print(f"- raw 内图片引用：**{tot_img}** 条，可下载 **{tot_dl}** 条"
          f"（{round(100 * tot_dl / max(tot_img, 1))}%），失败 {tot_fail} 条")
    ch = sum(1 for _p, r in rows if r.get("changed"))
    print(f"- md 需要改写：**{ch}** 篇；改写后仍残留占位外链："
          f"{sum(r['left'] for _p, r in rows)} 条")
    if args.set_date:
        dc = sum(1 for _p, r in rows if r.get("date_changed"))
        iso = sum(1 for _p, r in rows if r.get("iso"))
        print(f"- doc_date 回填：**{dc}** 篇；raw 有可用真实日期 **{iso}** 篇")
    if meta_out:
        print(f"- raw 真实日期已收集：**{len(meta_out)}** 篇 → `backup/site_raw_meta.json`")
    if skipped:
        print("\n### 跳过（前 8 条）")
        for name, why in skipped[:8]:
            print(f"- {name}: {why}")
    fails = [(p, ref, st) for p, r in rows for ref, st in r["failed"]]
    if fails:
        print(f"\n### 下载失败样例（共 {len(fails)} 条，前 8 条）")
        for rel, ref, st in fails[:8]:
            print(f"- HTTP {st}  {os.path.basename(rel)} → {ref}")
    print("\n### 样例（前 6 篇）")
    for rel, r in rows[:6]:
        print(f"- {rel}: 图{r['imgs']} 下载{r['downloaded']} 失败{len(r['failed'])} "
              f"改写(R1 {r['r1']}/R2 {r['r2']}) 残留{r['left']}"
              + (f" doc_date→{r['iso']}" if r.get("date_changed") else ""))

    if apply:
        os.makedirs(os.path.dirname(RAW_META_FILE), exist_ok=True)
        old = {}
        if os.path.exists(RAW_META_FILE):
            old = json.load(open(RAW_META_FILE, encoding="utf-8"))
        old.update(meta_out)
        with open(RAW_META_FILE, "w", encoding="utf-8") as f:
            json.dump(old, f, ensure_ascii=False, indent=1)
        print(f"\n- md 备份：`{os.path.relpath(BACKUP_ROOT, PROJECT_ROOT)}`"
              f"（还原：`--all --restore`）")
        print("- 下一步：重嵌这批文档（`index_docs.py md-source` 增量，或等全库重跑），"
              "之后 MCP 返图即可显示这些图")


if __name__ == "__main__":
    sys.exit(main())
