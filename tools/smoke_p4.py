#!/usr/bin/env python3
"""P4-3 图注治理冒烟与验收（kb_caption / T9）。

分两段：
- 只读段（默认）：工具注册、候选口径（image_refs 非空）、报告内容、
  图注抽取向导单测（explicit/alt/context/filename 四级 + 图注进嵌入文本）；
- 写入段（--commit）：临时文档带显式图注入库 → 断言图注进了 payload 与嵌入文本
  → `search_tech_kb` 精确检索图注文字可命中（T9）→ 回滚（删点+删文件+清状态）。

用法：
    python tools\\smoke_p4.py             # 只读冒烟
    python tools\\smoke_p4.py --commit    # 含入库/检索的 T9 验收（跑完回滚）
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PASS, FAIL = [], []
TOKEN = "P4CAPTIONTOKEN9Q2"


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _text(res) -> str:
    return "\n".join(c.text for c in res if getattr(c, "type", "") == "text")


# ---------------- 只读段 ----------------

def parser_checks():
    """图注抽取向导单测（纯解析层，不碰库）

    每个来源写**独立文件**：parse_md 有「短片段合并」规则，多节写在一个文件里
    会被并成一块，就测不出各自的来源了（首轮实测即踩到：4 节并成 1 块）。
    """
    print("\n== 解析层图注取向导 ==")
    import tempfile
    from document_parsers import parse_md

    filler = ("本段为保证切片不被短片段合并规则吞掉的说明文字，描述该小节在"
              "图注抽取冒烟中的用途，确保图注来源能被逐级独立验证。")
    cases = {
        "explicit": f"""---
title: 显式图注样例
---

## 显式图注小节

{filler}

![已有alt但先被显式标记压过](imgs/a.png)

图注：ACR 层级 Bot → Lift → ACRLift → PRH1 → Gripper
""",
        "alt": f"""---
title: alt 图注样例
---

## alt 小节

{filler}

![Conveyor Indicator Arrows](imgs/b.png)
""",
        "context": f"""---
title: 上下文图注样例
---

## 上下文小节

输送线的方向由箭头指示，方向反了会撞车。{filler}

![](imgs/c.png)
""",
        "filename": f"""---
title: 标题兜底样例
---

![](images/Conveyor%20Direction%20Arrows.png)

{filler}
""",
        "heading": f"""---
title: 章节标题兜底样例
---

![](img.png)

{filler}
""",
    }
    got, all_chunks = {}, []
    with tempfile.TemporaryDirectory() as td:
        for tag, text in cases.items():
            fp = os.path.join(td, f"图注冒烟样例_{tag}.md")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(text)
            chunks = parse_md(fp)
            all_chunks.extend(chunks)
            hit = [c for c in chunks if c.get("image_refs")]
            got[tag] = hit[0] if hit else {}
    print("  来源实测：" + "；".join(
        f"{k}→{v.get('image_caption_source', '无')}" for k, v in got.items()))

    v = got["explicit"]
    check("explicit：图片下方 `图注：` 压过 alt（最高优先）",
          v.get("image_caption_source") == "explicit" and "ACR 层级" in v.get("image_captions", ""),
          f"{v.get('image_caption_source')} / {v.get('image_captions', '')[:40]}")
    v = got["alt"]
    check("alt：作者 alt 次优先",
          v.get("image_caption_source") == "alt" and "Conveyor" in v.get("image_captions", ""),
          f"{v.get('image_caption_source')} / {v.get('image_captions', '')[:40]}")
    v = got["context"]
    check("context：图片上方紧邻正文兜底",
          v.get("image_caption_source") == "context" and "箭头" in v.get("image_captions", ""),
          f"{v.get('image_caption_source')} / {v.get('image_captions', '')[:40]}")
    v = got["filename"]
    check("filename：无正文无 alt 时用图片文件名兜底",
          v.get("image_caption_source") == "filename" and "Conveyor Direction Arrows" in v.get("image_captions", ""),
          f"{v.get('image_caption_source')} / {v.get('image_captions', '')[:40]}")
    v = got["heading"]
    check("heading：图片名无信息量时退回文档标题",
          v.get("image_caption_source") == "heading" and "章节标题兜底样例" in v.get("image_captions", ""),
          f"{v.get('image_caption_source')} / {v.get('image_captions', '')[:40]}")

    n_img = sum(1 for c in all_chunks if c.get("image_refs"))
    n_marked = sum(1 for c in all_chunks if c.get("image_refs") and "[图注]" in c.get("text", ""))
    check("图注进入嵌入文本（[图注] 行）", n_img > 0 and n_marked == n_img, f"{n_marked}/{n_img} 切片")
    check("图注已并入 search_text（精确通道可命中）",
          any("[图注]" in (c.get("search_text") or "") for c in all_chunks))


async def readonly_checks(server):
    print("\n== 只读冒烟 ==")

    tools = await server.list_tools()
    names = {t.name for t in tools}
    check("工具总数 = 11", len(names) == 11, f"实际 {len(names)}：{sorted(names)}")
    check("kb_caption 已注册", "kb_caption" in names)

    res = await server.call_tool("kb_stats", {})
    txt = _text(res)
    check("kb_stats 治理指标改用 image_refs 口径",
          "含图片段（`image_refs` 非空）" in txt and "已生成图注" in txt)

    res = await server.call_tool("kb_caption", {"mode": "summary"})
    check("mode=summary 明确未实现（二期 VLM）", "二期" in _text(res) and "VLM" in _text(res))

    res = await server.call_tool("kb_caption", {"dry_run": True, "limit": 5})
    txt = _text(res)
    check("kb_caption 候选口径 = image_refs 非空（P4-3①）",
          "`image_refs` **非空**" in txt and "候选切片" in txt)
    check("kb_caption 输出「未参与嵌入」告警", "参与嵌入" in txt or "已进向量" in txt)
    print("  ---")
    for ln in txt.splitlines()[:14]:
        print("  " + ln)
    print("  ---")

    # source 限定
    docs = _text(await server.call_tool("kb_list_docs", {"limit": 1}))
    src = ""
    for line in docs.splitlines():
        if line.startswith("| 1 |"):
            src = line.split("|")[2].strip()
            break
    if src:
        res = await server.call_tool("kb_caption", {"source": src, "dry_run": True})
        t2 = _text(res)
        check("kb_caption(source=…) 局部限定生效", "source 限定" in t2, t2.splitlines()[0] if t2 else "")
    else:
        check("取到样例文档", False, "kb_list_docs 无输出")


# ---------------- 写入段（T9） ----------------

async def commit_checks(server, keep: bool):
    print("\n== 写入段：T9 图注检索 ==")
    import index_docs
    import kb_templates as kbt

    md_path = os.path.join(kbt.INBOX_DIR, "_p4_caption",
                           "P4图注冒烟-ACR层级.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    # 关键：图注必须来自**正文之外**的信息源（此处是图片 alt），否则「图注参与嵌入」
    # 无法与「正文本来就有这些字」区分开——首轮设计的 `图注：` 写法就踩了这个坑
    # （显式图注本身也是正文行，检索命中的是正文而不是图注）。
    md = f"""---
title: P4 图注冒烟-ACR层级
kind: defect_log
canonical: true
---

## ACR 层级链

下图给出 ACR 各层级部件的连接关系。

![{TOKEN} ACRLift 层级链 Bot-Gripper](imgs/acr.png)

## 备注

该链路用于排查取货点注册失败。
"""
    added = False
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        info = server._index_written_file(md_path)
        added = True
        check("临时图注文档入库并向量化", "片段" in info, info)

        pts, _ = server.get_qdrant().scroll(
            collection_name=server.COLLECTION_NAME,
            scroll_filter=server.qm.Filter(must=[server.qm.FieldCondition(
                key="source", match=server.qm.MatchValue(value=os.path.abspath(md_path)))]),
            limit=10, with_payload=True, with_vectors=False)
        check("入库切片带 image_refs", bool(pts) and all(p.payload.get("image_refs") for p in pts))
        check("image_caption_source=alt（图注来自正文之外的作者 alt）",
              any(p.payload.get("image_caption_source") == "alt" for p in pts),
              str([p.payload.get("image_caption_source") for p in pts]))
        check("标识串不在正文里（证明命中只能来自图注）",
              TOKEN not in " ".join(ln for ln in (pts[0].payload.get("text") or "").splitlines()
                                    if "[图注]" not in ln))
        check("图注进入嵌入文本（text 含 [图注] 与标识串）",
              any(TOKEN in (p.payload.get("text") or "") and "[图注]" in (p.payload.get("text") or "")
                  for p in pts))

        # T9：图注文字可被检索命中（精确通道；图注里的标识串在正文中不存在）
        res = await server.call_tool("search_tech_kb", {"query": TOKEN, "mode": "exact", "top_k": 3})
        txt = _text(res)
        check("T9 图注文字可命中（exact 通道）",
              os.path.basename(md_path) in txt or TOKEN in txt,
              txt.splitlines()[0] if txt else "")

        # 返回体同时给出图注提示（人工可判读是不是真图注）
        res = await server.call_tool("search_tech_kb", {"query": "ACR 层级链", "top_k": 3})
        txt = _text(res)
        check("检索返回体展示图注行", "图注" in txt, txt.splitlines()[0] if txt else "")
        check("返图降级不报错（图片文件不存在）", "知识库查询错误" not in txt)

        # ---- 复现「存量旧图注只在 payload、未参与嵌入」的 2.1 状态，验证 reindex 覆盖范围 ----
        # 注意：精确通道的门是 search_text（MatchText 预筛 + 整串判定），text 只是兜底，
        # 所以两处都要抹掉图注行才算真的复现存量态（向量无法用 set_payload 回溯剥离，
        # 因此这个复现只对「精确通道」有效——而这正是 T9 的判定通道）。
        from kb_schema import normalize_for_search
        stale_text = "\n".join(ln for ln in (pts[0].payload.get("text") or "").splitlines()
                               if "[图注]" not in ln)
        server.get_qdrant().set_payload(
            collection_name=server.COLLECTION_NAME,
            payload={"text": stale_text, "search_text": normalize_for_search(stale_text),
                     "image_captions": "ACR 层级（旧图注）",
                     "image_caption_source": "context"},
            points=[pts[0].id])
        server._invalidate_caches()
        res = await server.call_tool("search_tech_kb", {"query": TOKEN, "mode": "exact", "top_k": 3})
        p2, _ = server.get_qdrant().scroll(
            collection_name=server.COLLECTION_NAME,
            scroll_filter=server.qm.Filter(must=[server.qm.FieldCondition(
                key="source", match=server.qm.MatchValue(value=os.path.abspath(md_path)))]),
            limit=1, with_payload=True, with_vectors=False)
        pl = (p2[0].payload if p2 else {}) or {}
        diag = (f"写入前stale含标识={TOKEN in stale_text}｜stale含图注行={'[图注]' in stale_text}｜"
                f"text含标识={TOKEN in (pl.get('text') or '')}｜"
                f"text==stale={pl.get('text') == stale_text}｜"
                f"search_text含标识={TOKEN in (pl.get('search_text') or '')}｜"
                f"命中={_text(res).splitlines()[0] if _text(res) else ''}")
        diag += "｜残行=" + "⏎".join(
            ln[:80] for ln in stale_text.splitlines() if TOKEN in ln)[:160]
        check("复现存量缺口：图注未参与嵌入时检索不到",
              "返回:0" in _text(res) or "完全一致" in _text(res), diag)

        # reindex 必须覆盖「有旧图注但未进向量」的文档（P4-3 修正：原实现漏 1,084 篇）
        res = await server.call_tool("kb_caption", {
            "source": os.path.abspath(md_path), "dry_run": False, "reindex": True})
        txt = _text(res)
        check("kb_caption(reindex) 覆盖「旧图注未参与嵌入」文档",
              "存量旧图注未参与嵌入 1 篇" in txt and "重嵌成功 1 篇" in txt,
              next((ln for ln in txt.splitlines() if "增量重嵌" in ln), txt[:80]))

        res = await server.call_tool("search_tech_kb", {"query": TOKEN, "mode": "exact", "top_k": 3})
        txt = _text(res)
        check("reindex 后图注恢复可检索（T9 对存量生效路径）",
              os.path.basename(md_path) in txt or TOKEN in txt,
              txt.splitlines()[0] if txt else "")
    finally:
        print("  回滚：")
        if added:
            try:
                index_docs.delete_file_points(os.path.abspath(md_path))
                print("    - 已删点")
            except Exception as e:
                print(f"    - 删点失败: {e}")
            state = index_docs._load_json_state(index_docs.INGEST_STATE_FILE)
            state.pop(os.path.abspath(md_path), None)
            index_docs._save_json_state(index_docs.INGEST_STATE_FILE, state)
        if not keep and os.path.exists(md_path):
            os.remove(md_path)
            print("    - 已删文件")
        try:
            os.rmdir(os.path.dirname(md_path))
        except OSError:
            pass


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="执行写入段（默认回滚）")
    ap.add_argument("--keep", action="store_true", help="保留样例文件")
    args = ap.parse_args()

    parser_checks()

    import kb_mcp_server as server
    await readonly_checks(server)
    if args.commit:
        await commit_checks(server, args.keep)
    else:
        print("\n（未加 --commit：跳过写入段 T9）")

    print(f"\n===== 冒烟结果：{len(PASS)} PASS / {len(FAIL)} FAIL =====")
    for f in FAIL:
        print(f"  FAIL: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
