#!/usr/bin/env python3
"""P3 冒烟与验收（T5 / T6 / T7 / T10 / T13）。

分两阶段：
- 只读段（默认执行）：工具注册、kb_dupes、kb_outline/kb_get_doc、返图——不写库不落盘；
- 写入段（--commit 才执行）：kb_note → kb_ingest(dry_run) → kb_ingest(正式) →
  kb_error_lookup → 新文档 outline/get_doc → 默认**回滚**（删点+删文件），
  加 --keep 可保留样例知识供人工查看。

用法：
    python tools/smoke_p3.py              # 只读冒烟
    python tools/smoke_p3.py --commit     # 全链路（跑完回滚）
    python tools/smoke_p3.py --commit --keep
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _text(res) -> str:
    return "\n".join(c.text for c in res if getattr(c, "type", "") == "text")


def _images(res) -> list:
    return [c for c in res if getattr(c, "type", "") == "image"]


async def readonly_checks(server):
    print("\n== 只读冒烟 ==")

    tools = await server.list_tools()
    names = {t.name for t in tools}
    expected = {"search_tech_kb", "kb_stats", "kb_list_docs", "kb_meta_lint",
                "kb_get_doc", "kb_outline", "kb_error_lookup", "kb_note",
                "kb_ingest", "kb_dupes"}
    check("工具总数 = 11（P4 新增 kb_caption）", len(names) == 11, f"实际 {len(names)}：{sorted(names)}")
    check("P3 六个工具全部注册", expected <= names, f"缺 {sorted(expected - names)}")

    # kb_dupes（T12/D-7：已知 KNOWN 副本对）
    res = await server.call_tool("kb_dupes", {"mode": "name", "limit": 5})
    txt = _text(res)
    check("kb_dupes(name) 可执行", "副本巡检" in txt or "未发现副本" in txt)
    check("kb_dupes 扫描含 kimi code workbentch（D-6）", "workbentch" in txt)

    # 取一篇真实文档做 get_doc / outline
    res = await server.call_tool("kb_list_docs", {"limit": 5})
    src = ""
    for line in _text(res).splitlines():
        if line.startswith("| 1 |"):
            src = line.split("|")[2].strip()
            break
    if not src:
        check("取到样例文档", False, "kb_list_docs 无输出")
        return
    check("取到样例文档", True, src)

    res = await server.call_tool("kb_outline", {"source": src})
    txt = _text(res)
    check("kb_outline 有章节树", "章节树" in txt and "片段" in txt, txt.splitlines()[0] if txt else "")

    res = await server.call_tool("kb_get_doc", {"source": src, "budget": 1200})
    txt = _text(res)
    check("kb_get_doc 返回正文", len(txt) > 400 and "路径:" in txt, f"{len(txt)} 字符")

    # 预算截断
    res = await server.call_tool("kb_get_doc", {"source": src, "budget": 500})
    check("kb_get_doc 预算截断", "字符预算" in _text(res) or len(_text(res)) < 1200)

    # 章节过滤
    res = await server.call_tool("kb_get_doc", {"source": src, "heading": "概述"})
    check("kb_get_doc 章节过滤生效",
          "章节过滤" in _text(res) or "未找到章节" in _text(res))

    # 不存在文档
    res = await server.call_tool("kb_get_doc", {"source": "__no_such_doc__.md"})
    check("不存在文档友好报错", "未找到文档" in _text(res))

    # 返图（T13）：找一篇含图文档
    img_res = None
    for q in ("AMR 层级 结构图", "Emulate3D 教程 示意图", "PLC 接线 图"):
        res = await server.call_tool("search_tech_kb",
                                     {"query": q, "top_k": 3, "attach_image": True})
        if _images(res):
            img_res = res
            break
    if img_res:
        check("T13 返图：top1 附 image content", True,
              f"{len(_images(img_res))} 张，提示语含'读图'")
        check("T13 提示语引导读图", "读图作答" in _text(img_res))
    else:
        check("T13 返图：top1 附 image content", False,
              "三组含图查询均未返回 image content（可能 top1 恰好为纯文本片段）")

    # 纯文本降级（attach_image=false 不报错）
    res = await server.call_tool("search_tech_kb",
                                 {"query": "AMR 层级 结构图", "top_k": 3, "attach_image": False})
    check("T13 降级：attach_image=false 不返回图且不报错",
          not _images(res) and "知识库查询错误" not in _text(res))


async def commit_checks(server, keep: bool):
    print("\n== 写入段（kb_note → kb_ingest → kb_error_lookup） ==")
    import kb_templates as kbt
    import index_docs

    topic = "冒烟P3错误串登记-AMR控制器缺子件"
    error_text = "does not have an AMR Controller as child"
    draft_path = kbt.target_path("draft", topic)
    added_sources = []

    try:
        # --- T10 步骤 1：kb_note 落草稿 ---
        res = await server.call_tool("kb_note", {
            "topic": topic, "evidence": f"报错原文：{error_text}",
            "meta": {"framework": "2.4.0.0", "model": "AMR"}})
        txt = _text(res)
        check("T10-1 kb_note 落盘规范草稿", os.path.exists(draft_path) and "草稿已落盘" in txt,
              draft_path)

        # --- T10 步骤 2：kb_ingest(dry_run) 出校验报告 ---
        res = await server.call_tool("kb_ingest", {
            "kind": "error_faq", "path": draft_path, "meta": {"title": topic},
            "dry_run": True})
        txt = _text(res)
        check("T10-2 kb_ingest(dry_run) 出校验报告",
              "校验报告" in txt and "内容哈希" in txt, txt.splitlines()[2] if txt else "")
        check("T10 dry_run 未落库", not os.path.exists(kbt.target_path("error_faq", topic)))

        # --- T10 步骤 3：正式入库 ---
        res = await server.call_tool("kb_ingest", {
            "kind": "error_faq", "path": draft_path, "meta": {"title": topic},
            "dry_run": False, "index": True})
        txt = _text(res)
        ok = "已入库" in txt
        check("T10-3 kb_ingest 正式入库并向量化", ok and "片段" in txt, txt.splitlines()[-1] if txt else "")
        out_path = kbt.target_path("error_faq", topic)
        if os.path.exists(out_path):
            added_sources.append((out_path, os.path.abspath(out_path)))

        # --- T5：错误串命中 FAQ（断言：走 error_faq 精确通道 + 被标为已登记） ---
        res = await server.call_tool("kb_error_lookup", {"error_text": error_text, "top_k": 3})
        txt = _text(res)
        check("T5 错误串命中已登记 FAQ",
              "error_faq 精确命中" in txt and "已登记FAQ" in txt,
              txt.splitlines()[0] if txt else "")
        # 未登记的错误串应退化并提示登记
        res = await server.call_tool("kb_error_lookup",
                                     {"error_text": "ZZZ_UNKNOWN_ERROR_9f3a_not_in_kb"})
        check("T5 未命中时提示 kb_note 登记",
              "kb_note" in _text(res) or "未找到" in _text(res))

        # --- T6/T7：新文档可被 get_doc / outline 读回 ---
        res = await server.call_tool("kb_outline", {"source": os.path.abspath(out_path)})
        check("T6 新入库文档可 outline", "章节树" in _text(res))
        res = await server.call_tool("kb_get_doc", {"source": os.path.basename(out_path)})
        check("T7 新入库文档可 get_doc（按文件名唯一解析）",
              "章节过滤" not in _text(res) and error_text in _text(res))

        # --- 校验拦截：正文过短必须报 error ---
        res = await server.call_tool("kb_ingest", {
            "kind": "defect_log", "payload": "太短", "meta": {"title": "短正文"},
            "dry_run": True})
        check("校验拦截：正文过短报 error", "正文过短" in _text(res) or "error" in _text(res))

        # --- P3-8 插件契约：未知产物目录给出映射表 ---
        res = await server.call_tool("kb_ingest", {
            "kind": "model_fact_card", "path": kbt.DRAFTS_DIR, "dry_run": True})
        txt = _text(res)
        check("P3-8 插件目录契约提示", "契约" in txt or "批量入库" in txt, txt.splitlines()[0] if txt else "")

    finally:
        if keep:
            print("  （--keep：保留样例草稿与入库文档）")
            return
        print("  回滚：")
        for _, ap in added_sources:
            try:
                index_docs.delete_file_points(ap)
                print(f"    - 已删点 {os.path.basename(ap)}")
            except Exception as e:
                print(f"    - 删点失败 {ap}: {e}")
        # 清理 ingest 状态与文件
        state = index_docs._load_json_state(index_docs.INGEST_STATE_FILE)
        for _, ap in added_sources:
            state.pop(os.path.abspath(ap), None)
        index_docs._save_json_state(index_docs.INGEST_STATE_FILE, state)
        for p in (kbt.target_path("draft", topic), kbt.target_path("error_faq", topic)):
            if os.path.exists(p):
                os.remove(p)
                print(f"    - 已删文件 {os.path.basename(p)}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="执行写入段（默认会回滚）")
    ap.add_argument("--keep", action="store_true", help="写入段保留样例知识")
    args = ap.parse_args()

    import kb_mcp_server as server   # noqa: 需要在 sys.path 设好后导入

    await readonly_checks(server)
    if args.commit:
        await commit_checks(server, args.keep)
    else:
        print("\n（未加 --commit：跳过写入段 T10）")

    print(f"\n===== 冒烟结果：{len(PASS)} PASS / {len(FAIL)} FAIL =====")
    for f in FAIL:
        print(f"  FAIL: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
