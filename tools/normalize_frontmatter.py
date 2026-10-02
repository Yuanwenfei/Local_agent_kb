#!/usr/bin/env python3
"""
front-matter 元数据批量规范化（P4-1 批量通道）

作用：给站点导出的 .md 批量补齐必填元数据（title / applies_to.framework / doc_date /
trust / canonical），使 `kb_meta_lint` 清零、版本与可信度过滤可用。

设计约束（开发计划 P4-1 / 设计 §8.1）：
- .md 的元数据**只认文档内 front-matter**（`parse_document` 里 md 分支不读 sidecar），
  所以必须改源文件；sidecar 只对 PDF/DOCX/TXT 生效。
- 本脚本**只做插入、不重写既有行**：正文与已有 front-matter 行保持字节级不变，
  因此不会踩 YAML 重新渲染的坑（引号/日期/特殊字符全部保持原样）。
- 改完用 `tools/backfill_payload.py --all-in-state` 让检索**立即生效**（set_payload，不重嵌）。
- 改 front-matter 会改文件内容 hash → 下次全库重跑会把这批文件判为 modified 重嵌一次
  （与 PARSER_VERSION 2.2 的图注一起，顺路）。

默认取值（2026-10-01 用户拍板）：
  applies_to.framework = "2026"（统一，Legacy 不单独降版本）
  trust = tutorial（先全量 tutorial，日后有实测结论再升 measured）
  canonical = true（站点每页一篇，重复组由 kb_dupes 单独降副本）
  source_origin = web（站点导出）/ kb（自研内嵌技能文档）
  doc_date = 已有 `fetched`；缺失回退文件 mtime

用法:
  python312\\python.exe tools\\normalize_frontmatter.py --all --dry-run   # 预演（默认，不落盘）
  python312\\python.exe tools\\normalize_frontmatter.py --all --apply     # 真写（自动备份）
  python312\\python.exe tools\\normalize_frontmatter.py --all --restore   # 从备份目录整体还原
  python312\\python.exe tools\\normalize_frontmatter.py <文件> --apply    # 只处理指定文件

范围（--all）：md-source 下的 Emulate3D-2026-Doc / Emulate3D-Tutorials / 内嵌技能文档
排除：md-source/E3D Tutorials（用户另行整理）、assets 目录、*.bak*
"""
import argparse
import datetime
import os
import re
import shutil
import sys

try:
    import yaml
except ImportError:  # 与 kb_schema 同款兜底：无 PyYAML 时跳过 YAML 修复
    yaml = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_SOURCE = os.path.join(PROJECT_ROOT, "md-source")

# vault → source_origin
VAULTS = (
    ("Emulate3D-2026-Doc", "web"),
    ("Emulate3D-Tutorials", "web"),
    ("内嵌技能文档", "kb"),
)
EXCLUDE_TOP = ("E3D Tutorials",)
BACKUP_ROOT = os.path.join(PROJECT_ROOT, "backup", "frontmatter-20261001")

FRAMEWORK = "2026"
TRUST = "tutorial"
MODEL = "general"

DATE_RE = re.compile(r'(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})')
# 匹配文件开头的 front-matter 块，body 为块内文本（不含两侧 ---）
FM_RE = re.compile(r'\A---[ \t]*\r?\n(?P<body>.*?)\r?\n---[ \t]*(?:\r?\n)?', re.S)
KEY_RE = lambda k: re.compile(rf'^{re.escape(k)}[ \t]*:')  # noqa: E731


def _kv(lines, key):
    for ln in lines:
        m = re.match(rf'^{re.escape(key)}[ \t]*:[ \t]*(.*?)[ \t]*$', ln)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return ""


def repair_yaml(lines: list[str], notes: list[str]) -> list[str]:
    """把导致 YAML 解析失败的标量值加引号（最多迭代 8 轮）。

    实测语料有 106 篇的 `title:` 值里带「冒号+空格」（如 `title: Error: Cannot Find Type…`），
    属**原有的非法 YAML**：整块 front-matter 会退回 `_naive_yaml` 兜底，
    于是嵌套的 `applies_to.framework` 直接丢失（扁平键侥幸保住）。
    这里只给出错行加引号，语义不变、不重写其它行。
    """
    if yaml is None or not lines:
        return lines
    _SCALAR_RE = re.compile(r'^([^\s#][^:]*:[ \t]*)(.*?)[ \t]*$')
    for _ in range(8):
        try:
            if isinstance(yaml.safe_load("\n".join(lines)), dict):
                return lines
        except yaml.YAMLError as e:
            mark = getattr(e, "problem_mark", None)
            if mark is None or not (0 <= mark.line < len(lines)):
                notes.append("YAML修复失败：定位不到出错行")
                return lines
            i = mark.line
            m = _SCALAR_RE.match(lines[i])
            if not m:
                notes.append(f"YAML修复失败：第 {i + 1} 行不是 key: value")
                return lines
            val = m.group(2)
            # 已经整串双引号包裹、或本身是流式/块式/锚点等特殊形式时不再自动改写
            if not val or (val.startswith('"') and val.endswith('"')) or \
                    val[:1] in ("[", "{", "|", ">", "&", "*", "#"):
                notes.append(f"YAML修复失败：第 {i + 1} 行为特殊形式，需人工确认")
                return lines
            # 整串加双引号（内含单引号如 `'pkg' Tool` 也能正确保留语义）
            lines[i] = m.group(1) + '"' + val.replace("\\", "\\\\").replace('"', '\\"') + '"'
            notes.append(f"YAML修复←{m.group(1).strip().rstrip(':')}")
    notes.append("YAML修复未收敛（8 轮仍失败）")
    return lines


def plan_edits(body: str, path: str, origin: str) -> tuple[list[str], list[str]]:
    """返回 (新 front-matter 行, 备注)；无可补时新行与 body 相同。"""
    lines = body.splitlines()
    notes: list[str] = []
    scalars: list[str] = []

    def has(key):
        return any(KEY_RE(key).match(ln) for ln in lines)

    if not has("title"):
        name_val = _kv(lines, "name")
        if name_val:
            title, src = name_val, "name"
        else:
            title, src = os.path.splitext(os.path.basename(path))[0], "文件名"
        scalars.append(f"title: {title}")
        notes.append(f"title←{src}")

    if not has("doc_date"):
        fetched = _kv(lines, "fetched")
        m = DATE_RE.search(fetched) if fetched else None
        if m:
            iso, src = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", "fetched"
        else:
            try:
                iso = datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
                src = "mtime"
            except OSError:
                iso, src = "", ""
        if iso:
            scalars.append(f"doc_date: {iso}")
            notes.append(f"doc_date←{src}")

    if not has("trust"):
        scalars.append(f"trust: {TRUST}")
        notes.append(f"trust={TRUST}")
    if not has("canonical"):
        scalars.append("canonical: true")
        notes.append("canonical=true")

    out = list(lines)
    # applies_to：若已有块则只补缺子键（插在块末），否则整体追加
    idx = next((i for i, ln in enumerate(lines)
                if re.match(r'^applies_to[ \t]*:[ \t]*$', ln)), None)
    sub: list[str] = []
    if idx is None:
        if not has("applies_to"):
            sub = ["applies_to:", f'  framework: "{FRAMEWORK}"', f"  model: {MODEL}"]
            notes.append(f"applies_to.framework={FRAMEWORK}")
    else:
        j = idx + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].startswith((" ", "\t"))):
            j += 1
        block = lines[idx + 1:j]
        if not any(re.match(r'^\s+framework[ \t]*:', b) for b in block):
            sub.append(f'  framework: "{FRAMEWORK}"')
            notes.append(f"applies_to.framework={FRAMEWORK}")
        if not any(re.match(r'^\s+model[ \t]*:', b) for b in block):
            sub.append(f"  model: {MODEL}")
        if sub:
            out[j:j] = sub
            sub = []

    # source_origin 由 vault 决定，放在标量末尾
    if not has("source_origin"):
        scalars.append(f"source_origin: {origin}")
        notes.append(f"source_origin={origin}")

    out.extend(scalars)
    out.extend(sub)
    out = repair_yaml(out, notes)  # 原有非法 YAML 一并修（如 title 含「冒号+空格」）
    if not notes:
        return lines, []
    return out, notes


def split_fm(text: str):
    m = FM_RE.match(text)
    if not m:
        return None
    return m.group("body"), text[m.end():]


def collect_files(paths: list[str], scan_vaults: bool) -> list[str]:
    """显式路径（文件/目录）+ 可选 vault 扫描，返回去重后的 .md 清单"""
    files: list[str] = []

    def _add_tree(root):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d.lower() != "assets" and d not in EXCLUDE_TOP]
            for fn in filenames:
                if fn.lower().endswith(".md") and ".bak" not in fn.lower():
                    files.append(os.path.join(dirpath, fn))

    for p in paths:
        ap = os.path.abspath(p)
        if os.path.isfile(ap):
            files.append(ap)
        elif os.path.isdir(ap):
            _add_tree(ap)
        else:
            print(f"  [skip] 路径不存在：{p}")
    if scan_vaults:
        for vault, _origin in VAULTS:
            root = os.path.join(MD_SOURCE, vault)
            if os.path.isdir(root):
                _add_tree(root)
    return sorted(set(files))


def vault_origin(path: str) -> str:
    """站点导出的两个 vault → web；其余（自研/未归类）→ kb"""
    for vault, origin in VAULTS:
        if os.path.join(MD_SOURCE, vault).lower() in path.lower():
            return origin
    return "kb"


def process(path: str, apply: bool, samples: list, rel: str) -> dict:
    origin = vault_origin(path)
    with open(path, "rb") as f:
        raw = f.read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    eol = "\r\n" if "\r\n" in text[:4000] else "\n"

    got = split_fm(text)
    if got is None:  # 无 front-matter：新建
        body, rest = "", text
        created = True
    else:
        body, rest = got
        created = False

    new_lines, notes = plan_edits(body, path, origin)
    if not notes:
        return {"status": "ok", "changed": False, "notes": []}

    new_body = eol.join(new_lines)
    new_text = f"---{eol}{new_body}{eol}---{eol}{rest}"
    if len(samples) < 8:
        samples.append((rel, created, body.splitlines(), new_lines))

    if not apply:
        return {"status": "dry", "changed": True, "notes": notes}

    backup = os.path.join(BACKUP_ROOT, rel)
    if not os.path.exists(backup):
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        shutil.copy2(path, backup)
    out = new_text.encode("utf-8")
    if bom:
        out = b"\xef\xbb\xbf" + out
    with open(path, "wb") as f:
        f.write(out)
    return {"status": "applied", "changed": True, "notes": notes, "backup": backup}


def restore(files) -> int:
    n = 0
    for path in files:
        rel = os.path.relpath(path, PROJECT_ROOT)
        backup = os.path.join(BACKUP_ROOT, rel)
        if os.path.exists(backup):
            shutil.copy2(backup, path)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="front-matter 元数据批量补齐（只插入、不重写）")
    ap.add_argument("files", nargs="*", help="指定文件（不给则配合 --all 扫描 md-source）")
    ap.add_argument("--all", action="store_true", help="处理 md-source 下三个 vault")
    ap.add_argument("--dry-run", action="store_true", help="预演，不写盘（默认行为）")
    ap.add_argument("--apply", action="store_true", help="真写（自动备份到 backup/frontmatter-20261001）")
    ap.add_argument("--restore", action="store_true", help="从备份目录还原")
    args = ap.parse_args()

    if not args.files and not args.all:
        ap.error("请给文件路径，或使用 --all")

    files = collect_files(args.files, args.all)
    if not files:
        ap.error("没找到任何 .md")

    if args.restore:
        n = restore(files)
        print(f"已从 {os.path.relpath(BACKUP_ROOT, PROJECT_ROOT)} 还原 {n} 个文件")
        return

    apply = bool(args.apply) and not args.dry_run
    mode = "真写（备份+落盘）" if apply else "预演（不落盘）"
    samples: list = []
    stat = {"changed": 0, "same": 0}
    key_count: dict[str, int] = {}
    per_vault: dict[str, int] = {}

    for path in files:
        rel = os.path.relpath(path, PROJECT_ROOT)
        try:
            r = process(path, apply, samples, rel)
        except Exception as e:  # 单个文件失败不影响整体
            print(f"  [fail] {rel}: {e}")
            continue
        if r["changed"]:
            stat["changed"] += 1
            per_vault[vault_origin(path)] = per_vault.get(vault_origin(path), 0) + 1
            for n in r["notes"]:
                k = n.split("=")[0].split("←")[0]
                key_count[k] = key_count.get(k, 0) + 1
        else:
            stat["same"] += 1

    print(f"## normalize_frontmatter（{mode}）")
    print()
    print(f"- 扫描文件：**{len(files)}** 篇")
    print(f"- 需要改动：**{stat['changed']}** 篇；已合规无需改：{stat['same']} 篇")
    if key_count:
        print("- 补齐项统计：" + "；".join(f"{k}×{v}" for k, v in sorted(key_count.items())))
    if per_vault:
        print("- 按 vault：" + "；".join(f"{k}×{v}篇" for k, v in sorted(per_vault.items())))

    if samples:
        print()
        print(f"### 改动样例（前 {len(samples)} 篇）")
        for rel, created, old_lines, new_lines in samples:
            print(f"\n**{rel}**" + ("（无 front-matter → 新建）" if created else ""))
            print("```yaml")
            if created:
                print("（原文件无 front-matter）")
            for ln in new_lines:
                mark = "  " if ln in old_lines else "+ "
                print(f"{mark}{ln}")
            print("```")

    if not apply:
        print()
        print("_预演结束，未写盘。确认后用 `--all --apply` 真写（自动备份）；"
              f"写后执行 `tools\\\\backfill_payload.py --all-in-state` 让检索立即生效。_")
    else:
        print()
        print(f"- 备份目录：`{os.path.relpath(BACKUP_ROOT, PROJECT_ROOT)}`（还原：`--all --restore`）")
        print("- 下一步：`python312\\\\python.exe tools\\\\backfill_payload.py --all-in-state`"
              f"（秒级，不重嵌）")
        print("- 复查：MCP `kb_meta_lint(top_docs=20)` 期望缺失 0 篇；"
              "trust 生效后跑 `python312\\\\python.exe tools\\\\smoke_p3.py`；"
              "量化回归集已移出仓库，口径见 README「验收回归」小节")


if __name__ == "__main__":
    sys.exit(main())
