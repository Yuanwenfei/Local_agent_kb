# Local Agent KB

<div align="center">

**本地知识库 —— 让 AI 助手拥有你的私有文档检索能力**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green)](https://modelcontextprotocol.io/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%201.13-red)](https://qdrant.tech/)
[![Tools](https://img.shields.io/badge/MCP%20Tools-11-orange)](./kb_mcp_server.py)
[![Tests](https://img.shields.io/badge/acceptance-T1--T14-brightgreen)](./本地知识库开发计划_20261001.md)

</div>

---

## 📖 概述

**Local Agent KB** 是一个基于 **MCP (Model Context Protocol)** 的本地知识库系统，为 AI 编程助手提供私有文档的**混合检索**能力：检索到的片段会带上通道来源、可信度、适用版本、文档日期与图片路径，AI 可以据此直接作答、并对结论负责。

核心是一句话：**把"文档里写了什么"变成"AI 能引用的证据"**。因此除了向量检索，还内置了元数据规范（v2）、版本过滤与漂移告警、图片管道（图注 + 返图）、知识入库通道（草稿 → 校验 → 转正）与一套可复现的验收回归。

### ✨ 核心特性

- 🔀 **四通道混合检索** — dense(0.7) + sparse BM42(0.3) + 精确串(1.0) + 术语锚点(1.0)，客户端加权 RRF 融合，并发发出（延迟增幅 <30%）
- 🧭 **元数据规范 v2** — front-matter 五项必填（`title`/`applies_to.framework`/`doc_date`/`trust`/`canonical`），缺失按 R1 降权不过滤；支持 `version` / `model` / `trust` / `kind` / `section` / `path_prefix` 过滤与 `⚠version-drift` 告警
- 🖼️ **图片管道** — 图片链接 URL 解码后本地可达；图注五级抽取（显式标记/alt/正文/章节路径/文件名）并**参与嵌入**；top1 片段自动附带**主要插图**缩略图
- 📚 **知识类型与入库通道** — 8 类知识模板（事实卡/签名表/错误 FAQ/缺陷登记/枚举示例/QLP 片段/版本矩阵/资产登记）+ 草稿 → 校验 → 转正闭环
- 🔍 **多种检索入口** — 通用检索、全文取回、章节大纲、错误串专用查表、文档清单、元数据体检、副本巡检
- ⚡ **增量索引** — 按内容哈希 + 解析器版本判定，仅处理变动文件；模型按需加载、空闲自动释放
- 🛡️ **数据质量保障** — 模型健康检查、零向量监控（预检 + 批内 + 全量扫描）、段健康监控
- 🔒 **完全本地** — 数据不出本机；仅 Qdrant 运行在 Docker（端口已收窄到回环）

---

## 🏗️ 架构

```text
┌──────────────────────────────────────────────────────────┐
│              AI 编程助手 (Copilot / Kimi Code)            │
│                        (云端 LLM)                         │
│                            │                              │
│                    MCP Tool Call (stdio)                  │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│                kb_mcp_server.py（本地进程）                │
│                                                           │
│  查询 ──┬─► dense  (e5-large, 1024d) ─┐                   │
│         ├─► sparse (BM42)             ├─► 加权 RRF 融合    │
│         ├─► exact  (search_text)      │        │          │
│         └─► anchor (术语 df 校验)     ┘        ▼          │
│                                        后处理：目录页降权／ │
│                                        可信度权重／时效／   │
│                                        单文档≤2／返图       │
│                                               │           │
│         ┌─────────────────────┬───────────────┘           │
│         │  kb_schema.py       │  kb_templates.py          │
│         │  元数据 v2 / 图片    │  8 类知识模板 / 入库校验    │
│         └─────────────────────┴───────────────┐           │
└───────────────────────────────────────────────┼───────────┘
                                                │
                              ┌─────────────────▼──────────┐
                              │      Qdrant Server 1.13     │
                              │  dense(1024/Cosine) + bm42  │
                              │  Docker 容器 / named volume │
                              └─────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  index_docs.py（索引脚本）                                 │
│  解析 → 分块 → 向量化 → 入库（增量）→ 零向量扫描           │
└──────────────────────────────────────────────────────────┘
```

### 技术栈

| 层级 | 选型 | 说明 |
|------|------|------|
| **协议层** | MCP (stdio) | AI 助手与本地工具的标准通信协议 |
| **检索网关** | Python MCP Server (`kb_mcp_server.py`) | 编排四通道检索、过滤、后处理与返回体渲染 |
| **稠密向量** | fastembed `intfloat/multilingual-e5-large` | 1024 维 Cosine，纯 ONNX 本地推理 |
| **稀疏向量** | fastembed `Qdrant/bm42-all-minilm-l6-v2-attentions` | 命名向量 `bm42`，与稠密向量写入同一集合 |
| **向量存储** | Qdrant 1.13（Docker） | 双向量检索、payload 过滤、count 聚合 |
| **元数据层** | `kb_schema.py` | front-matter v2（嵌套 YAML）/ sidecar / 图片路径解析 / lint |
| **模板层** | `kb_templates.py` + `templates/` | 8 类知识模板、入库校验、草稿落盘 |
| **文档解析** | `document_parsers.py`（PyMuPDF / python-docx） | PDF 表格转 Markdown、页眉页脚过滤、图注抽取 |

---

## 🚀 快速开始

### 前置条件

- [Docker](https://docs.docker.com/get-docker/)（用于运行 Qdrant）
- Python 3.12（本仓库自带嵌入式运行时 `python312\`）

> 下文命令以 Windows + 本仓库自带运行时为准：`python312\python.exe`。

### 1. 启动 Qdrant

```powershell
docker compose up -d
```

数据持久化在 Docker **named volume**（`local_kb_qdrant_data`），端口**只绑回环**：

```yaml
ports:
  - "127.0.0.1:6333:6333"   # REST API
  - "127.0.0.1:6334:6334"   # gRPC
```

> **两个必须知道的坑**
> 1. **别改回 bind mount**（`./data:/qdrant/storage`）：Windows 对 mmap 写入同步不可靠，Qdrant 段优化时会把向量数据归零（历史上在膨胀比 12.57x 后触发过）。因此**备份只能用快照**，见「运维与排障」。
> 2. Qdrant 默认**无鉴权**，所以端口只绑 `127.0.0.1`。若要暴露到局域网，请加 `QDRANT__SERVICE__API_KEY` 并同步配置 MCP 的 `QDRANT_API_KEY`。

### 2. 安装依赖

```powershell
python312\python.exe -m pip install -r requirements.txt
```

首次运行会自动下载 Embedding 模型（约 2.1GB）到 `models/`。若直连 HuggingFace 不通，先设镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 3. 索引入库文档

```powershell
# 增量索引（仅处理新增/修改/删除的文件）
python312\python.exe index_docs.py md-source

# 全量重嵌（元数据或解析器变更后使用）
python312\python.exe index_docs.py md-source --reindex-all
```

支持 `.md`、`.pdf`、`.docx`、`.txt`。默认文档源目录 `md-source/`（可用 `KB_DOCS_DIR` 覆盖）。

### 4. 配置 MCP Server

在 AI 助手的 MCP 配置里指向本仓库（以 CodeBuddy / Kimi 的 `mcp.json` 为例）：

```json
{
  "mcpServers": {
    "local-agent-kb": {
      "command": "E:\\Local_agent_kb\\python312\\python.exe",
      "args": ["E:\\Local_agent_kb\\kb_mcp_server.py"],
      "env": {
        "KB_USE_GPU": "0",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "KB_COLLECTION": "emulate3d_docs",
        "KB_IDLE_TIMEOUT": "600",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

> **改过 `kb_mcp_server.py` 后必须重启 MCP 服务**（进程不会热加载）。

### 5. 验证联通

```powershell
# 按 MCP 客户端的启动方式拉起 server，做一次握手 + tools/list
python312\python.exe tools\probe_mcp_stdio.py
```

预期输出 `OK: MCP 握手与 tools/list 均正常` 与 11 个工具。

---

## 🧩 MCP 工具（11 个）

### 检索类

| 工具 | 用途 | 关键参数 |
|------|------|----------|
| `search_tech_kb` | 混合检索主入口 | `query`、`mode`(hybrid/semantic/exact)、`version`、`model`、`trust`、`kind`、`section`、`path_prefix`、`canonical_only`、`top_k`、`score_threshold`、`attach_image` |
| `kb_get_doc` | 取整篇/某章节**连续全文**（片段不够时用） | `source`、`heading`、`budget`(默认 3000 字符) |
| `kb_outline` | 取章节树（读长文前先看它，少读无关内容） | `source`、`max_depth` |
| `kb_error_lookup` | 错误串专用查表（根因 + 守卫/修复范式） | `error_text`、`top_k`、`exact_only` |

### 盘点类

| 工具 | 用途 |
|------|------|
| `kb_stats` | 规模/类型/可信度分布、向量配置、治理指标（含图/目录页/副本）；`detail=true` 才全扫 |
| `kb_list_docs` | 文档清单与元数据（片段数/版本/日期/可信度/是否正本） |
| `kb_meta_lint` | 元数据体检：缺必填项、索引字段未回填、`canonical_uid` 冲突 |
| `kb_caption` | 图注治理：抽图注 / 报告图注是否已进向量；`mode=summary` 为二期 VLM |
| `kb_dupes` | 副本巡检（hash/同名/同文本），给正本建议与可粘贴的降副本元数据 |

### 入库与治理类

| 工具 | 用途 |
|------|------|
| `kb_note` | 把检索失败/踩坑现场登记为规范草稿（落 `kb-inbox/drafts/`） |
| `kb_ingest` | 草稿转正：`kind` + `payload`/`path` + `meta`，**先 `dry_run=true` 看校验报告**再落盘向量化 |

### `search_tech_kb` 返回体约定

- 分数口径：`hybrid` 返回 **RRF 融合分**（单通道命中约 0.5、双通道 1.0，**不是余弦相似度**）；`semantic` 返回余弦。
- `score_threshold`：`semantic` 下作用于最终结果；`hybrid` 下只作用于 dense 预筛（sparse/anchor/exact 不受限），**不要用 0.65 这类高阈值**。
- 每个片段标注：通道、可信度权重、适用版本（`⚠version-drift` 表示与请求版本不一致）、文档日期（`(推断)` 表示由文件时间兜底）、是否正本。
- `attach_image=true`（默认）：top1 片段含本地图片时附带**主要插图**缩略图（取该片段本地图中体积最大者，≤512px JPEG）；纯文本客户端自动降级为路径列表。

---

## 🎯 检索机制

### 四通道加权融合

```
score = Σ w_c · 1/(k + rank_c)        k = 1（与 Qdrant 内置 RRF 行为一致）
w: dense 0.7 | sparse 0.3 | exact 1.0 | anchor 1.0
```

- **dense**：语义召回（e5-large）
- **sparse**：BM42 稀疏向量，补关键词/标识符召回
- **exact**：`search_text` 精确串（报错原文、类名、控件语法；归一化去引号/空白）
- **anchor**：查询含明确标识符（如 `RackConfigurator`）且 df ≤ 60 时，字面命中强力加权

### 后处理规则

| 规则 | 取值 | 说明 |
|------|------|------|
| 候选池 | `top_k × 3` | 后处理前的池子 |
| 目录页降权 | `×0.6` **且不占 Top3** | 硬约束：目录页保留召回但不占最优位 |
| 可信度权重 | `measured 1.0` / `tutorial 0.9` / `pending 0.75` | 缺失兜底 `pending` |
| 时效衰减 | 6 个月线性 → `0.85` 后不再衰减 | **只对「结论类」`kind`**（`note`/`defect_log`/`error_faq`/`model_fact_card`/`version_matrix`）；教程/手册/API 参考恒定 1.0（"教程过时"由版本过滤表达）；`date_inferred` 一律豁免 |
| 单文档上限 | `≤2` 条 | 避免同一篇霸榜 |
| 低置信提示 | hybrid 单通道支持 <0.5；semantic 余弦 <0.60 | 提示转 `kb_note` 回流 |

---

## 📐 元数据规范 v2

### 必填五项与兜底

`title`、`applies_to.framework`、`doc_date`、`trust`、`canonical`。缺失按 **R1：降权不过滤** 兜底（`unknown` / 文件时间 + `date_inferred` / `pending`），并由 `kb_meta_lint` 出清单。

```yaml
---
title: Conveyor Properties
source-id: demo3d_2026/conveyors_properties
source-url: https://store.sim3d.com/demo3d_2026/conveyors_properties
fetched: 2026-09-26          # 抓取日期（原始导出字段，保留）
doc_date: 2026-09-26         # 文档日期（≤ 结论形成日期）
trust: tutorial              # measured | tutorial | pending
canonical: true              # true=正本；false 需配 superseded_by
source_origin: web           # kb | plugin | web | decompile
applies_to:
  framework: "2026"          # 适用框架版本（参与 version 过滤与漂移告警）
  model: general             # 适用机型/控制器（general 表示通用）
kind: doc                    # 知识类型，见下
---
```

> **`.md` 的元数据只认文档内 front-matter**（sidecar `<文档>.meta.yaml` 只对 PDF/DOCX/TXT 生效）。改完 front-matter 用 `tools/backfill_payload.py --all-in-state`（秒级、**不重嵌**）让检索立即生效。

### 知识类型（8 类 + 草稿）

`model_fact_card`（模型事实卡）、`part_signature`（部件签名表）、`version_matrix`（版本差异矩阵）、`error_faq`（错误串 FAQ）、`defect_log`（缺陷登记）、`enum_examples`（枚举示例）、`qlp_snippet`（QLP 可跑片段）、`asset_pointer`（资产路径登记）、`draft`（草稿）。

各类自带默认 `trust`/`source_origin` 与固定小节骨架（`templates/`）。

---

## 🖼️ 图片管道

1. **路径可达**：相对链接必经 URL 解码（`%20` 等）；实测不解码仅 ~15% 可达，解码后 100%。
2. **图注五级**（`document_parsers._image_caption`）：显式标记（`图注：…` / `<!-- caption: … -->`）→ 图片 `alt` → 紧邻正文 → 章节路径 → 文件名。
3. **图注参与嵌入**：以 `[图注] …` 行并入切片文本，使"图里写了什么"可被检索命中（解析器版本 `2.2` 起）。
4. **返图**：top1 片段的首图缩略图（体积最大者，≤512px JPEG，可用 `KB_THUMB_*` 调）。
5. **站点外链补图**：`tools/fetch_site_images.py` 通过站点原始内容接口把外链图落地到本地 `assets/`，并改写 md 为相对路径（预演/备份/回滚齐备）。

---

## 📂 项目结构

```text
.
├── kb_mcp_server.py          # MCP Server：四通道检索 + 11 个工具 + 后处理/返回体
├── kb_schema.py              # 元数据 v2：front-matter 解析、lint、payload 映射、图片路径
├── kb_templates.py           # 知识类型模板层：渲染 / 校验 / 草稿落盘
├── document_parsers.py       # 解析器：MD / PDF / DOCX / TXT（含图注抽取）
├── index_docs.py             # 批量索引（增量 + 质量保障）
├── index_docs.bat            # Windows 快捷索引
├── docker-compose.yml        # Qdrant 编排（named volume / 回环绑定）
├── requirements.txt
├── index_state.json          # 全量索引状态（自动生成）
├── kb_ingest_state.json      # 单文件入库状态（自动生成）
├── templates/                # 8 类知识模板
├── kb-inbox/                 # 草稿与待转正产物（drafts/ 等）
├── md-source/                # 默认文档源目录（含 assets/ 图片）
├── tools/                    # 交付工具与冒烟脚本
│   ├── normalize_frontmatter.py   # 批量补 front-matter（只插入/备份/可回滚）
│   ├── backfill_payload.py        # 元数据回填（set_payload，--sync 清理过期键）
│   ├── fetch_site_images.py       # 站点外链图片落地
│   ├── audit_images.py            # 图片审计（不依赖 Qdrant）
│   ├── ensure_schema.py           # 集合与 payload 索引创建
│   ├── probe_mcp_stdio.py         # MCP 握手 + tools/list 连通性探针
│   └── smoke_p1..p4.py            # 各阶段冒烟自检
├── regression/               # 验收回归
│   ├── queries.jsonl              # 固定查询集（24 条）
│   ├── eval.py / eval_p2.py       # 度量脚本（dense / 四通道）
│   ├── summarize.py               # 结果汇总
│   ├── baseline_20261001.md       # 基线报告
│   └── results/                   # 每次运行结果（含 baseline.json / latest.json）
├── backup/                   # 快照与回滚备份
└── logs/                     # 索引日志（自动生成）
```

---

## ⚙️ 配置（环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant 地址 |
| `KB_COLLECTION` | `emulate3d_docs` | 集合名 |
| `KB_DOCS_DIR` | `<项目>/md-source` | 默认文档源目录 |
| `KB_ASSETS_ROOT` | `<项目>/md-source` | 图片资产根（HTML `images/x.jpg` 反查用） |
| `KB_SITE_BASE` | `https://store.sim3d.com` | 站点相对图片链接补全域名 |
| `KB_INBOX_DIR` | `<项目>/kb-inbox` | 草稿与待转正产物根 |
| `KB_CANONICAL_HINT` | `local_agent_kb` | 正本判据（路径含此片段的优先） |
| `KB_DUPES_EXTRA_DIRS` | `E:\kimi code workbentch` | `kb_dupes` 额外巡检目录（不入库） |
| `KB_USE_GPU` | `0` | `1` 启用 CUDA（ONNX Runtime） |
| `KB_IDLE_TIMEOUT` | `600` | 模型空闲释放秒数（`0` 不释放） |
| `KB_CHUNK_MAX_CHARS` | `1200` | 单切片上限字符数 |
| `KB_DOC_BUDGET` | `3000` | `kb_get_doc` 单次返回字符预算 |
| `KB_THUMB_MAX_SIDE` / `KB_THUMB_QUALITY` | `512` / `80` | 返图缩略图边长与 JPEG 质量 |
| `KB_DF_CACHE_TTL` | `600` | 锚点通道 df 计数缓存（秒） |

---

## 🛠️ 运维与排障

### 常用 Docker 命令

```powershell
cd e:\Local_agent_kb
docker compose up -d              # 启/重建（改 compose 后用）
docker compose stop|start qdrant  # 临时停/起（数据保留）
docker logs --tail 50 local_kb_qdrant
```

> 这台 Docker 可能是共用的（本机还跑着其它服务的容器），重启 Docker 会连带影响它们。

### 备份（只能用快照）

```powershell
# 1) 让 Qdrant 生成快照
curl.exe -X POST "http://localhost:6333/collections/emulate3d_docs/snapshots"
# 2) 把快照从容器里拷出来（named volume 在 WSL2 VM 内，Windows 侧直接拷不到）
docker cp "local_kb_qdrant:/qdrant/snapshots/emulate3d_docs/<snapshot-name>" "E:\Local_agent_kb\backup\"
```

### 排障清单

| 现象 | 检查 |
|------|------|
| `MCP error -32000: Connection closed` | 多数是 **import 失败**而非协议问题。本仓库跑在**嵌入式 Python**（`python312._pth`），`sys.path` **不含脚本目录**——新增顶层模块必须在入口 `sys.path.insert(0, <脚本目录>)`。用 `tools\probe_mcp_stdio.py` 一眼区分 |
| 改了 `kb_mcp_server.py` 不生效 | MCP 进程不会热加载，**重启 MCP 服务** |
| 检索不到新写的元数据 | 先 `tools\backfill_payload.py --all-in-state`（payload 秒级生效）；图片/正文变更才需要重嵌 |
| 图看不到 | 确认是本地图（外链图不会返图）；`kb_caption` 看"图注是否已进向量" |
| 单文件入库状态与全量状态不一致 | `index_docs.py` 用 `index_state.json`；`kb_ingest`/`kb_caption` 用 `kb_ingest_state.json`，属设计如此 |

---

## ✅ 质量保障

### 解析与索引质量

- 模型健康检查（热身确认输出非零向量）、**零向量监控**（批内 >10% 自动重建 session）、**预索引质量检查**、**全量零向量扫描**；
- 优化器安全配置（`indexing_threshold=50000`、`max_optimization_threads=1`）、每 5000 chunks 重建 session、段数量健康监控；
- 删文件保护：单次消失比例 >80% 时阻止自动清理。

### 验收回归

```powershell
python312\python.exe regression\eval_p2.py --compare baseline   # 四通道（权威指标集）
python312\python.exe regression\eval.py    --compare baseline   # dense 单通道
python312\python.exe tools\smoke_p4.py --commit                 # 分阶段冒烟（带自动回滚）
```

- 固定查询集 **24 条**（`regression/queries.jsonl`：API 用法、框架行为、错误串、版本差异、模型事实、缺口探针）；
- 指标：`MRR@5`、`Recall@5`、分数梯度、目录页 Top1 数、单文档占比、低置信数、延迟，以及 **T1–T14** 断言（版本漂移、图注检索、返图降级、时效口径等）；
- 达成口径见 `本地知识库开发计划_20261001.md`（含每项实测值与修正记录）。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。开发建议：

- 二期 `kb_caption(mode=summary)`：用 VLM 读图写**真图注**（当前仅规则抽取的"弱图注"）
- 更强的时间/版本信号：拿到文档真实发布日后启用更细的时效策略
- 检索评估扩充：把线上失败 query 回流进 `regression/queries.jsonl`

---

## 📄 License

MIT。本仓库**暂未随附 `LICENSE` 文件**——如需正式声明，补一份 MIT 全文并署名即可。

---

<div align="center">
Made with ❤️ for the AI-assisted development community
</div>
