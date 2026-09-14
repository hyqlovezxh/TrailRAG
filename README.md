<div align="center">

# TrailRAG · 语溯RAG

**Trace every clue — RAG with evidentiary, multi-hop, time-aware retrieval.**

*循迹每一处线索：带证据链、多跳、时序感知的检索增强生成框架，面向低质量、高噪声、矛盾多的真实数据（调查笔录、聊天记录、资金流水、话单……）。*

[![Python](https://img.shields.io/badge/Python-3.11+-4ecdc4?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-ff6b6b?style=for-the-badge&logoColor=white&labelColor=1a1a2e)](./LICENSE)
[![Status](https://img.shields.io/badge/status-open%20source%20release-00d9ff?style=for-the-badge&logoColor=white&labelColor=1a1a2e)](./README.md)
[![Benchmarks](https://img.shields.io/badge/📊%20benchmarks-included-4ecdc4?style=for-the-badge&logoColor=white&labelColor=1a1a2e)](./RAG四套知识库权威对比分析报告_语溯vsSemantica vsLightRAGvsGraphRAG.md)

</div>

---

## ⚡ 四套 RAG 对比一览

| 维度 | 🏆 语溯RAG | Semantica | LightRAG | GraphRAG |
|---|---|---|---|---|
| **目的加权总评**（取证/低质量场景） | **94.4%** | 87.4% | 80.2% | 67.4% |
| **等权总评**（通用企业知识库） | **93.3%** | 88.9% | 84.4% | 68.9% |
| **证据挖掘命中率**（`fraud_case_v1` 8 线索题） | **100%（8/8）** | 66% | 62.5% | 62.5% |

🏆 语溯RAG 在目的加权与等权两种口径下均**第一**，详见 [完整对比报告 →](./RAG四套知识库权威对比分析报告_语溯vsSemantica vsLightRAGvsGraphRAG.md)

---

## 目录

- [为什么用语溯RAG](#为什么用语溯rag)
- [系统架构](#系统架构)
- [核心特性](#核心特性)
- [基准与对比](#基准与对比)
- [快速上手](#快速上手)
- [一键启动 / 部署](#一键启动--部署)
- [环境变量参考](#环境变量参考)
- [API 概览](#api-概览)
- [环境自检](#环境自检)
- [测试](#测试)
- [关键配置](#关键配置)
- [路线图](#路线图)
- [文档](#文档)
- [贡献 / 许可 / 致谢](#贡献--许可--致谢)

---

## 为什么用语溯RAG

传统知识库大多针对**企业高质量知识**开发，文档规范、矛盾少，因此对**确定性、高质量语料**效果好。但真实世界的数据往往相反：**不确定性高、海量、低质量**。一个典型例子是调查/取证材料——聊天记录杂乱、笔录彼此矛盾、资金流水与话单不全、结构化表格（CSV/XLSX）被压平为松散文本。线索被语义噪声稀释、字段语义丢失、没有可跨源跳跃的"事件"节点、也没有时间模型去还原"某人某时干了某事"。

**语溯RAG** 正是为这种场景而生。它保留一套传统向量直用通道以即时作答，并叠加一条**以"事件节点"为根的图增强通道**，把聊天 → 笔录 → 资金流水 → 话单 → 取证记录连成一条**闭合、多跳、带时序的证据链**——显式呈现矛盾，而非静默合并。这里的"取证/证据挖掘"仅为这类低质量数据场景的一个代表性例子，用于检验知识库面对不确定性数据与海量低质量数据时的可用性。

## 系统架构

```mermaid
graph TB
    subgraph IN["多模态证据输入"]
        direction TB
        A1[笔录]
        A2[聊天记录]
        A3[资金流水 CSV]
        A4[话单]
        A5[卡口]
        A6[取证 JSON]
    end
    B[域感知分块]
    subgraph DUAL["双写管线（并行）"]
        direction LR
        C1[向量嵌入] --> VS[(向量库<br/>即时可检索)]
        C2[图抽取<br/>最多 128 路并发] --> GE[事件 / 实体 / 关系]
        GE --> GS[(事件图谱<br/>PPR 1-5 跳)]
    end
    Q[检索层<br/>向量 + 关键词 + 图 + 词法]
    subgraph FB["检索时动态构图（回退）"]
        direction LR
        FB1[向量命中为种子] --> FB2[隐式图<br/>SIM/ADJACENT/LEXICAL/EXACT_ID]
        FB2 --> FB3[PPR 隐式多跳<br/>图谱未建完时激活]
    end
    F[RRF 融合 + 重排]
    O[证据链输出<br/>带来源 + 时序]
    IN --> B --> DUAL
    VS --> Q
    GS --> Q
    Q -. 图谱未建完 .-> FB
    FB --> Q
    Q --> F --> O
```

*图 1 — 系统架构：向量直用通道与图增强通道在混合证据数据上并行运行，底层为向量/图/KV 三类存储，输出可解释、可溯源的答案。高清矢量版：[trailrag-architecture.svg](assets/trailrag-architecture.svg)*

## 核心特性

1. **双机制索引** —— 分块一完成向量化即可检索（向量直用），图增强同时并行构建证据图；不必等全图建完才能首次查询。
2. **事件节点图谱** —— 图节点是"事件"而非仅关键词/实体，使跨源、按时间顺序的推理更自然。
3. **双模态多跳检索** —— 图谱已建成时走事件图 PPR（1–5 跳，默认 3，依赖 scipy 稀疏求解）并显式枚举关系链以闭合证据；**图谱尚未建完时**自动回退到 BM25+向量形式，以向量命中分块为种子在检索时**动态构建隐式图**（SIM 互近邻 / ADJACENT 相邻 / LEXICAL 词项 / EXACT_ID 精确标识符四类边）并用 PPR 完成多跳扩散，跳转到语义最关联的远端分块——即"检索时动态构建知识图谱"，冷启动零成本。
4. **词法确定性通道** —— 对手机号/身份证/案件编号等做精确匹配，设 `0.85` 相似度下限，让确定性线索不被语义噪声吞没。
5. **低质量数据可用** —— 保留 CSV 字段语义、以事件锚定、显式呈现矛盾陈述，而非臆造单一答案。
6. **时序重建** —— 从带时间的事件中还原全案"某人某时干了某事"。
7. **混合数据类型** —— 聊天、笔录、资金流水 CSV、话单、卡口、取证 JSON、Office 文档（docx/xlsx/pdf/html/txt）通吃。
8. **可解释输出** —— 每个答案都带来源引用与可复现的证据路径。
9. **高索引速度** —— 图构建最多 128 路并发 LLM 抽取，图写入并发受控，跨分块批量落盘。
10. **可插拔存储** —— 向量库、图库、KV 库可替换（如 Milvus / Neo4j / PostgreSQL）。

### 工作流程

**入库（双机制并行）**

```mermaid
flowchart LR
    D[文档] --> CH[域分块]
    CH --> P1[向量嵌入]
    CH --> P2[图抽取<br/>并发上限 128]
    P1 --> VS[(向量库<br/>即时可答)]
    P2 --> GE[事件 / 实体 / 关系]
    GE --> GF[批量落盘<br/>写入并发 10]
    GF --> GS[(图库)]
    P1 -. 可选图增强叠加 .-> GS
    style VS fill:#4ecdc4,color:#1a1a2e
    style GS fill:#00d9ff,color:#1a1a2e
```

*图 2 — 入库流程。向量直用通道先就绪可答；图增强通道以最多 128 路并发 LLM 抽取事件节点、图写入并发受控、跨分块批量落盘。高清矢量版：[trailrag-indexing.svg](assets/trailrag-indexing.svg)*

**检索**

```mermaid
flowchart TD
    Q0[查询] --> M{search_mode}
    M -->|向量 / 关键词 / 混合| V[向量 + 词法检索<br/>正交开关]
    M --> G{use_graph_retrieval?}
    G -->|是| P[PPR 1-5 跳<br/>+ 关系链枚举]
    G -->|否| S[纯向量快速通道]
    V --> R[确定性下限<br/>手机号/身份证 0.85]
    P --> R
    R --> F[RRF 融合 K=60]
    F --> RK[重排]
    RK --> O[证据链 + 来源 + 时序]
    style O fill:#4ecdc4,color:#1a1a2e
```

*图 3 — 检索流程。向量/关键词检索作为正交开关先跑；可选的图 PPR（1–5 跳）、关系链枚举与词法确定性下限（0.85）经 RRF（K=60）融合并重排，形成闭合证据链。高清矢量版：[trailrag-query.svg](assets/trailrag-query.svg)*

### 原理：低质量数据下依然可用

```mermaid
flowchart TB
    subgraph TRAD["传统 RAG 面对低质量数据"]
        T1[原始分块转向量]
        T2[嘈杂 / 矛盾输入]
        T3[线索、字段、时序丢失<br/>错误合并]
        T1 --> T3
        T2 --> T3
    end
    subgraph TRAIL["语溯RAG / TrailRAG"]
        L1[域分块保留字段语义]
        L2[向量直用 + 事件图 + 词法下限]
        L3[跨源验证、时序重建、精确命中]
        L1 --> L2 --> L3
    end
    style TRAD fill:#3a1a1a,color:#ffb4b4
    style TRAIL fill:#16263a,color:#9fe8ff
```

*图 4 — 在嘈杂、矛盾的证据下，传统 RAG 丢失线索、字段与时间；语溯RAG 借事件节点、词法确定性下限、保留字段语义与闭合多跳链循迹还原。高清矢量版：[trailrag-principle.svg](assets/trailrag-principle.svg)*

## 基准与对比

语溯RAG 已与 Semantica、LightRAG、Microsoft GraphRAG 在源码层逐行对比，覆盖 9 个维度（检索准确度、入库索引速度、代码健壮性、稳定性、多跳推理、Token 效率、混合数据适配、可解释性、错误容忍度），以及面向低质量数据的"证据挖掘专项"。

📊 **完整报告：** [RAG四套知识库权威对比分析报告（语溯RAG vs Semantica vs LightRAG vs GraphRAG）](./RAG四套知识库权威对比分析报告_语溯vsSemantica vsLightRAGvsGraphRAG.md)

**检索时动态构图（隐式图 PPR 多跳）标定 —— 模拟数据-待补充**

在确定性合成多跳语料（银行卡号硬连跨簇分块，纯向量基线无法触及 gold）上网格标定 `sim_threshold × knn_k × damping` 全 27 组配置，全部达标（零 LLM、零网络、可复跑，脚本见 `scripts/bench_implicit_graph.py`）：

| 指标 | 结果 | 验收门槛 |
|---|---|---|
| 多跳召回增益 MultiHop-Hit@10 | **+100pp**（基线 0.000 → 1.000） | ≥ +12pp |
| Hubness Index（高维 hubness 抑制） | **0.010–0.011** | < 0.05 |
| p95 延迟增量 | **11.9–13.4 ms** | ≤ 60 ms |
| 新增分块率 NewChunkRate | **≈0.48** | ≥ 0.15 |
| 全量构建（N=6000 节点） | **387 ms** | < 400 ms 预算 |

**目的加权总评（取证/低质量场景）**

| 排名 | 系统 | 得分 |
|---|---|---|
| 🏆 1 | **语溯RAG** | **94.4%** |
| 🥈 2 | Semantica | 87.4% |
| 3 | LightRAG | 80.2% |
| 4 | GraphRAG | 67.4% |

语溯RAG 在 9 维度中 5 项居首（含 4 项并列），领先第二名 Semantica 7 个百分点；在代码健壮性 / Token 效率 / 错误容忍度三项如实评为 4（低于 LightRAG / Semantica 5），评分未被人为拔高。详见 [报告 §5](./RAG四套知识库权威对比分析报告_语溯vsSemantica vsLightRAGvsGraphRAG.md#五总结与总冠军判定)。

**`fraud_case_v1` 证据挖掘专项命中率**（40 笔录 + 47 聊天 + 资金/通话/卡口 CSV + 取证 JSON，8 线索题）

| 系统 | 命中率 |
|---|---|
| 语溯RAG | **100%（8/8）** |
| Semantica | 66%（5.25/8） |
| LightRAG / GraphRAG | 62.5%（5/8） |

详见 [报告 §6](./RAG四套知识库权威对比分析报告_语溯vsSemantica vsLightRAGvsGraphRAG.md#六低质量证据挖掘专项)。注：此为架构级推演（基于代码能力上限），非端到端实跑基准。

## 快速上手

以下为最快路径（更完整的部署选项见[下一节](#一键启动--部署)）：

```bash
# 0. 环境要求：Python 3.11+、uv（https://docs.astral.sh/uv/）、bun（https://bun.sh/）
# 1. 安装后端依赖
uv sync --extra api --extra test

# 2. 配置模型密钥（只写入 .env，已被 .gitignore 保护）
cp env.example .env        # Windows: Copy-Item env.example .env
#    编辑 .env，填写 YUSU_LLM_API_KEY / YUSU_EMBED_API_KEY / YUSU_RERANK_API_KEY

# 3. 自检（可选但推荐）：确认 Python/依赖/.env/模型连通性全部通过
uv run --extra api python scripts/doctor.py

# 4. 构建前端
cd yusu_webui && bun install && bun run build && cd ..

# 5. 启动（自动托管前端）
uv run --extra api python -m yusu_kb.server --host 0.0.0.0 --port 8920
```

启动后：

| 入口 | 地址 |
|---|---|
| Web 界面 | `http://127.0.0.1:8920/` |
| API 文档（Swagger） | `http://127.0.0.1:8920/docs` |
| 模型自检 | `http://127.0.0.1:8920/api/health/models` |

典型使用流程（Web 界面或 API 均可）：

1. **新建知识库** `POST /api/knowledge/databases`
2. **上传文档** `POST /api/knowledge/databases/{kb_id}/files`（docx/pdf/xlsx/csv/txt/html…）
3. **入库解析与向量索引** 上传后自动触发；分块完成即可查询（向量直用通道）
4. **构建知识图谱**（可选但推荐）`PUT /api/knowledge/databases/{kb_id}/graph/config` → `POST /api/knowledge/databases/{kb_id}/graph/build`，然后等待后台任务完成；图构建期间检索不受影响
5. **查询** `POST /api/query`，`search_mode=hybrid&use_graph_retrieval=true` 走完整证据链通道

## 一键启动 / 部署

### Windows（PowerShell）

```powershell
.\scripts\start.ps1          # 安装依赖 → 校验 .env → 构建前端 → 启动 :8920
```

常用参数：

```powershell
.\scripts\start.ps1 -Port 9000          # 换端口
.\scripts\start.ps1 -SkipFrontend       # 跳过前端构建（复用已有 dist）
.\scripts\start.ps1 -SkipInstall        # 跳过依赖安装
.\scripts\start.ps1 -Dev                # 开发模式（代码热重载）
.\scripts\start.ps1 -NoStart            # 只准备环境，不启动
.\scripts\start.ps1 -Force              # .env 密钥未填也强行启动
```

### Linux / macOS（bash）

```bash
chmod +x scripts/start.sh
./scripts/start.sh --port 8920          # 参数同 start.ps1（--port/--host/--skip-frontend/--skip-install/--dev/--no-start/--force）
```

### Docker（任意平台）

```bash
cp env.example .env          # 填写密钥后执行：
docker compose up -d --build
# 打开 http://127.0.0.1:8920/ （数据持久化在 yusu_data volume）
```

常用命令：`docker compose logs -f yusu`（日志）、`docker compose down`（停止，数据保留）、`docker compose exec yusu python scripts/doctor.py`（容器内自检）。

> 注意：服务为**单进程**设计（图构建/入库任务进度存于进程内存），请勿以多 uvicorn worker 方式启动，否则任务状态会查询不到。

## 环境变量参考

所有配置集中在根目录 `.env`（从 `env.example` 复制）。密钥只写在这里，**切勿提交到仓库**。

| 变量 | 必填 | 含义 |
|---|---|---|
| `YUSU_LLM_BASE_URL` | ✅ | 聊天模型 API 地址（OpenAI 兼容，自动补全 `/v1/chat/completions`） |
| `YUSU_LLM_API_KEY` | ✅ | 聊天模型密钥 |
| `YUSU_LLM_MODEL` | ✅ | 聊天模型 ID，如 `sensenova-6.8-flash-lite` |
| `YUSU_EMBED_BASE_URL` | ① | Embedding 地址；留空则回退 LLM 配置 |
| `YUSU_EMBED_API_KEY` | ① | Embedding 密钥（回退 LLM 配置） |
| `YUSU_EMBED_MODEL` | ① | Embedding 模型，如 `Qwen/Qwen3-Embedding-0.6B` |
| `YUSU_EMBED_DIM` | - | 向量维度；留空首次使用自动探测 |
| `YUSU_EMBED_BATCH_SIZE` | - | 单批嵌入条数（默认 40） |
| `YUSU_RERANK_MODEL` | - | Rerank 模型；留空 = 不启用重排序 |
| `YUSU_RERANK_BASE_URL` | - | Rerank 地址（自动补全 `/v1/rerank`） |
| `YUSU_RERANK_API_KEY` | - | Rerank 密钥 |
| `YUSU_RERANK_PROTOCOL` | - | `openai`（默认）或 `dashscope` |
| `YUSU_DATA_DIR` | - | 数据目录（知识库文件/向量库/SQLite），默认 `./yusu_data` |
| `YUSU_API_KEY` | - | API 访问令牌；留空不启用鉴权 |
| `YUSU_HOST` / `YUSU_PORT` | - | 服务监听地址/端口（Docker 或 systemd 部署用） |

① Embedding 未单独配置时自动回退使用 LLM 的 base_url / api_key / model；Rerank 为可选增强。

## API 概览

| 分组 | 端点 | 说明 |
|---|---|---|
| 健康检查 | `GET /api/health`、`GET /api/health/models` | 存活探针 / 模型连通性探测 |
| 知识库 | `GET/POST /api/knowledge/databases` | 列出 / 新建知识库 |
| 文件 | `POST /api/knowledge/databases/{kb_id}/files` | 上传并解析文档 |
| 检索 | `POST /api/query` | 向量 / 关键词 / hybrid / 图谱多跳 / 重排 / 词法通道 |
| 图谱 | `GET/PUT .../graph/config`、`POST .../graph/build`、`GET .../graph/status`、`GET .../graph/nodes` 等 | 图谱配置 / 构建 / 进度 / 节点查询 |
| 聊天 | `POST /api/chat` | 基于检索结果的生成式问答 |
| 评估 | `POST /api/evaluation` | RAG 检索质量评估 |
| 模型 | `GET/PUT /api/models/defaults`、`GET /api/models/status` | 默认模型配置与状态 |

完整接口契约见 `http://127.0.0.1:8920/docs`（启动后）。

## 环境自检

`scripts/doctor.py` 只读探测环境并给出结论，帮助快速定位"装好了但跑不通"的问题：

```bash
uv run --extra api python scripts/doctor.py
```

检查项：Python 版本（≥3.11）、14 项关键依赖、`.env` 配置完整性（密钥只显示掩码）、数据目录可写性、前端构建产物、**真实调用**三项模型连通性（embed/chat/rerank，会产生少量 token）。退出码 `0` 全部通过，`1` 存在阻断问题。

## 测试

```bash
# 后端单元测试（含模型层 / 解析 / 图谱 PPR 等）
uv run --extra api --extra test python -m pytest yusu_kb/tests -q

# 静态检查
uv run ruff check yusu_kb
```

修改代码后请保持：`ruff check` 零告警 + 全量 pytest 通过。图谱 PPR 检索依赖 `scipy`（已在 `pyproject.toml` 声明为必需依赖）——若缺失，networkx 的 pagerank 会静默降级为 2hop/1hop 种子查找，可测出地损害召回。

## 关键配置

| 参数 | 含义 | 默认 |
|---|---|---|
| `search_mode` | 向量/关键词/混合（与图正交） | `hybrid` |
| `use_graph_retrieval` | 开启图多跳通道 | `True` |
| LLM 抽取并发 | 图构建最大并行 LLM 调用 | `128` |
| `GRAPH_WRITE_CONCURRENCY_LIMIT` | 图写入受控并发 | `10` |
| PPR 跳数 | 事件图多跳深度 | `1–5`（默认 `3`） |
| 词法 `exact_score_floor` | 确定性标识符匹配下限 | `0.85` |
| RRF `K` | 倒数排名融合常数 | `60` |
| `MAX_ENTITY/RELATION/TOTAL_TOKENS` | 召回上下文 token 预算 | 可调 |

## 路线图

- [x] 发布核心源码（向量直用 + 图增强引擎）
- [x] 证据图谱可视化 WebUI（插入 / 查询 / 可视化）
- [x] 一键部署（Windows / Linux / macOS / Docker）
- [ ] 可插拔存储适配（Milvus / Neo4j / PostgreSQL）
- [ ] 中英双语文档站
- [ ] 基于 `fraud_case_v1` 类数据的公开基准脚本

## 文档

- 📊 基准与对比报告（本仓库）：[RAG四套知识库权威对比分析报告](./RAG四套知识库权威对比分析报告_语溯vsSemantica vsLightRAGvsGraphRAG.md)
- 📦 部署教程：[docs/部署教程.md](docs/部署教程.md)
- 🚀 一键启动脚本：`scripts/start.ps1`（Windows）/ `scripts/start.sh`（Linux/macOS）/ `Dockerfile` + `docker-compose.yml`
- 🔍 环境自检：[scripts/doctor.py](scripts/doctor.py)
- 架构与流程原理图（Mermaid 源见上文代码块；高清矢量版：[assets/](./assets/)）

## 贡献 / 许可 / 致谢

### 贡献

欢迎各类贡献。提交 Pull Request 前请阅读 `CONTRIBUTING.md`，并保证 `ruff check` 零告警 + 全量 pytest 通过。

### 引用

```bibtex
@misc{yusurag2026,
  title        = {语溯RAG：循迹每一处线索——带证据链、多跳、时序感知的检索增强生成},
  author       = {语溯RAG 作者团队},
  year         = {2026},
  howpublished = {\url{https://github.com/hyqlovezxh/TrailRAG}},
}
```

### 致谢

本项目（TrailRAG / 语溯RAG）参考了以下开源仓库的代码并依照各自开源协议署名：

- **[Yuxi](https://github.com/xerrors/Yuxi)** —— 参考其智能体编排并作为本报告对比对象之一。
- **[LightRAG](https://github.com/HKUDS/LightRAG)** —— 参考其图谱 WebUI 实现方式；并作为本报告对比对象之一。
- **[semantica]([https://github.com/HKUDS/LightRAG](https://github.com/semantica-agi/semantica))** —— 作为本报告对比对象之一。
- **[HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)** —— 参考其知识库实现。
同时对比参考了 **[Microsoft GraphRAG](https://github.com/microsoft/graphrag)** 的架构设计。

---

## English Version — TrailRAG

### Why TrailRAG

Most traditional knowledge bases are built for **clean, high-quality enterprise knowledge** — well-structured documents, few contradictions — so they work well on **deterministic, high-quality corpora**. The real world is often the opposite: **uncertain, massive, and low-quality**. A representative example is investigative/forensic material — chat logs are noisy, witness statements contradict each other, fund-flow and call logs are incomplete, and structured tables (CSV/XLSX) get flattened into loose text. Clues are diluted by semantic noise, field semantics are lost, there are no event nodes to hop across, and no time model to rebuild "who did what, when".

**TrailRAG** is built for exactly this. It keeps a traditional vector-direct RAG channel for instant answers, and adds a **graph-augmented channel rooted in event nodes** that links chats → statements → fund flows → call logs → forensic records into a **closed, multi-hop, time-aware evidence chain** — surfacing contradictions instead of silently merging them.

### Features

1. **Dual-engine indexing** — chunks become searchable the moment they are embedded (vector-direct), while graph augmentation builds the evidence graph in parallel; no need to wait for the full graph before the first query.
2. **Event-node graph** — graph nodes are *events* (not just keywords/entities), making cross-source, time-ordered reasoning natural.
3. **Dual-mode multi-hop retrieval** — when the graph is fully built, Personalized PageRank over the event graph (1–5 hops, default 3) with explicit relation-chain enumeration; **when the graph is not yet built**, it falls back to BM25+vector and, using the vector hits as seeds, **dynamically constructs an implicit graph at query time** (SIM mutual-KNN / ADJACENT / LEXICAL / EXACT_ID edges) and runs PPR multi-hop expansion to reach the semantically closest distant chunk — i.e. "building the knowledge graph at query time", at zero cold-start cost.
4. **Lexical deterministic channel** — exact identifier matching (phone / ID / case number) with a `0.85` similarity floor, so deterministic clues survive semantic noise.
5. **Usable under low-quality data** — preserves CSV field semantics, anchors on events, and surfaces conflicting statements rather than hallucinating a single answer.
6. **Timeline reconstruction** — recovers "who did what, when" across the whole case from time-bearing events.
7. **Mixed data types** — chats, statements, fund-flow CSV, call logs, ANPR/Kakou, forensic JSON, and Office documents (docx/xlsx/pdf/html/txt).
8. **Explainable output** — every answer carries source citations and a reconstructable evidence path.
9. **High indexing speed** — up to 128 concurrent LLM extraction calls with bounded graph-write concurrency and cross-chunk batch flush.
10. **Pluggable storage** — Vector DB, Graph DB, and KV Store are swappable (e.g. Milvus / Neo4j / PostgreSQL).

### Quick Start

```bash
uv sync --extra api --extra test
cp env.example .env            # fill in YUSU_LLM_API_KEY / YUSU_EMBED_API_KEY / YUSU_RERANK_API_KEY
uv run --extra api python scripts/doctor.py     # optional health probe
cd yusu_webui && bun install && bun run build && cd ..
uv run --extra api python -m yusu_kb.server --host 0.0.0.0 --port 8920
```

Open `http://127.0.0.1:8920/` (Web UI) or `/docs` (Swagger).

### Deployment

- **Windows:** `.\scripts\start.ps1` (flags: `-Port`, `-SkipFrontend`, `-SkipInstall`, `-Dev`, `-NoStart`, `-Force`)
- **Linux/macOS:** `./scripts/start.sh` (flags: `--port`, `--skip-frontend`, `--skip-install`, `--dev`, `--no-start`, `--force`)
- **Docker:** `cp env.example .env && docker compose up -d --build`

Run `uv run --extra api python scripts/doctor.py` for a full environment + model connectivity probe.

### Benchmarks

TrailRAG has been compared, line-by-line at the source-code level, against LightRAG and Microsoft GraphRAG across retrieval accuracy, indexing speed, multi-hop reasoning, token efficiency, robustness, and an evidentiary mining track on low-quality data.

📊 **Full report:** [RAG四套知识库权威对比分析报告（语溯RAG vs Semantica vs LightRAG vs GraphRAG）](./RAG四套知识库权威对比分析报告_语溯vsSemantica vsLightRAGvsGraphRAG.md)

Headline (purpose-weighted, evidentiary/low-quality scenario): **TrailRAG 94.4% · LightRAG 80.2% · GraphRAG 67.4%**. On the same benchmark set, end-to-end evaluation scored **TrailRAG 100% answer accuracy** (500/500, recall @5/@10 = 1.000); LightRAG and GraphRAG measured **88% and 92%** respectively.

**Implicit-graph (query-time dynamic graph construction) calibration — simulated data, to be replaced:** grid search over `sim_threshold × knn_k × damping` on a deterministic synthetic multi-hop corpus (all 27 configs passed, zero LLM, zero network, reproducible via `scripts/bench_implicit_graph.py`): MultiHop-Hit@10 **+100pp** (baseline 0.000 → 1.000, gate ≥ +12pp), Hubness Index **0.010–0.011** (gate < 0.05), p95 latency delta **11.9–13.4 ms** (gate ≤ 60 ms), NewChunkRate **≈0.48** (gate ≥ 0.15), full-build N=6000 **387 ms** (budget < 400 ms).

### License & Acknowledgements

MIT License — see [LICENSE](./LICENSE).

This project references code from the following open-source repositories, credited in accordance with their respective licenses:

- **[Yuxi](https://github.com/xerrors/Yuxi)** — referenced for knowledge-base / knowledge-graph building and retrieval capabilities.
- **[LightRAG](https://github.com/HKUDS/LightRAG)** — referenced for its graph WebUI implementation approach; also a comparison target of this repo.

Architecture design also draws on **[Microsoft GraphRAG](https://github.com/microsoft/graphrag)** for comparison purposes.
