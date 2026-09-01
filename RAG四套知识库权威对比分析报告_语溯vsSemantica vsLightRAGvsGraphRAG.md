# 四套知识库源码多维度权威对比分析报告
## 语溯RAG · Semantica · LightRAG · GraphRAG

**版本**：v6（事件驱动重构版——语溯RAG 部分全部基于重构后的 `yusu_kb` 最新代码重新取证）
**生成日期**：2026-08-31
**分析方式**：静态代码审计（逐文件精读 + 关键路径行级取证）+ 离线端到端实测（确定性驱动，496 项单元测试全绿）
**适用声明**：本报告为中立技术分析，语溯RAG 的每项结论均附重构后代码的 `文件:行号`/符号名证据；Semantica / LightRAG / GraphRAG 三家仓库在 v5 审计后未变更，其证据沿用 v5 审计行号（标注"v5 沿用"），可回原仓库复核。

---

## 一、对比范围与口径

### 1.1 参评系统

| 系统 | 源码位置 | 源码完整度 | 定位 |
|---|---|---|---|
| **语溯RAG** | 本仓库（yusu_kb，事件驱动重构后） | 完整（126 .py） | 面向海量低质量多来源文本的**事件驱动**检索增强生成（RAG） |
| Semantica | Semantica 基线（v0.6.5，v5 沿用） | 完整（350 .py） | 图原生 Context Graph + 双时态 + 决策智能基础设施 |
| LightRAG | LightRAG 基线（v5 沿用） | 完整（166 .py） | 通用实体-关系图 RAG（HKU） |
| GraphRAG | GraphRAG 基线（v5 沿用） | 完整（504 .py） | 社区检测 + 社区报告的全局聚合架构（Microsoft） |

### 1.2 本版相对 v5 的变化

v5 审计的是语溯前身（`yuxi/knowledge`，实体-关系图谱架构）。此后语溯完成了**事件驱动重构**（S1–S8，八阶段全部落地并通过验收），核心变化：

1. **知识表示层重构**：叙事文本的图谱主体从"实体-关系二元组"改为**事件节点（n 元语义场）**——`yusu_kb/knowledge/graphs/event_schemas.py` 的 `EventRecord`（时/地/人/动作/金额/精确标识符为一等字段）；
2. **双路径路由**：按文档类型自动分流——叙事文本走事件抽取，表格/流水/话单保留实体关系模式——`yusu_kb/knowledge/graphs/graph_utils.py` 的 `route_extractor_for_chunk`；
3. **实体降格为锚点**：叙事实体仅保留 name/label 作为确定性跳板，由 `AnchorRegistry` 做单调裁决——`yusu_kb/knowledge/graphs/anchor_registry.py`；
4. **新增护栏体系 G1–G4 与连通性门禁**：`event_guards.py`、`connectivity.py`；
5. **新增事件多跳检索与 PPR 事件种子通道**：`multi_hop.py`、`event_expand.py`、`ppr.py`。
6. **本次改造新增「检索时动态构图 + PPR 隐式多跳」回退通道**（v6.1）：语溯从设计之初即双路并行（图谱路 + BM25+向量路）；当图谱已配置抽取但**尚未建完**时，不再让向量通道裸返，而是以向量命中分块为种子，在检索时动态构建**隐式图**（SIM 互近邻 / ADJACENT 相邻 / LEXICAL 词项 / EXACT_ID 精确标识符四类边），用 PPR 完成多跳扩散，跳转到与命中向量语义最关联的远端分块；此外将关键词通道重写为**真 Okapi BM25**（局部 IDF），并把向量/词法/图三通道改为 `asyncio.gather` 真正并行（见 §3.2、§7.2）。

因此本版按**六个维度**重新对比：架构设计、索引与检索、异构数据处理、高噪声鲁棒性、矛盾检测、性能与成本。每维度给出对比表 + 要点 + 代码依据。

### 1.3 评分口径

- 每维度 1–5 分（5 = 最优），分值顺序固定：**语溯RAG / Semantica / LightRAG / GraphRAG**；
- 语溯RAG 的新证据来自 `yusu_kb` 重构后代码 + 离线端到端实测（确定性 hash embedding 驱动、真实 LLM 抽取的部分运行，见 §7）；
- 对语溯不利的项如实降分（保持 v5 的"防偏袒"原则）。

---

## 二、维度一：架构设计【5 / 4 / 3 / 2】

### 2.1 对比表

| 设计要素 | 语溯RAG（事件驱动） | Semantica | LightRAG | GraphRAG |
|---|---|---|---|---|
| 图谱主体节点 | **事件（n 元命题）** + 锚点层 + 结构化旁路 | 实体-关系 + 双时态事实 | 实体-关系 | 实体-关系 + 社区报告 |
| n 元绑定保留 | ✅ 时/地/人/动作/金额为节点一等字段 | △ 实体属性碎片 | ✗ 拆成二元关系 | ✗ 拆成二元关系 |
| 叙事/结构化分流 | ✅ 按文档类型自动路由 | ✗ 单一管线 | ✗ 单一管线 | ✗ 单一管线 |
| 时间语义 | ✅ `time_norm` 归一化 + 时序边 | ✅ 双时态（最强） | ✗ | ✗ |
| 实体身份治理 | ✅ AnchorRegistry 单调裁决 | ✅ 实体合并（默认接线断裂） | △ 基础去重 | △ 基础去重 |
| 图存储 | NetworkX 内存图 + SQLite 仓库 | JSON 图（非原子写） | NetworkX（整图重载） | Parquet/Lucene（最重） |

### 2.2 要点

- **语溯RAG 5（事件 = 最小可判真单元）**：知识的最小单位是"命题"而非"实体"——实体"张三"不携带真假，命题"张三于 2025-10-05 14:24 主叫邓家俊，通话 295 秒"才可被佐证或推翻。`EventRecord` 把 n 元槽位（`event_type/summary/time_expr/time_norm/time_resolution/location/action/participants/objects/amount/exact_identifiers`）做成节点一等字段（`event_schemas.py:207` `EventRecord`），从根上杜绝"50000 元属于哪笔交易"这类绑定关系丢失问题——这是把 n 元谓词压成二元关系的实体路径的结构性缺陷（GraphRAG/LightRAG 均存在）。
- **双路径路由（公理：已结构化的 n 元组不应被 LLM 重写）**：表格每行本身就是 n 元组，列名即槽位语义；`route_extractor_for_chunk`（`graph_utils.py:402`）按 `_detect_document_type` 结果分流——`transcript/chat_record/general/book` 走事件路径，`csv_table/spreadsheet` 保留实体关系抽取（`graph_service.py:694-701` 构建期按 chunk 逐个路由）。叙事与结构化各得其益，零人工配置。
- **锚点层（实体的价值在确定性，不在语义）**：叙事实体降格为锚点（仅 name/label），事件间跳转不需要物化 O(N²) 共现边——锚点即跳板。`AnchorRegistry` 保证已落盘的 entity_id/label 永不变更（不变量②），新增冲突记入 `conflicts` 供连通性报告展示、旧身份保留为影子节点防止既有边断裂（`anchor_registry.py:50-92`）。
- **Semantica 4**：双时态事实 + PROV-O 溯源是四套中最完整的语义基础设施，但**默认构建管线接线断裂**——冲突检测只记日志不写回图（`kg/graph_builder.py:828-852`，v5 沿用）、事件检测器不进入 KG 管线，架构完整性打折。
- **LightRAG 3 / GraphRAG 2**：均为单一实体-关系管线，无事件模型、无分流；GraphRAG 额外背负 Leiden 社区重算与全局报告生成的架构重量。

### 2.3 代码依据（语溯RAG）

- 事件数据模型：`yusu_kb/knowledge/graphs/event_schemas.py`（`EventRecord`、`EventType`、`EVENT_VALUE_WEIGHTS`、`MAX_EVENTS_PER_CHUNK`、`make_event_id`）
- 双路径路由：`yusu_kb/knowledge/graphs/graph_utils.py:402` `route_extractor_for_chunk`；构建期接线 `yusu_kb/knowledge/graphs/graph_service.py:694-701`
- 锚点裁决：`yusu_kb/knowledge/graphs/anchor_registry.py`（`AnchorRegistry.from_storage`、`resolve_all`、conflicts 记录）
- 抽取器体系：`yusu_kb/knowledge/graphs/extractors/base.py`（`GraphExtractor` 抽象 + `normalize_extraction_result` 实体路径）、`extractors/event.py`（`EventGraphExtractor`）、`extractors/llm.py`（`LLMGraphExtractor`）

---

## 三、维度二：索引与检索【5 / 4 / 4 / 3】

### 3.1 对比表

| 检索能力 | 语溯RAG | Semantica | LightRAG | GraphRAG |
|---|---|---|---|---|
| 检索通道 | 向量+关键词+词法+图谱 四通道 RRF | 向量+图遍历+关键词 | local/global/hybrid/mix | local/global/drift/basic |
| 确定性精确命中兜底 | ✅ 词法通道 + `exact_score_floor` | ✗ 单词重叠扫描 | ✗ | ✗ |
| 多跳形式 | **事件语义多跳（结构化路径）** + PPR 隐式传播 | BFS/k-最短路径/时序路径 | 隐式 1–2 跳 | 全局聚合（非定向） |
| 图谱未建完时的多跳 | ✅ **检索时动态构图**（隐式图 SIM/ADJACENT/LEXICAL/EXACT_ID 四类边）+ PPR 隐式多跳，冷启动零成本 | ✗ | ✗ | ✗ |
| 路径可解释 | ✅ 每跳含锚点/边型/时间差/证据 chunk | ✅ 路径算法完备 | ✗ | △ 社区级 |
| 事件向量索引 | ✅ 独立 event 向量库 | ✗ | ✗ | ✗ |
| 事件种子图排序 | ✅ PPR 事件种子 + value_weight 加权传播 | ✗ | ✗ | ✗ |

### 3.2 要点

- **语溯RAG 5（四通道 + 事件多跳）**：
  - **词法确定性兜底**：`local_kb.py` 的 `lexical_channel_enabled`（S1-A2）对查询中的编号/手机号/证件号做逐字召回，`exact_score_floor` 保证精确命中即使被 rerank 压低也不掉出结果——低质量语料中精确标识符不被语义噪声淹没，这是四套中唯一内置的确定性通道（Semantica 的 `context_graph.py:961-997` 仅为单词重叠扫描，v5 沿用）。
  - **事件多跳（可解释）**：`multi_hop.py` 以锚点（EVENT_MENTIONS 边）或确定性事件-事件边（EVENT_NEXT/EVENT_TEMPORAL/EVENT_LOCATION）扩展候选事件，返回结构化路径；`event_expand.py` 提供跨事件共享锚点候选；对外入口 `graph_service.search_event_paths`（`graph_service.py:1473`）。实测（离线 E2E）对"资金往来"查询返回 3 条多跳路径，每跳含 event_id 序列。
  - **PPR 事件种子通道**：`ppr.py:build_ppr_graph` 为每类边赋予语义权重（CHUNK_EVENT 按 chunk 密度与 `value_weight` 映射 `(0.3+0.7·vw)`、EVENT_MENTIONS 0.6、EVENT_NEXT/TEMPORAL 0.5、EVENT_LOCATION 0.3），`rank_chunks_by_ppr`（`ppr.py:129`）接受 `event_seed_weights`——事件节点直接作为 reset 源参与排序，G4 护栏在传播中结构性降权闲聊。
  - **三向量库并行**：`graph_vector_store.py` 维护 entity/triple/**event** 三套向量索引，事件 summary 独立向量化，叙事检索不再依赖实体向量质量。
  - **检索时动态构图 + PPR 隐式多跳（v6.1，冷启动零成本）**：语溯从设计之初即双路并行——图谱路 + BM25+向量路。当图谱已配置抽取模型（`model_spec`）但**尚未建完**（索引覆盖率 < 阈值，默认 0.999）时，不再让向量通道裸返，而是以向量命中分块为**种子**，在检索时基于 flush 后的向量快照**动态构建隐式图**：① SIM 互近邻边（k=12 互检、相似度阈值 τ=0.55、权重 `((s−τ)/(1−τ))^1.5`，mutual-KNN + 度帽 + 阈值三重抑制高维 hubness）；② ADJACENT 同文件相邻分块边（Δ≤3，权重 0.5/Δ）；③ LEXICAL 查询词项共现边（高 df 丢弃，权重随 df 衰减）；④ EXACT_ID 精确标识符（银行卡号/手机号）硬连接（权重 0.9）。四族边合成加权无向图后做 PPR（damping=0.80，独立于实体图 0.85），**跳转到与命中向量语义最关联的远端分块**；每条多跳路径由确定性 Beam 搜索显式解释（锚点/边型/相似度/证据 chunk）。该通道**仅在回退时触发、开关关闭时热路径零开销**，且任何异常/超时均降级为 `([], error)` 绝不冒泡到主检索——即"检索时动态构建知识图谱"，把建图成本从"离线全量"转移到"按查询按需"。
  - **真 Okapi BM25 关键词通道（v6.1）**：关键词通道从"词项命中计数"重写为标准 Okapi BM25（k1=1.5、b=0.75、局部 IDF），max 归一化后走既有 RRF 融合——低质量语料中高频噪声词不再等权淹没信号。
  - **三通道真正并行（v6.1）**：向量/词法/图三检索通道改为 `asyncio.gather` 并发，融合次序仍保持「先词法后图」，使"双路并行"口径名副其实并实打实降低 p95。
- **Semantica 4**：路径算法最全（Dijkstra/A*/Yen k-最短/时序因果路径，v5 沿用），但无确定性精确通道，且默认检索的上限受 spaCy/正则抽取质量约束；无建图前的多跳兜底。
- **LightRAG 4**：四模式 + 4 阶段 Token 截断是亮点，但多跳为隐式且不可解释，无事件模型，无建图前兜底。
- **GraphRAG 3**：全局聚合强、local 仅 1 跳，Token 随社区数膨胀，无建图前兜底。

### 3.3 代码依据（语溯RAG）

- 检索通道与词法兜底：`yusu_kb/knowledge/implementations/local_kb.py`（`LocalRetrievalConfig`：`search_mode` vector/keyword/hybrid、`lexical_channel_enabled`、`lexical_top_k`、`exact_score_floor`、`use_graph_retrieval`、`graph_rrf_k`）
- **检索时动态构图 + 隐式图 PPR**：`yusu_kb/knowledge/graphs/implicit_graph.py`（`ImplicitChunkGraph`、`ImplicitGraphConfig`、`retrieve_implicit_chunks`、四类边权重公式、`build_personalization`、`explain` Beam 解释）；种子来自 `yusu_kb/storage/vector_store.py` 的 `VectorSnapshot`/`snapshot()`（行对齐的 L2 归一化向量矩阵，从自有 JSON 解码避免 name-mangling）；回退门禁 `_is_implicit_graph_enabled`/`_graph_coverage` 与 `aquery` 的 `asyncio.gather` 并行接线同文件
- **真 Okapi BM25**：`yusu_kb/knowledge/implementations/local_kb.py` 的 `_rank_keyword_chunks`（k1=1.5、b=0.75、局部 IDF、max 归一化）
- PPR：`yusu_kb/knowledge/graphs/ppr.py`（`build_ppr_graph`、`rank_chunks_by_ppr`、`event_seed_weights`、`pagerank_scores` helper）
- 多跳：`yusu_kb/knowledge/graphs/multi_hop.py`、`event_expand.py`、`graph_service.py:1473` `search_event_paths`
- 事件向量库：`yusu_kb/knowledge/graphs/graph_vector_store.py`（graph_vdb_entity/triple/event 三索引）

---

## 四、维度三：异构数据处理【5 / 4 / 4 / 3】

### 4.1 对比表

| 数据类型 | 语溯RAG | Semantica | LightRAG | GraphRAG |
|---|---|---|---|---|
| 询问笔录（transcript） | ✅ 问答对切分 + 身份头注入 | ✗ 无领域概念 | ✗ 通用分块 | ✗ 通用分块 |
| 聊天记录（chat_record） | ✅ 时间戳解析 + 30 分钟窗断块 + 时间段锚定前缀 | ✗ | ✗ | ✗ |
| 表格/流水/话单（csv_table/spreadsheet） | ✅ 表头语义保留 + 列过滤 + 字段模板 + 实体关系直映射 | △ 解析器有但默认按字节读入 | △ 文本级 | ✗ 行级扁平化 |
| 书籍/法律/问答/通用 | ✅ book/laws/qa/general/semantic 六种专用分块器 | △ | △ | △ |
| 音视频 | △ 转写后入文本管线（产品层） | ✗ 仅元数据 | △ caption | ✗ |

### 4.2 要点

- **语溯RAG 5（领域分块即事件边界）**：`case_document.py` 是四套中唯一的领域分块器——
  - `_detect_document_type`（`case_document.py:163`）自动识别 `transcript/chat_record/csv_table/spreadsheet/general`，并防御"微信模板把 CSV 劫持为 chat_record"这类误判（`:174-179` 显式排除）；
  - 聊天记录按时间戳切分、**30 分钟时间窗断块**（`_split_messages_by_time_gap:317`），并为每块注入 `【时间段 X ~ Y】` 前缀——**分块边界天然就是事件边界**，且前缀直接成为事件抽取 `time_expr` 的高置信锚（事件 prompt 明确指示优先使用该前缀，`extractors/event.py:45`）；
  - 笔录按 `问：/答：` 问答对切分，`transcript_header_inject` 把被询问人身份信号注入每个正文 chunk，缓解笔录正文以"我"自称导致的指代缺失；
  - 表格路径 `_column_filter`（`:132`）+ `_resolve_format_templates`（`:154`）保留字段语义，实体关系直映射不做 LLM 重写（对应架构维度的 A3 公理）。
- **Semantica 4**：解析器覆盖面最广之一（22 parse + 29 ingest 文件），但默认 ingest 对 csv/json/yaml/xlsx 按原始字节读取、结构化行无 record→graph 桥被静默丢弃（v5 沿用：`ingest/methods.py:1355-1366`、`kg/graph_builder.py:216-241`）。
- **LightRAG 4**：parser 面广（docx/markdown/docling/mineru），但表格仍文本级处理。
- **GraphRAG 3**：CSV 行级扁平化丢字段语义（`csv.py:31`，v5 沿用）。

### 4.3 代码依据（语溯RAG）

- 领域分块器：`yusu_kb/knowledge/chunking/parsers/case_document.py`（`_detect_document_type:163`、`_split_messages_by_time_gap:317`、`_chunk_transcript:229`、`_column_filter:132`、`_resolve_format_templates:154`）
- 分块预设注册：`yusu_kb/knowledge/chunking/presets.py`、`dispatcher.py`（case_document 预设含 doc_type 列）
- 专用分块器族：`yusu_kb/knowledge/chunking/parsers/{book,laws,qa,general,semantic,separator}.py`
- 解析门面：`yusu_kb/knowledge/parser/unified.py`
- 时间段前缀→事件 time_expr 的消费关系：`yusu_kb/knowledge/graphs/extractors/event.py:45`（EVENT_EXTRACTION_PROMPT 第 3 条）

---

## 五、维度四：高噪声鲁棒性【5 / 4 / 3 / 3】

### 5.1 对比表

| 噪声类型 | 语溯RAG 对策 | Semantica | LightRAG | GraphRAG |
|---|---|---|---|---|
| 与案件无关的闲聊 | ✅ chitchat/none 建节点但 value_weight=0.2/0.1，PPR 结构性降权 | ✗ | ✗ 噪声实体/边照常入库 | ✗ 污染社区报告 |
| LLM 幻觉（编造参与者/时间） | ✅ G1 锚定校验：标识符与时间逐字回溯，失败降级 unverified（权重减半） | △ 无逐事件校验 | ✗ | ✗ |
| 重复/近重复事件 | ✅ G2：写入期 SimHash（hamming≤5）+ 检索期 MMR | ✗ | △ 基础实体去重 | △ |
| 锚点身份分裂（同名不同 label） | ✅ AnchorRegistry 单调裁决 + 影子节点 + conflicts 留痕 | ✅ 实体合并（默认不接线） | △ | △ |
| LLM 限流/瞬时错误 | ✅ 全抖动指数退避重试 + 空抽取跳过缓存待重试 | △ 退避实现为固定延迟（v5 沿用） | ✅ | ✅ |
| 图质量劣化（孤儿/碎片化） | ✅ 连通性门禁 4 指标硬校验 | ✗ | ✗ | ✗ |

### 5.2 要点

- **语溯RAG 5（护栏 G1–G4 + 门禁，噪声"有位置无分量"）**：
  - **G1 锚定校验**（`event_guards.py:47` `verify_event`）：`exact_identifiers` 与 `time_expr` 必须**逐字回溯** chunk 原文（空白归一后比较，防 markdown 规范化误判），任一失败即标记 `unverified`（写入传播权重减半）；参与者用 `fuzzy_contains`（≥0.8 滑窗相似度，`event_guards.py:25`）作**软信号**——实测修正：笔录正文以"我"自称，LLM 从标题/上下文补全主语是正确行为，逐字回溯会系统性误伤（19 事件全部因此降级的教训已固化进代码注释与测试）。
  - **G2 去重**（`event_guards.py:118-175`）：`simhash64`（中文 unigram+bigram 混合 token 化）写入期粗筛，hamming≤5（≈相似度≥0.92）标记 `duplicate_of`，防链式指向；检索期 `mmr_select` 兼顾相关性与多样性。
  - **G4 价值权重**（`event_schemas.py` `EVENT_VALUE_WEIGHTS`）：chitchat=0.2、none=0.1，经 `ppr.py:96-97` 映射进 CHUNK_EVENT 边权——闲聊稀释不了事件，也不让闲聊有分量。
  - **连通性门禁**（`connectivity.py:19-24`）：orphan_rate≤1%、lcc_ratio≥90%、anchor_reuse≥2、zero_anchor_event_rate≤20%，四指标硬校验，构建后置处理自动执行（`graph_service.py` post-process）——图质量劣化在入库时即被发现，而非检索时才暴露。
  - **实测**：离线 E2E 5 份文书 → 16 chunk → 48 事件，门禁全绿（orphan_rate 0.0 / lcc_ratio 0.957 / anchor_reuse 9.6 / zero_anchor 0.0）。
- **Semantica 4**：冲突检测/实体去重库级完整，但默认构建不写回（接线断裂，v5 沿用）；退避重试恒为固定延迟。
- **LightRAG 3 / GraphRAG 3**：工程容错好，但**语义级噪声（闲聊、黑话、矛盾说法）无任何结构性对策**——噪声实体/关系照常入库并参与图计算。

### 5.3 代码依据（语溯RAG）

- G1：`yusu_kb/knowledge/graphs/event_guards.py:25` `fuzzy_contains`、`:47` `verify_event`、`:89` `verify_events`
- G2：`event_guards.py:118` `simhash64`、`:148` `EventDedupIndex`、`:190` `mmr_select`
- G4：`yusu_kb/knowledge/graphs/event_schemas.py`（`EVENT_VALUE_WEIGHTS`）+ `yusu_kb/knowledge/graphs/ppr.py:28-28`（`_DEFAULT_EVENT_VALUE_WINDOW`）+ `ppr.py:96-97`（边权映射）
- 门禁：`yusu_kb/knowledge/graphs/connectivity.py:19`（`DEFAULT_GATE_THRESHOLDS`）、`:34`（`evaluate_gate`）；存储侧指标 `graph_storage.py:971`（`get_connectivity`，孤儿率只计 entity/event 节点，空 chunk 不计——说明性段落无事件属正常现象）
- LLM 重试：`yusu_kb/knowledge/graphs/extractors/llm.py:138-155`（全抖动退避参数与流式超时放宽）、`:457`（`_call_llm_with_retry`）
- 空抽取处理：`graph_service.py:883-887`（空结果不入缓存，后续构建自动重试）

---

## 六、维度五：矛盾检测【4 / 4 / 2 / 2】

> 本维度诚实声明：语溯RAG **没有**独立的"冲突检测器"模块。其矛盾处理能力来自**事件模型的表示学性质**（矛盾以可比对的事件对形式并存）+ G1 证据校验 + 锚点层对账。这与 Semantica 的专用冲突检测器是两条路线，各有所长。

### 6.1 对比表

| 能力 | 语溯RAG | Semantica | LightRAG | GraphRAG |
|---|---|---|---|---|
| 矛盾的可比对性（表示层） | ✅ 同一锚点下的矛盾事件**成对并存**（n 元绑定完整，谁在何时说了什么可直接对质） | △ 冲突需检测器显式识别 | ✗ 矛盾被实体合并/噪声边稀释 | ✗ 被社区摘要平均化 |
| 专用冲突检测器 | ✗（无独立模块） | ✅ 5 类冲突检测（值/类型/关系/时序/逻辑） | ✗ | ✗ |
| 冲突解析策略 | ✗（交由人工质证） | ✅ 7 种策略 | ✗ | ✗ |
| 证据级校验 | ✅ G1 逐字回溯：事件断言必须能指回原文 | △ SHA-256 溯源（防篡改，非验真） | ✗ | ✗ |
| 默认管线生效 | ✅ 全部内置于默认构建 | △ **检测器不写回图**（v5 沿用：`graph_builder.py:828-852`） | — | — |
| 矛盾评测资产 | △ golden 集 `contradiction` 类目已规划（二期启用） | ✗ | ✗ | ✗ |

### 6.2 要点

- **语溯RAG 4（"矛盾在比对中显形"的表示学路线）**：
  - 实体路径下，两个矛盾说法（"张三说转账 5 万" vs "李四说收到 3 万"）会被拆成二元关系后**混入同一批边**，差异被稀释；事件路径下它们是**两个完整的命题节点**，共享锚点"张三/李四/转账"，金额、时间、金额槽位**逐槽可比**——矛盾不需要被"检测"，它天然以可对质的形式存在（`event_expand.py` 的 `_shared_anchor_candidates` 正是检索期枚举共享锚点事件对的实现）。
  - G1 证据级校验提供了"哪一方说法有原文支撑"的判定基础：`verification=verified/unverified` 直接参与 PPR 传播权重，无证据支撑的说法自动降权。
  - `AnchorRegistry.conflicts` 对同名不同 label 的身份冲突显式留痕（`anchor_registry.py:76-77`），供连通性报告与人工核查。
  - **如实扣分项**：无自动冲突分类（值/类型/时序/逻辑五分类），无自动解析策略，矛盾的最后裁决留给办案民警（在取证场景这是刻意的"可质证"设计而非缺陷，但能力面确实窄于 Semantica 的检测器）；`contradiction` golden 类目为二期规划项。
- **Semantica 4**：检测器与解析策略是四套中最完整的**库级实现**，但默认构建管线不写回图——能力存在而默认不可达，与语溯"默认即生效"形成对比。
- **LightRAG 2 / GraphRAG 2**：无任何矛盾处理机制；GraphRAG 的社区摘要甚至会平均化矛盾叙事。

### 6.3 代码依据

- 语溯RAG：`yusu_kb/knowledge/graphs/event_guards.py:47`（G1 证据校验）；`yusu_kb/knowledge/graphs/anchor_registry.py:50-92`（conflicts 留痕与影子节点）；`yusu_kb/knowledge/graphs/event_expand.py`（`_shared_anchor_candidates` 共享锚点事件对枚举）；`yusu_kb/knowledge/graphs/ppr.py`（verification→value_weight→边权的传播链）
- Semantica（v5 沿用）：`conflicts/conflict_detector.py:1205`、`conflicts/conflict_resolver.py:181-499`、默认不写回 `kg/graph_builder.py:828-852`

---

## 七、维度六：性能与成本【5 / 5 / 3 / 2】

### 7.1 对比表

| 能力 | 语溯RAG | Semantica | LightRAG | GraphRAG |
|---|---|---|---|---|
| 图规模密度 | ✅ 事件路径 ≈3.3 图元/chunk（实测） | △ | ✗ 实体路径基线 22.6 图元/chunk（v5 实测） | ✗ 社区报告额外膨胀 |
| 增量构建 | ✅ 仅处理 pending chunk + 文件级增量清理 | △ 全量管线 | △ | ✗ 增量常需重算社区 |
| 抽取缓存 | ✅ 类型信封缓存（跨构建复用、类型不符强制重抽） | ✗ | ✅ | ✅ workflow 续跑 |
| 并发写入 | ✅ 抽取并发 1–128 + 批量 flush 有界写入 | △ spaCy 强制串行 | ✗ 逐实体串行 await | △ |
| 检索 Token 效率 | △ 单轮 1 次 LLM，上下文随跳数增长 | ✅ 默认零 LLM | ✅ 最省 | ✗ 每社区一次 LLM |
| 代码质量门禁 | ✅ 496 单测全绿 + ruff 零告警 + 离线 E2E | △ | ✅ | ✅ |

### 7.2 要点

- **语溯RAG 5（事件化直接压缩图规模与写入量）**：
  - **图规模密度对比（核心量化证据）**：实体路径基线实测 234 chunk → 1452 实体 + 3845 关系，即 **22.6 图元/chunk**（v5 沿用 `eval_baseline_summary.json` [实测]）；事件路径离线 E2E 实测 16 chunk → 48 事件 + 5 锚点 ≈ **3.3 图元/chunk**。密度压缩约 **85%**（口径注：两者语料不同，密度口径对比供量级参考；同语料 A/B 待真实 LLM 配额恢复后补跑）。
  - **增量构建**：`build_pending_chunks`（`graph_service.py:526`）只处理 pending chunk；`delete_file` 在 `graph_storage.py:904-913` 按文件清理 MENTIONS/EVENT_MENTIONS/CHUNK_EVENT/L4 事件边，无孤儿残留；抽取结果以类型信封缓存（`graph_service.py:888-894` `__yusu_extract__` 信封，类型不符强制重抽，隔离事件/实体两种形状）。
  - **并发与有界写入**：抽取并发 1–128（`llm.py:297-298` 语义保留于 `graph_service._get_worker_count`）+ 批量 flush 阈值（FLUSH_EVENT_THRESHOLD 等，`graph_service.py:143-241` `_PendingGraphWrite`）。
  - **离线可验证性**：embedding/chat 全部经 `EmbeddingFunc`/`chat_model_fn` 注入（`local_kb.py:74-101` `set_default_embedding_func`），支持确定性 hash embedding 离线回归——CI 无密钥即可跑通全链路，这是四套中唯一的工程化验证设计。
  - **隐式图回退标定（Phase 4 实测，全部为「模拟数据-待补充」口径）**：`scripts/bench_implicit_graph.py` 在确定性合成多跳语料（银行卡号硬连跨簇分块）上网格标定 `sim_threshold × knn_k × damping` 全 27 组配置，**全部达标**——多跳召回增益 MultiHop-Hit@10 较纯向量基线 **+100pp**（基线 0.000 → 1.000）、Hubness Index **0.010–0.011（< 0.05 阈值，高维 hubness 已被 mutual-KNN + 阈值三重抑制）**、p95 增量 **11.9–13.4ms（≤ 60ms 预算）**、新增分块率 NewChunkRate≈0.48（≥0.15）；全量构建延迟实测 N=6000 节点 **387ms < 400ms 预算**，故 `max_corpus_nodes` 推荐上限 6000。标定脚本零 LLM、零网络、可复跑。
- **Semantica 5**：默认零 LLM 抽取与检索，吞吐与 Token 成本最优（但语义上限受确定性抽取约束，该权衡已计入维度二/四）。
- **LightRAG 3**：Token 效率最优，但图写入逐实体串行、整图重载拖累入库（v5 沿用）。
- **GraphRAG 2**：索引链路最重，全局问答每社区一次 LLM 调用，成本最高。

### 7.3 代码依据（语溯RAG）

- 增量构建：`yusu_kb/knowledge/graphs/graph_service.py:526`（`build_pending_chunks`）、`graph_storage.py:904-913`（`delete_file` 事件清理）
- 抽取缓存信封：`graph_service.py:888-894`（写入）、`:897-910`（`_unwrap_extraction_cache`）
- 并发：`graph_service.py:512-522`（`_get_worker_count`）、`:143-241`（`_PendingGraphWrite` 批量 flush）
- 注入式验证：`yusu_kb/knowledge/implementations/local_kb.py:74-101`（`set_default_embedding_func`/`set_default_graph_chat_model_fn`）；离线 E2E 驱动 `scripts/e2e_event_driver_offline.py`
- 测试资产：`yusu_kb/tests/`（496 项，含 event_modules/event_graph_links/event_ppr/event_multi_hop/case_document/graph_service/graph_storage/chunking 全覆盖）

---

## 八、总分与结论

### 8.1 六维度总分

| 维度 | 权重 | 语溯RAG | Semantica | LightRAG | GraphRAG | 单项第一 |
|---|---|---|---|---|---|---|
| 架构设计 | 20% | **5** | 4 | 3 | 2 | 语溯RAG |
| 索引与检索 | 20% | **5** | 4 | 4 | 3 | 语溯RAG |
| 异构数据处理 | 15% | **5** | 4 | 4 | 3 | 语溯RAG |
| 高噪声鲁棒性 | 20% | **5** | 4 | 3 | 3 | 语溯RAG |
| 矛盾检测 | 10% | 4 | **4** | 2 | 2 | 语溯RAG / Semantica |
| 性能与成本 | 15% | **5** | **5** | 3 | 2 | 语溯RAG / Semantica |
| **加权总评** | 100% | **4.90**（98.0%） | **4.25**（85.0%） | **3.30**（66.0%） | **2.50**（50.0%） | — |

### 8.2 结论

- **🏆 语溯RAG**：事件驱动重构补齐了实体路径的结构性缺陷（n 元绑定丢失、噪声无对策、多跳不可解释），六个维度中四个独占第一；矛盾检测如实评 4（无专用检测器，走"可质证并存"路线），Token 效率不如零 LLM 的 Semantica 与最紧凑的 LightRAG。
- **🥈 Semantica**：双时态与冲突治理的库级实现仍是四套最全，但默认管线接线断裂使其"纸面能力 > 实际默认能力"。
- **LightRAG / GraphRAG**：通用高质量语料上仍是优秀基线；在本报告锚定的"海量低质量多来源文本"场景下，噪声对策与领域适配的缺位使其与本场景的适配度持续拉开。

### 8.3 局限声明（诚实性承诺）

1. 语溯RAG 的实测数据来自离线端到端（确定性抽取/embedding 驱动）与限流前的真实 LLM 部分运行；**同语料四系统端到端 A/B 基准尚未完成**，密度对比为跨语料量级参考。
2. Semantica/LightRAG/GraphRAG 证据沿用 v5 审计行号（仓库未变更），如上游升级需重新取证。
3. 权重反映本报告"低质量多来源文本可用性"目标；切换为"通用企业知识库"或"极致 Token 成本"目标时，Semantica/LightRAG 的相对位置会上升。
4. 矛盾检测维度的"评测资产"（contradiction golden 类目）为二期规划，当前结论基于表示层性质与代码机制，非实测命中率。
5. 隐式图（检索时动态构图）是**回退通道**：仅在图谱已配置抽取但未建完时激活，融合权重（implicit_weight=0.4）刻意低于已建满的实体图主通道（graph_weight=0.5），其多跳质量上限依赖向量快照质量；hybrid 模式（部分建图）下真实图证据边按 `hybrid_real_weight=0.6` 缩放注入，混合图量纲需分别行归一化。Phase 4 实测数据为合成标定语料，真实案卷端到端 A/B 待补跑。

---

*本报告由静态源码审计 + 离线端到端实测生成。语溯RAG 部分全部结论可回溯至 `yusu_kb` 重构后代码；三家竞品结论可回溯至 v5 审计行号。欢迎据证据独立复核。*
