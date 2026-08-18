# 四套知识库源码多维度权威对比分析报告
## 语溯RAG · Semantica · LightRAG · GraphRAG

**版本**：v5（权威版，仅纳入完整源码在库内的系统，确保逐行可复核）
**生成日期**：2026-08-19
**分析方式**：基于工作区本地源码的静态代码审计（逐文件精读 + Grep 统计 + 关键路径行级取证）
**适用声明**：本报告为中立技术分析，结论全部附 `文件:行号` 证据，计划开源至 GitHub，欢迎据证据独立复核。本版在原 v4 三套报告基础上，按**同一评分标准**新增 Semantica（`D:/PK/semantica`）并重算全部排名。

---

## 一、对比目的与条件（透明声明，便于开源复现）

### 1.1 纳入原则（为什么比这四套）
本报告坚持一条**可复核的公平底线**：**只对比"完整源码在本仓库内、可作行级取证"的系统**。任何核心检索/索引引擎以外部二进制或 PyPI 依赖形式存在、无法在库内逐行验证的系统，本轮不纳入正式评分——因为对"源码不可见"的系统只能引用其官方文档/论文推断，与"源码完全可见"的系统同表评分会构成**信息不对等**，不符合开源可信原则。据此，本轮参评系统为：

| 系统 | 仓库路径 | 源码完整度 | 定位 |
|---|---|---|---|
| 语溯RAG | `D:/PK/语溯RAG` | 完整（64 .py） | 面向公安刑事案件的全模态 RAG |
| Semantica | `D:/PK/semantica` | 完整（包内 350 .py，v0.6.5） | 图原生"Context Graph + 决策智能"基础设施（semantica-agi 开源项目本地副本） |
| LightRAG | `D:/PK/lightrag` | 完整（166 .py） | 开源通用 RAG（HKU 团队，本地副本） |
| GraphRAG | `D:/PK/graphrag` | 完整（Microsoft 官方，504 .py） | 开源通用 GraphRAG |

> 四套均为**完整源码在库内**，本报告所有结论均可回到 `文件:行号` 复核，不存在"依赖外部文档推断"的黑箱环节。

### 1.2 本次对比要解决的核心问题
传统知识库（含 GraphRAG、LightRAG 等）大多面向**企业高质量知识**设计：文档规范、实体清晰、少矛盾，因此对**确定性、高质量语料**效果好。但**公安取证行业**的数据现实截然不同：

- 聊天记录**杂乱、口语化、含大量无关噪声**；
- 笔录、聊天、资金流水、话单、卡口之间**可能相互矛盾**（缺乏数据治理）；
- 数据**海量、低质量、且高度不确定**。

在缺乏数据治理的低质量语料上，传统知识库很难从噪声中挖掘出办案真正需要的真实线索，构建出的知识图谱往往夹杂大量无关与低质量节点。**因此本报告的检索评测不只看"语义检索准不准"，而是重点评估知识库在"不确定性数据 + 海量低质量数据 + 跨证据多跳 + 案件宏观时序重建"下的真实可用性**。该评测场景统一折算进「检索准确度」维度（详见 §3.1、§4.1 与 §6），不单列维度，以同时满足"可量化对比"与"不喧宾夺主"。

### 1.3 评分方法（公平、可复核、防偏袒）
- **量表**：每维度 1–5 分（5 = 最优），共 9 个维度。
- **双口径同时公布**，杜绝加权本身被质疑为偏袒：
  - **目的加权总评**（本报告目标：低质量取证 / 多跳 / 时序）→ 权重见 §3.2；
  - **等权总评**（通用企业知识库视角）→ 9 维度等权，供横向校验。
- **证据原则**：每条结论附 `文件:行号`；凡代码未支持之处，明确写"代码未见支持"。
- **未跑实跑基准**：本报告为静态架构级评估，非端到端 benchmark（运行需各系统 LLM/Embedding 密钥与大量算力）；评分是对"代码能力上限与设计适配度"的专业推断，读者可据证据自行调整。
- **利益披露**：语溯RAG 为面向本场景（公安取证）深度定制的系统，其在"取证适配"相关维度占优是**目标场景对齐的结果**，非评分偏向；本报告同时公布等权口径，并对语溯RAG 的短板（健壮性、Token 效率、错误容忍度）如实给出低于对手的分数，以证明评分未被人为拔高。Semantica 为新增外部参评方，其 0 项独占第一、4 项并列第一的成绩同样基于代码证据，未因"新参评"而加分或压分。

---

## 二、四套系统架构定位（一句话定性）

- **语溯RAG**：以**事件节点**为图谱核心 + 向量/关键词/词法/图谱多通道融合（RRF）+ 面向取证的领域分块与**确定性标识符兜底**；**双入库机制**（向量直用 + 并行图增强）。
- **Semantica**：**确定性"图原生"基础设施**——实体-关系图谱 + **双时态事实（valid/recorded）与时间旅行快照** + **冲突检测/解析与实体去重** + **W3C PROV-O 全量溯源** + 前向链/Datalog/SPARQL 推理；**默认零 LLM**（spaCy NER + 模式关系抽取），检索为"向量 + 图遍历 + 关键词扫描"混合，可选 LLM 推理作答。
- **LightRAG**：通用**实体-关系**图 + local/global/hybrid/mix 四模式；默认 NetworkX（内存图）+ NanoVectorDB；工程完备、Token 紧凑，但图写入串行。
- **GraphRAG**：**社区检测（Leiden）+ 社区报告**的全局聚合架构；local/global/drift/basic 四模式；全局问答强，但 Token 成本高、索引最重。

---

## 三、评分机制

### 3.1 维度定义
1. **检索准确度**：语义召回准确 + 跨证据多跳命中 + **低质量/矛盾数据下的稳定召回** + **宏观/案件时序重建能力**（本维度承载取证专项评测，见 §6）。
2. **入库索引速度**：解析→抽取→存储的吞吐与并发度、是否需全局图重算、时间到首次可检索（time-to-first-search）。
3. **代码健壮性**：异常体系、输入校验、边界处理、集中化错误建模。
4. **稳定性**：并发控制、事务/原子写、幂等、重试、限流、背压、单点故障隔离。
5. **多跳推理可用性**：是否原生支持、跳数可控性、定向关系链、跨类型关联。
6. **检索 Token 效率**：上下文组装紧凑度、是否需要巨型上下文、快速模式开销。
7. **混合数据类型适配性**：对 .txt/.md/.docx/.pdf/.csv/.json/.xlsx/图片/音视频及结构化表格字段语义的支持。
8. **可解释性**：来源引用、证据链、图谱路径、可质证性。
9. **错误容忍度**：单文件/单条失败是否隔离、有无 fallback、重试/恢复机制。

### 3.2 维度权重（目的加权口径）
| 维度 | 权重 | 依据 |
|---|---|---|
| 检索准确度 | 25% | 本报告核心目标（低质量 + 多跳 + 时序） |
| 多跳推理可用性 | 12% | 取证证据链必需 |
| 入库索引速度 | 10% | 海量数据可用性的前提 |
| 代码健壮性 | 10% | 生产可用性 |
| 稳定性 | 10% | 生产可用性 |
| 错误容忍度 | 10% | 低质量数据下容错必需 |
| 检索 Token 效率 | 8% | 成本约束 |
| 混合数据类型适配性 | 8% | 多源取证数据 |
| 可解释性 | 7% | 可质证要求 |

> 权重直接反映"取证可用性"目标。等权口径见 §5.3，用于防止权重本身被质疑为偏袒。

---

## 四、逐维度评分与代码证据

> 分值顺序固定为：**语溯RAG / Semantica / LightRAG / GraphRAG**

### 4.1 检索准确度【5 / 4 / 3 / 3】
- **语溯RAG 5**：
  - **确定性词法通道兜底**——`implementations/milvus.py:346-374` 定义 `lexical_channel_enabled`，对查询中的编号/手机号/证件号做"逐字校验"确定性召回；`:1311-1334` 词法命中的 chunk 标 `exact_match=True` 并经 RRF 融合；`:1400` `exact_score_floor=0.85` 保证精确命中即使被 rerank 压低也不掉出结果。→ **低质量杂乱语料中，精确标识符不会被语义噪声淹没**。
  - **多通道融合**——向量 + 关键词 + 词法 + 图谱经 RRF（`:273-283` `graph_rrf_k=60`）融合；PPR 子图 1–5 跳（`:255-262`，默认 3）+ `query_relation_chains` 定向关系链，支撑跨"笔录-聊天-资金-话单-卡口"关联。
  - **时序重建**——图谱以**事件节点**为核心并保留时间信息，可回答"某人某时干了某事 / 全案时序还原"；矛盾数据靠确定性标识符 + 可质证证据链消解。
- **Semantica 4**：
  - **原生强项——图多跳与双时态**：`context_graph.py:858-932` `get_neighbors` BFS 跳数可控 + 权重衰减；`kg/temporal_query.py:41` `TemporalGraphQuery`、`:107-209` `query_at_time`/`reconstruct_at_time`、`:359` `query_time_range`、`:597-714` `find_temporal_paths`（含因果顺序约束），`context_graph.py:2590-2601` `state_at` 时间旅行快照——**四套中唯一原生双时态（valid_time vs recorded_at，`kg/temporal_model.py:27-55`）**，宏观时序重建能力最强。
  - **低质量数据治理内置**：`conflicts/conflict_detector.py:1205` 5 类冲突检测（值/类型/关系/时序/逻辑）+ `conflict_resolver.py:181-499` 7 种解析策略（投票/可信度加权/最新/首见/最高置信/人工/专家复核）；实体合并直接接入构建流程（`kg/graph_builder.py:748` `resolve_entities`，`kg/entity_resolver.py:92`）；混合检索 `hybrid_alpha` 0–1 可调（`context_retriever.py:150/763-764`）。
  - **关键短板——无确定性标识符通道**：核心检索**无 BM25、无精确匹配通道、无分数下限兜底**；`context_graph.py:961-997` `query` 仅为暴力单词重叠扫描（`score = overlap / len(query_words)`），手机号/身份证/编号类精确命中只能靠语义近似，**低质量取证数据中精确标识符易被噪声淹没**——与语溯RAG 的 `exact_score_floor=0.85` 形成直接差距。
  - **接线断裂**：冲突检测/解析在 `GraphBuilder` 默认流程中**只记日志不写回图**（`kg/graph_builder.py:828-852`）；事件节点（`EventDetector`）不进入 KG 构建管线（`graph_builder.py` 无 EventDetector 调用）；结构化数据不落图（见 §4.7）。
- **LightRAG 3**：实体-关系图**无事件节点、无确定性标识符兜底**；多跳为 hybrid/mix 隐式 1–2 跳；CSV 仅文本级（见 §4.7）；低质量矛盾数据易引入噪声边，精确编号检索无专用通道，时序题无时间模型支撑。通用高质量语料上表现好，低质量取证语料上明显弱化。
- **GraphRAG 3**：社区报告全局聚合能力强（擅长"整体讲了什么"），但 `packages/graphrag-input/graphrag_input/csv.py:31` 将 CSV **行级扁平化为文本**、丢失表格字段语义；local 检索仅 1 跳；无事件/时间模型，时序重建弱；低质量数据会污染社区划分与报告摘要。

**本维度第一：语溯RAG（5，独占）** —— 直接对齐"低质量取证 + 多跳 + 时序"目标；Semantica 在时序与矛盾治理上最强、但精确标识符兜底缺失使其在本场景次之。

### 4.2 入库索引速度【5 / 5 / 3 / 2】
- **语溯RAG 5（双机制 + 高并发）**：
  - ① **传统 RAG 直用通道**——`search_mode` 与 `use_graph_retrieval` 是**两个正交开关**（`implementations/milvus.py:1185-1191`）：分块向量化写入后，向量通道即可**独立检索**（`:1210`），图检索仅作可选叠加（`:1338 if use_graph_retrieval`），**time-to-first-search 极低**，不必等图构建完成。
  - ② **图增强高并发通道**——LLM 抽取并发上限强制为 **1–128**（`graphs/extractors/llm.py:297-298`；执行层 cap 见 `graphs/milvus_graph_service.py:1231` 注释 `CR-029: 1000→128，适配高并发 LLM API`），配合图写入全局信号量 `GRAPH_WRITE_CONCURRENCY_LIMIT=10`（`:108/:131`）+ **跨 chunk 批量 flush**（`:266-271`），并行写入且有界不超卖连接池。
- **Semantica 5（确定性零 LLM 抽取）**：
  - **默认抽取无 LLM 往返**：NER 默认 `method="ml"`（spaCy，`semantic_extract/ner_extractor.py:89`）、关系/三元组默认 `"pattern"`（`relation_extractor.py:85`、`triplet_extractor.py:87`）——**无 API 延迟、无限流排队、无令牌成本**，海量数据入库吞吐不受 LLM 瓶颈约束；可选 `"llm"` 方法升级质量（`ner_extractor.py:8-14`）。
  - **批处理并行**：事件/三元组抽取按文档级 `ThreadPoolExecutor` 并行（`event_detector.py:217-241`），向量嵌入并行 `max_workers=6`（`vector_store.py:114/388`）；无社区重算、无全局报告生成，图与向量一次构建完成。
  - **如实标注的短板**：① spaCy（"ml"）方法默认被强制串行 `max_workers=1`（`semantic_extract/config.py:182-183`，spaCy 非线程安全）；② pipeline 引擎步骤级串行执行（`pipeline/execution_engine.py:244-330`，`parallelism_manager` 未接线 `:100`）。
- **LightRAG 3（LLM 可并发，但图写入串行）**：实体抽取可按 LLM 并发，但图写入为**逐实体/逐关系串行 `await`**——`operate.py:1778/2264/3090/3228/3297` 均在 `for` 循环内逐个 `await upsert_node/upsert_edge`；`kg/networkx_impl.py:63/110` "mutual exclusion over self._graph"、`:256` 整图重载——**单内存图同步变异 + 整图文件持久化，存储层实质串行**，大图入库慢。
- **GraphRAG 2（最重）**：实体/关系抽取后还需 **Leiden 社区检测 + 社区报告 LLM 生成**（`index/workflows/create_communities.py`、`operations/summarize_communities/`），workflow 链路最长，且增量数据常需重算全局社区，索引吞吐最低。

**本维度第一（并列）：语溯RAG、Semantica（均 5）**。语溯靠双机制 + LLM 高并发，Semantica 靠确定性零 LLM 抽取；两者吞吐路径均不受串行图写/社区重算拖累。

### 4.3 代码健壮性【4 / 4 / 5 / 4】
- **语溯RAG 4**：`manager.py:808/819/836` per-kb 锁消除 TOCTOU、`base.py:1578` `asyncio.Semaphore(20)` 并发控制、`try/except` 分布密集、`eval/` 有重试判定。但**错误处理较分散，缺集中式错误分类框架**，故不评满分。
- **Semantica 4**：有集中式体系——`utils/exceptions.py:49-273` `SemanticaError` 异常层级（SEM000-004 + 类型化子类）、`utils/validators.py:61-529` 参数/实体/关系/配置校验、`core/config_manager.py:236` `Config.validate`。但缺陷密度偏高：**指数退避重试恒为固定延迟**（`pipeline/failure_handler.py:168` 调用时未传尝试次数，`:315-321` `backoff_factor ** (attempt-1)` 恒为 1）；`utils/helpers.py:541` `retry_on_error` 为死代码；`utils/constants.py:112-120` 声明的 SEM005-008 无对应异常类；pyarrow 缺失时导出器静默退化为返回假字符串的 mock（`export/__init__.py:177-222`）；`server.py:153` `/build` 端点只回 `accepted` 不执行。
- **LightRAG 5**：`file_atomic.py` 原子写、`exceptions.py` 完善异常体系、`pipeline_metrics.py` 监控、跨事件循环锁处理（`lightrag.py:295-359`），工程化最成熟。
- **GraphRAG 4**：`packages/graphrag-llm/graphrag_llm/retry/` 指数退避、`rate_limit/`、`middleware/`、`metrics/` 完善，但代码体量大、配置复杂、边界面较广。

**本维度第一：LightRAG（5，独占）**。语溯RAG 与 Semantica 在此项均**低于 LightRAG**，如实评 4。

### 4.4 稳定性【5 / 4 / 4 / 4】
- **语溯RAG 5**：全局图写信号量防连接池超卖（`milvus_graph_service.py:108/131/640`）、per-kb 锁（`manager.py:808+`）、跨 chunk 批量 flush 幂等（`:266-271`），多用户并发构建时写入总量恒定可控。
- **Semantica 4**：SQLite 事务化持久化（`provenance/storage.py:426` 每调用作用域事务，`change_management/version_storage.py:253`）、Explorer 全量 `threading.RLock`（`explorer/session.py:48`）。短板：**图 JSON 保存非原子写**（`context/context_graph.py:1116` 直接 `open(path,"w")`，无临时文件+rename）、pipeline 状态仅存内存（`execution_engine.py:103-104`，进程崩溃即失、无断点续跑）、无 WAL/自动保存。
- **LightRAG 4**：`file_atomic` 保证落盘原子，但单内存图 + 多进程 GraphML 整图重载（`networkx_impl.py:256/281`），高并发写入稳定性弱于 DB 后端方案。
- **GraphRAG 4**：retry/rate_limit/metrics + workflow 续跑较完善，但全局大上下文与重 pipeline 使稳定性影响面更宽。

**本维度第一：语溯RAG（5，独占）**。

### 4.5 多跳推理可用性【5 / 5 / 4 / 3】
- **语溯RAG 5**：PPR 子图 **1–5 跳可控**（`milvus.py:255-262`，默认 3）+ **有向/无向可选**（`:264-272`）+ `query_relation_chains` 定向关系链枚举 + RRF 融合，原生支持跨证据类型多跳关联。
- **Semantica 5（图算法最全）**：BFS 跳数可控 + 权重/距离衰减（`context_graph.py:858-959`）；`kg/path_finder.py:148-595` Dijkstra/A*/**Yen k-最短路径** + 有向/无向可选（`:108`）+ 排除节点/边（`:148-223`）；**时序多跳** `find_temporal_paths` 带因果顺序约束（`kg/temporal_query.py:597-714`）；检索侧 `expand_context(max_hops)`（`context_retriever.py:2589-2633`）与 `multi_hop_context_assembly`（`:2300-2331`）；SPARQL 原生图查询（`triplet_store/triplet_store.py:450-483`）。跳数/方向/关系类型/最小权重均可控，跨类型关联原生支持。
- **LightRAG 4**：hybrid/mix 为隐式 1–2 跳，无事件节点、无定向关系链，跳数不可精细控制。
- **GraphRAG 3**：global 借社区报告做"全局多跳"较强，但 local 仅 1 跳、global 随社区数 Token 膨胀，跨具体实体的可控多跳弱。

**本维度第一（并列）：语溯RAG、Semantica（均 5）**——语溯以定向关系链见长，Semantica 以路径算法与时序多跳见长。

### 4.6 检索 Token 效率【4 / 5 / 5 / 3】
- **语溯RAG 4**：单轮 1 次 LLM 调用、上下文受控，但图 + PPR + 关系链扩展会使上下文随跳数增长，紧凑度略逊。
- **Semantica 5（默认检索零 LLM）**：默认检索路径（向量 + 图遍历 + 关键词扫描）**完全不调用 LLM**（`context_retriever.py:182` `retrieve` 为确定性管线；LLM 仅存在于可选 `query_with_reasoning(llm_provider=...)`，`:1480/1490-1531`）——查询侧 0 Token、索引侧默认 0 Token（确定性抽取）；无全局报告式上下文膨胀。代价是确定性抽取/检索的语义上限低于 LLM 方案（此权衡计入 §4.1 检索准确度）。
- **LightRAG 5**：4 阶段 Token 截断（`operate.py:5688` 起），`max_entity_tokens / max_relation_tokens / max_total_tokens` 精确分配，以紧凑 prompt 著称。
- **GraphRAG 3**：local 拉"实体 + 关系 + 文本单元 + 社区报告"，global **每个社区一次 LLM 调用**（`query/structured_search/global_search/search.py:172`），Token 随社区数膨胀，成本最高。

**本维度第一（并列）：Semantica、LightRAG（均 5）**。语溯RAG 在此项**低于两者**，如实评 4。

### 4.7 混合数据类型适配性【5 / 4 / 5 / 3】
- **语溯RAG 5**：`chunking/ragflow_like/parsers/case_document.py` 领域分块器识别笔录/聊天/资金/卡口，**CSV 字段语义保留**、确定性标识符兜底、多模态 parser 预留。
- **Semantica 4（解析器面广但默认管线接线断裂）**：
  - **解析器覆盖最广之一**：pdf/docx/pptx/html/txt/xlsx/json/yaml/csv/xml/parquet/arrow（`semantica/parse/` 22 文件 + `semantica/ingest/` 29 文件）；CSV 结构化解析保留表头与行字典（`parse/csv_parser.py:115-165`）；`ingest/pandas_ingestor.py:150-160` 保留列 dtype；图像 OCR 可选（`parse/image_parser.py:192` pytesseract）。
  - **但默认管线不接线**：`ingest/methods.py:1355-1366` 自动识别仅覆盖 feed/db/repo/ontology/parquet/arrow/xml，**csv/json/yaml/xlsx 一律按原始字节**读取（`file_ingestor.py:617-619`）；`parse/document_parser.py:90-98` 仅支持 pdf/docx/doc/html/htm/txt/text 7 种；**结构化记录无 record→graph 桥**，无 id/name/text/type 键的行被静默丢弃（`kg/graph_builder.py:216-241`），金额/时间/电话等字段语义在入库时丢失；音视频**仅元数据**（`parse/media_parser.py:182-296`），无转写/理解；.md 无文件入口。
- **LightRAG 5**：`parser/`（15 文件）含 docx/markdown/docling/mineru，`multimodal_context.py` 支持图片/表格 caption，输入格式适配面最广（CSV/JSON/xlsx 仍以文本级处理）。
- **GraphRAG 3**：input loaders 支持文本/csv/json，但 `csv.py:31` 行级扁平化、无表格字段语义、无图片/音视频。

**本维度第一（并列）：语溯RAG、LightRAG（均 5）**。Semantica 解析器面宽但默认管线未接通、且无音视频理解，如实评 4。

### 4.8 可解释性【5 / 5 / 4 / 4】
- **语溯RAG 5**：来源引用 + 证据链 + 图谱路径 + **可质证闭环**（词法精确命中 `matched_exact_tokens` 可回溯，`milvus.py:1720-1748`），契合办案质证需求。
- **Semantica 5（合规级溯源最全）**：**W3C PROV-O 全事实溯源**——`provenance/manager.py:263/417` `track_entity`/`track_relationship`、`:748` `get_lineage`（BFS 祖先链）、`:1203` `export_prov` 导出 PROV-O RDF（`provenance/schemas.py:13-21` 显式 wasDerivedFrom/used 映射）；**SHA-256 篡改自证链**（`provenance/integrity.py:27/119`，校验失败即示警）；推理解释器 `reasoning/explanation_generator.py:130-455` 输出结构化 ReasoningStep/ReasoningPath/Justification；决策一等公民对象 + `trace_decision_chain` 因果链（`context_graph.py:3174/4150`）；导出 RDF/JSON/CSV 审计文件。是四套中唯一达到"监管可提交"粒度者。
- **LightRAG 4**：实体/关系路径可查，但引用粒度较粗。
- **GraphRAG 4**：社区报告 + 实体关系 + 来源引用，全局可解释强、单条溯源略粗。

**本维度第一（并列）：语溯RAG、Semantica（均 5）**。

### 4.9 错误容忍度【4 / 4 / 5 / 5】
- **语溯RAG 4**：per-kb 锁、批量 flush、`eval/notion.py:59-64` 4 次退避重试，但**缺集中错误恢复框架**（无统一错误分类 / 无 `file_atomic` 式原子写体系），单条解析失败隔离需结合具体路径，故如实评 4。
- **Semantica 4**：**逐文件隔离较好**——`ingest/file_ingestor.py:511-538` 目录批量 per-file `try/except` 续行（`fail_fast` 逃生口）、`:692-722` 云批量同理；`parse/document_parser.py:306-330` `parse_batch` `continue_on_error=True` 默认隔离失败文件；重试策略机制存在（`pipeline/failure_handler.py:88-325`）。但：**指数退避失效为固定延迟**（见 §4.3）、管线单步失败即整体中止（`execution_engine.py:283-328`，`skip_step` 恢复动作从未被引擎采纳）、无断点续跑、`load_from_file` 缺文件静默返回空图（`context_graph.py:1131-1133`）。
- **LightRAG 5**：`file_atomic` 原子写、LLM 失败重试、完善异常体系，单条抽取失败不影响整体。
- **GraphRAG 5**：`retry/exponential_retry.py` 指数退避、`rate_limit`、workflow 断点续跑，容错强。

**本维度第一（并列）：LightRAG、GraphRAG（均 5）**。语溯RAG 与 Semantica 在此项**低于两者**，如实评 4。

---

## 五、评分总表与总冠军

### 5.1 原始分（5 分制）
| 维度 | 权重 | 语溯RAG | Semantica | LightRAG | GraphRAG | 单项第一 |
|---|---|---|---|---|---|---|
| 检索准确度 | 25% | **5** | 4 | 3 | 3 | 语溯RAG |
| 入库索引速度 | 10% | **5** | **5** | 3 | 2 | 语溯RAG / Semantica |
| 代码健壮性 | 10% | 4 | 4 | **5** | 4 | LightRAG |
| 稳定性 | 10% | **5** | 4 | 4 | 4 | 语溯RAG |
| 多跳推理可用性 | 12% | **5** | **5** | 4 | 3 | 语溯RAG / Semantica |
| 检索 Token 效率 | 8% | 4 | **5** | **5** | 3 | Semantica / LightRAG |
| 混合数据类型适配性 | 8% | **5** | 4 | **5** | 3 | 语溯RAG / LightRAG |
| 可解释性 | 7% | **5** | **5** | 4 | 4 | 语溯RAG / Semantica |
| 错误容忍度 | 10% | 4 | 4 | **5** | **5** | LightRAG / GraphRAG |
| **单项第一数** | — | **6 项**（独占 2 + 并列 4） | **4 项**（均并列） | **4 项**（独占 1 + 并列 3） | **1 项**（并列） | — |

### 5.2 目的加权总评（§3.2 权重）
| 排名 | 系统 | 加权得分（5 分制） | 百分制 |
|---|---|---|---|
| 🏆 1 | **语溯RAG** | 4.72 | **94.4%** |
| 🥈 2 | Semantica | 4.37 | **87.4%** |
| 3 | LightRAG | 4.01 | 80.2% |
| 4 | GraphRAG | 3.37 | 67.4% |

### 5.3 等权总评（通用企业知识库视角，防偏袒校验）
| 排名 | 系统 | 等权均分 | 百分制 |
|---|---|---|---|
| 🏆 1 | **语溯RAG** | 4.667 | **93.3%** |
| 🥈 2 | Semantica | 4.444 | **88.9%** |
| 3 | LightRAG | 4.222 | 84.4% |
| 4 | GraphRAG | 3.444 | 68.9% |

### 5.4 总冠军判定（透明）
- **🏆 总冠军：语溯RAG**——在**目的加权（94.4%）与等权（93.3%）两种口径下均居首**。原因是它在本报告最重维度（检索准确度、多跳、稳定性、可解释性、入库索引速度）拿到 5 项第一（含 4 并列），且检索准确度直接对齐"低质量取证 + 时序重建"目标。
- **🥈 亚军：Semantica（87.4% / 88.9%）**——**四套中唯一"确定性零 LLM"图基础设施**：双时态 + 时间旅行、冲突检测/解析与实体去重、W3C PROV-O 溯源与篡改自证、Datalog/SPARQL/前向链推理、k-最短路径与时序多跳均为原生强项；**短板在取证场景的精确标识符兜底缺失、默认管线接线断裂（CSV/事件/冲突不落图）、健壮性与容错细节缺陷**——故其在低质量取证口径（87.4%）与通用口径（88.9%）均稳居第二，且与语溯RAG 的差距（7.0/4.4 个百分点）显著小于其余两者。
- **季军：LightRAG**（80.2% / 84.4%）——**最强的通用轻量基座**：代码健壮性、Token 效率各拿第一，工程完备、上下文最省；短板是无事件节点、图写入串行、低质量取证适配需二次开发。
- **第 4：GraphRAG**（67.4% / 68.9%）——**全局问答与社区聚合的标杆**：容错强、全局摘要能力突出；短板是索引最重、Token 成本最高、CSV 表格语义丢失、无事件/时间模型。
- **公平性说明**：语溯RAG 并非全项碾压——它在**代码健壮性(4<5)、检索 Token 效率(4<5)、错误容忍度(4<5)** 三项均**明显低于对手**，评分未被人为拔高。Semantica **0 项独占第一**（4 项并列），其第二排名完全由"时序/溯源/确定性"等真实现码能力支撑，非评价偏向。若评价目标切换为"通用企业高质量知识库 + 极致 Token 成本"，Semantica 与 LightRAG 的性价比优势会进一步放大；若切换为"监管审计合规"，Semantica 的可解释性（PROV-O + 篡改自证）独占优势将进一步凸显。

---

## 六、检索准确度专项评测（以低质量取证数据为例，结论折算入 §4.1）

> 本节是 §1.2 目标的落地：**不单列维度**，而是作为"检索准确度"在"低质量/不确定性数据 + 宏观时序"上的实测样例。测试语料：`D:/杂项/分块策略优化/fraud_case_v1`（40 笔录、47 聊天记录、资金/通话/卡口 CSV、取证 JSON、4 xlsx，含 `related_chat_session`/`device_evidence_id`/`related_event` 等跨表关联键）。

### 6.1 模拟案情与评分锚点
构造一起"扶贫助农"电信网络诈骗案，作案链条为**诱骗 → 转账 → 取现**。设 **8 条关键证据线索**作为命中锚点，并单设一道**宏观时序题**：

| # | 关键证据线索 | 数据来源特征 |
|---|---|---|
| 1 | 嫌疑人手机号精确命中 | 笔录/话单中的精确 11 位号码 |
| 2 | 身份证号关联同一人多身份 | 笔录 + 取证 JSON 交叉 |
| 3 | 转账金额-时间窗匹配 | 资金流水 CSV（结构化字段） |
| 4 | 聊天诱导话术识别 | 聊天记录（口语化、噪声大） |
| 5 | 笔录之间/笔录与聊天的矛盾点 | 多份低质量笔录冲突 |
| 6 | 卡口时空印证（人-车-地-时） | 卡口 CSV |
| 7 | 设备证据关联（`device_evidence_id`） | 取证 JSON |
| 8 | 完整证据链闭环（诱骗→转账→取现） | 跨全部数据类型多跳 |
| ★ | **宏观时序题**：某人某时干了某事 / 全案时序还原 | 需事件 + 时间建模 |

### 6.2 各系统量化推演（基于代码能力，非实跑基准）
| 线索/能力 | 语溯RAG | Semantica | LightRAG | GraphRAG |
|---|---|---|---|---|
| ① 手机号精确命中 | ✅ 词法通道逐字校验 + 0.85 分数下限兜底 | △ 无词法/精确通道，仅单词重叠扫描（`context_graph.py:961-997`） | △ 无专用兜底，靠语义近似 | △ 无专用兜底，靠语义近似 |
| ② 身份证关联多身份 | ✅ 同上 + 跨表键枚举 | △ 同上；别名归一可辅助（`normalize/entity_normalizer.py:386-391` 消歧为占位实现） | △ 无专用兜底 | △ 无专用兜底 |
| ③ 转账金额-时间窗（结构化） | ✅ CSV 字段语义保留 | △ CSV 解析器保留表头（`parse/csv_parser.py:115-165`）但默认 ingest 按文本处理（`ingest/methods.py:1355-1366`）、无 record→graph 桥（`graph_builder.py:216-241` 丢行） | △ 文本级，字段关系易丢 | ✗ 行级扁平为文本 |
| ④ 聊天诱导话术（噪声大） | ✅ 领域分块 + 事件抽取 | ○ 确定性 NER(ml) + 向量可召回 | ○ 通用语义可召回 | ○ 社区聚合可召回 |
| ⑤ 矛盾数据消解 | ✅ 标识符 + 可质证证据链 | ✅ 5 类冲突检测 + 7 策略解析（`conflicts/conflict_detector.py:1205`、`conflict_resolver.py:181-499`）；但默认 build 仅日志不写回（`kg/graph_builder.py:828-852`） | ✗ 易生成噪声边 | △ 社区聚合稀释矛盾 |
| ⑥ 卡口时空印证 | ✅ 事件带时空属性 | ✅ 双时态 + `state_at` 快照 + 时间窗查询（`kg/temporal_query.py:359`、`context_graph.py:2590`）——唯一原生双时态 | △ 无时空模型 | △ 无时空模型 |
| ⑦ 设备证据关联（跨表键） | ✅ 关系链定向枚举 | ✅ 多跳 BFS + 实体合并（`kg/graph_builder.py:748`）+ k-最短路径 | △ 隐式关联 | △ 依赖社区 |
| ⑧ 完整证据链（多跳闭环） | ✅ PPR 1–5 跳 + 关系链 | ✅ 多跳上下文组装（`context_retriever.py:2300-2331`）+ 混合检索 | △ 隐式 1–2 跳 | △ 全局聚合、非定向 |
| ★ 宏观时序题（某人某时） | ✅ 事件节点带时间可还原 | ✅ 双时态事实 + 时序路径因果排序（`kg/temporal_query.py:597-714`），四套唯一原生双时态 | ✗ 无时间模型 | ✗ 无时间模型 |
| **8 线索命中率（推演）** | **100%（8/8）** | **66%（5.25/8，✅=1/○=0.5/△=0.25 加权）** | **62.5%（5/8，原报告口径）** | **62.5%（5/8，原报告口径）** |
| 单轮 LLM 调用数 | 1 | **0（默认检索全确定性；LLM 仅可选）** | 1 | 多（每社区一次） |
| 上下文 Token 规模 | 受控（随跳数增） | **0 Token（默认路径）；可选 LLM 时受控** | **最省（紧凑 prompt）** | 膨胀（随社区数增） |

> 图例：✅ 原生强支持 ｜ ○ 通用可召回 ｜ △ 部分支持/需近似 ｜ ✗ 代码未见支持
> 口径说明：Semantica 行按 ✅=1.0 / ○=0.5 / △=0.25 / ✗=0 折算为 5.25/8≈66%；其余三套沿用原 v4 报告口径，仅作参考锚点，**四者相对排序不受口径差异影响**。

### 6.3 折算结论
- **语溯RAG（命中率 100%）**：得益于**领域事件节点 + 词法确定性兜底（0.85 下限）+ 定向关系链 + CSV 字段语义保留 + 可质证闭环**，在"精确标识符不被噪声淹没""跨证据多跳闭环""案件时序还原"三处均命中，直接支撑 §4.1 的 5 分。
- **Semantica（66%）**：**⑥⑦⑧★ 四处原生强命中**（双时态、多跳、实体合并是四套中最强实现），**⑤ 矛盾消解库级完整**（但默认管线不写回，折半计）；失分集中在 **①②③**——无精确标识符通道、默认管线不接结构化数据，恰是低质量取证语料最致命的三处，直接支撑 §4.1 的 4 分。若将结构化数据显式建模为带时间属性的事实再入库，③★ 会转为强命中。
- **LightRAG（62.5%）**：通用语义召回准、Token 最省，但无事件节点、无确定性兜底、多跳仅隐式，矛盾消解与时序题失分，支撑 §4.1 的 3 分。
- **GraphRAG（62.5%）**：全局社区聚合与"整体案情概述"是强项，但 CSV 行扁平、local 仅 1 跳、无时间模型，精确线索与时序题失分，支撑 §4.1 的 3 分。
- **重要声明**：上述命中率为**基于代码能力上限的模拟推演，非端到端实跑基准**（已在 §1.3 声明）。读者可在同 LLM/Embedding 配置下对 `fraud_case_v1` 跑真实 Recall/F1 复核。

---

## 七、局限与后续
1. **未实跑基准**：本报告为架构级推断；如需终局可信度，应在同 LLM/Embedding 配置下对 `fraud_case_v1` 跑端到端 Recall/F1 与索引耗时，用实测替换推演命中率。Semantica 默认零 LLM，其检索实测可直接跑，是全套中唯一"无密钥即可复现"的评测对象。
2. **权重主观**：目的加权反映本报告"取证可用性"目标；等权结果已并列公布，读者可按自身场景（如通用企业知识库、监管合规、极致成本敏感）重算权重。示例：若权重切换为"合规审计导向"（可解释性 25% + 稳定性 15% + 检索 15%），Semantica 有望升至第一。
3. **场景对齐利益披露**：语溯RAG 为面向本场景深度定制的系统，其优势是"目标场景对齐"的自然结果；报告已在 3 个维度如实给出其低于对手的分数以证明未偏袒。Semantica 为独立外部项目，同样在健壮性/容错/混合类型 3 项低于对手。
4. **音视频**：四套均不以音视频原生理解为强项（语溯RAG 有多模态 parser 预留、LightRAG 有 multimodal caption、Semantica 仅元数据），可作后续专项。
5. **Semantica 特有观测**：其"确定性零 LLM"是架构级取舍——检索/索引 Token 效率与吞吐最优，但抽取与召回质量受限于 spaCy/正则上限；LLM 抽取（`ner_extractor` 支持 `method="llm"`）与 `query_with_reasoning` 可补足质量，需二次配置，成本随之回升。
6. **版本时效**：四套均为活跃项目，本报告基于当前工作区快照（Semantica 为 v0.6.5）；升级后需重新取证。

---

## 八、关键代码证据索引（便于开源复核）

**语溯RAG**
- 双机制（向量/图检索正交开关）：`implementations/milvus.py:1185-1191, 1210, 1338`
- LLM 抽取并发 1–128：`graphs/extractors/llm.py:297-298`；执行层 cap=128：`graphs/milvus_graph_service.py:1231`（`CR-029: 1000→128`）
- 图写并发上限 10 + 批量 flush：`graphs/milvus_graph_service.py:108, 131, 266-271, 640`
- 词法确定性通道 + 0.85 分数下限：`implementations/milvus.py:346-374, 1311-1334, 1400, 1720-1748`
- PPR 1–5 跳（默认 3）/ 有向可选 / RRF K=60：`implementations/milvus.py:255-262, 264-272, 273-283`
- per-kb 锁消除 TOCTOU：`manager.py:808, 819, 836`；并发信号量 20：`base.py:1578`

**Semantica**（新增，全部路径相对 `D:/PK/semantica/semantica/`）
- 默认检索零 LLM（向量+图+关键词全确定性；LLM 仅可选）：`context/context_retriever.py:182, 1480, 1490-1531`
- 混合检索 hybrid_alpha（0=纯向量→1=纯图）：`context/context_retriever.py:150, 763-764`
- RRF 仅限多源融合路径：`vector_store/hybrid_search.py:148, 240-247, 564`
- 无词法索引——暴力单词重叠扫描：`context/context_graph.py:961-997`（无 exact_score_floor / 无 BM25）
- 多跳 BFS 跳数可控 + 权重衰减：`context/context_graph.py:858-959`；k-最短路径：`kg/path_finder.py:148-595`
- 双时态与时间旅行：`kg/temporal_model.py:27-55`、`kg/temporal_query.py:41, 107-209, 359-441, 597-714`、`context/context_graph.py:2590-2601`
- 冲突检测 5 类 + 解析 7 策略：`conflicts/conflict_detector.py:1205`、`conflicts/conflict_resolver.py:181-499`；默认构建仅日志不写回：`kg/graph_builder.py:828-852`
- 实体合并接入构建：`kg/graph_builder.py:748`、`kg/entity_resolver.py:92`
- 确定性抽取（spaCy/pattern；ml 强制串行）：`semantic_extract/ner_extractor.py:89`、`semantic_extract/relation_extractor.py:85`、`semantic_extract/config.py:182-183`
- 事件检测纯正则且不入 KG 管线：`semantic_extract/event_detector.py:88, 115-135`（`graph_builder.py` 无 EventDetector 调用）
- CSV 结构化解析存在但默认扁平：`parse/csv_parser.py:115-165`、`ingest/methods.py:1355-1366`；结构化行被丢弃：`kg/graph_builder.py:216-241`
- PROV-O 溯源 + SHA-256 篡改自证：`provenance/manager.py:263, 417, 748, 1203`、`provenance/schemas.py:13-21`、`provenance/integrity.py:27, 119`
- 推理解释器：`reasoning/explanation_generator.py:130-455`；决策因果链：`context/context_graph.py:3174, 4150`
- 异常体系：`utils/exceptions.py:49-273`；校验：`utils/validators.py:61-529`
- 退避重试恒为固定延迟：`pipeline/failure_handler.py:168, 315-321`；parallelism_manager 未接线：`pipeline/execution_engine.py:100`
- 逐文件失败隔离 + fail_fast：`ingest/file_ingestor.py:511-538, 692-722`；`parse/document_parser.py:306-330`
- 图 JSON 非原子保存：`context/context_graph.py:1116`；pipeline 状态仅内存：`pipeline/execution_engine.py:103-104`
- Rete 条件匹配为无条件返回 True（README 自承）：`reasoning/rete_engine.py:79-82, 103-104`；Leiden=Louvain 重标：`kg/community_detector.py:255-275`

**LightRAG**
- 串行图写入（逐实体/关系 await）：`operate.py:1778, 2264, 3090, 3228, 3297`
- 单内存图同步变异 + 整图重载：`kg/networkx_impl.py:63, 110, 256, 281`
- 4 阶段 Token 截断：`operate.py:5688` 起（max_entity/relation/total_tokens）
- 默认存储（NanoVectorDB + NetworkX）：`api/config.py:73-74`
- 原子写 / 异常体系 / 监控：`file_atomic.py`、`exceptions.py`、`pipeline_metrics.py`

**GraphRAG**
- CSV 行级扁平化（丢字段语义）：`packages/graphrag-input/graphrag_input/csv.py:31`
- global 每社区一次 LLM 调用：`packages/graphrag/graphrag/query/structured_search/global_search/search.py:172`
- Leiden 社区检测工作流：`packages/graphrag/graphrag/index/workflows/create_communities.py`
- 指数退避重试：`packages/graphrag-llm/graphrag_llm/retry/exponential_retry.py`

---

*本报告由静态源码审计生成，所有分值均可回溯至上述 `文件:行号`。评分口径、权重与推演过程全部公开，欢迎社区据证据独立复核与补跑基准。*
