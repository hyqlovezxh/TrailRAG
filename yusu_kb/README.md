# 语溯开源版（YUSU Open Edition）

独立、自包含的知识库系统：上传文档 → 解析 → 分块 → 向量索引 → 检索 → 大模型问答。
基于 YUSU 知识库体系重新实现（OpenAI 兼容 API，无需本地推理服务），前端复用改造后的 WebUI 壳。

## 特性

- **知识库管理**：创建/删除知识库，上传文档（docx/pdf/md/txt/pptx/xlsx/html 等），解析、索引、更新内容
- **检索**：向量 + 关键词（SQLite LIKE）+ 混合模式，可选 Reranker 重排序，文件级过滤
- **问答**：检索增强的流式聊天（OpenAI 兼容 `/chat/completions`）
- **自包含**：全部功能在单一 Python 包内实现（`yusu_kb/`），无外部服务依赖
- **持久化**：SQLite（元数据/文件/分块）+ nano-vectordb（向量）落盘于 `YUSU_DATA_DIR`

## 架构

```
yusu_kb/
├── knowledge/        # 知识库契约、LocalKB 实现、分块、解析、检索
├── models/           # 自包含模型层：embedding / reranker / chat（httpx）
├── storage/          # vector_store（nano-vectordb 封装）+ sqlite + 本地文件
├── repositories/     # SQLite 仓储
├── utils/            # 日志、时间、哈希
└── api/              # FastAPI 服务（健康检查 / 知识库 / 聊天）
yusu_webui/       # 前端（React 壳，API 已改接 YUSU 后端）
```

## 快速开始

前置：Python >= 3.11、[uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装依赖（含 API 与测试 extras）
uv sync --extra api --extra test

# 2. 配置密钥
cp env.example .env   # 编辑填写 YUSU_LLM_* / YUSU_EMBED_*（.env 不提交）

# 3. 启动 API 服务（默认 http://localhost:8920）
uv run --extra api python -m yusu_kb.server
# 或 uv run --extra api yusu-server [--host H] [--port P]

# 4. 前端
cd yusu_webui
bun install
bun run dev          # 开发模式（连接页填写 http://127.0.0.1:8920）
bun run build        # 构建产物 dist/，由 API 服务自动托管（/ 路径）
```

## 环境变量

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `YUSU_DATA_DIR` | 否 | 数据目录，默认 `./yusu_data` |
| `YUSU_LLM_BASE_URL` / `YUSU_LLM_API_KEY` / `YUSU_LLM_MODEL` | 是 | 聊天模型（OpenAI 兼容） |
| `YUSU_EMBED_BASE_URL` / `YUSU_EMBED_API_KEY` / `YUSU_EMBED_MODEL` | 是 | Embedding 模型；未配置时回退 LLM 配置 |
| `YUSU_EMBED_DIM` | 否 | Embedding 维度，留空自动探测 |
| `YUSU_RERANK_MODEL` / `YUSU_RERANK_BASE_URL` / `YUSU_RERANK_API_KEY` | 否 | Reranker，留空不使用 |
| `YUSU_API_KEY` | 否 | API 鉴权令牌，留空不鉴权 |

## 开发

```bash
uv run pytest yusu_kb/tests          # 单测（139 用例）
uv run ruff check yusu_kb            # lint
uv run ruff format --check yusu_kb   # format

cd yusu_webui
bun test                              # 前端测试（54 用例）
bun run lint                          # eslint
```

## License

MIT