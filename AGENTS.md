# AGENTS.md — 语溯开源版

给 AI 编码助手的项目约定。

## 项目结构

- `yusu_kb/` — Python 知识库系统（全部自包含，不依赖 LightRAG）
- `yusu_webui/` — 前端 WebUI（React，API 层接 YUSU 后端）
- `docs/superpowers/` — 设计文档与实施计划（含 spec）

## 常用命令（Windows PowerShell）

```powershell
uv sync --extra api --extra test        # 安装依赖（同步 uv.lock）
& ".\.venv\Scripts\python.exe" -m pytest yusu_kb\tests -q   # 单测
& ".\.venv\Scripts\ruff.exe" check yusu_kb                  # lint
& ".\.venv\Scripts\python.exe" -m yusu_kb.server             # 启动 API
cd yusu_webui; bun install; bun run dev                  # 前端开发
```

## 红线

- **密钥安全**：模型 API key 只允许写入本地 `.env`（已被 .gitignore 保护）。
  禁止把密钥写入任何代码、文档、spec、计划、测试或提交内容。
- **只读参考**：`D:\YUSU` 是源项目，只读。移植代码允许逐字复制并保留原中文注释；
  新写代码用英文注释。移植时剥离源项目的 registry/配置中心依赖，改为 env 直配。
- **无外部依赖漂移**：模型层（embed/chat/rerank）全部基于 httpx，禁止引入
  langchain / aiohttp / openai SDK。

## 编码约定

- 不添加无说明的 `# noqa`；BLE001 捕获须附中文说明注释。
- 新代码遵循 ruff（默认规则集，target py311）+ `pytest-asyncio` auto 模式。
- 修改后必须 `ruff check` 零告警 + 全量 pytest 通过才算完成。
- 仓库不是 git 仓库：不要执行 git 操作；需要时先询问用户。

## 环境变量契约

- `YUSU_LLM_BASE_URL/API_KEY/MODEL` — 聊天（OpenAI 兼容 /chat/completions）
- `YUSU_EMBED_BASE_URL/API_KEY/MODEL` — embedding（回退 LLM 配置），`YUSU_EMBED_DIM` 可选
- `YUSU_RERANK_MODEL/BASE_URL/API_KEY` — 可选 reranker
- `YUSU_DATA_DIR` — 数据目录（默认 ./yusu_data）
- `YUSU_API_KEY` — 可选 API 鉴权