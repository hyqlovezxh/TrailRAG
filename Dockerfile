# 语溯开源版（YUSU / TrailRAG）生产镜像
#
# 构建：docker build -t yusu-kb:latest .
# 运行：docker run -d -p 8920:8920 --env-file .env -v yusu_data:/app/yusu_data yusu-kb:latest
#
# 三阶段：前端构建 → 依赖安装 → 精简运行时

# ---------- Stage 1: 构建前端 ----------
FROM oven/bun:1 AS webui
WORKDIR /webui

# 先只复制依赖清单，命中缓存后再复制源码，避免改一行代码就重装依赖
COPY yusu_webui/package.json yusu_webui/bun.lock* yusu_webui/package-lock.json* ./
RUN bun install --frozen-lockfile || bun install

COPY yusu_webui/ ./
RUN bun run build


# ---------- Stage 2: 安装 Python 依赖 ----------
FROM python:3.12-slim AS deps
WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 依赖层单独缓存：只有 pyproject/uv.lock 变化才重新解析依赖
COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p yusu_kb && touch yusu_kb/__init__.py \
    && uv sync --extra api --no-install-project --frozen

COPY yusu_kb/ ./yusu_kb/
RUN uv sync --extra api --frozen


# ---------- Stage 3: 运行时 ----------
FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    YUSU_HOST=0.0.0.0 \
    YUSU_PORT=8920 \
    YUSU_DATA_DIR=/app/yusu_data

# curl 供容器健康检查使用；libgomp1 是 scipy/numpy 的 OpenMP 运行时依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /opt/venv /opt/venv
COPY --from=deps /app/yusu_kb ./yusu_kb
COPY --from=webui /webui/dist ./yusu_webui/dist
COPY pyproject.toml README.md env.example ./
COPY scripts/doctor.py ./scripts/doctor.py

# 非 root 运行；数据目录需可写
RUN useradd --create-home --shell /bin/bash yusu \
    && mkdir -p /app/yusu_data \
    && chown -R yusu:yusu /app
USER yusu

VOLUME ["/app/yusu_data"]
EXPOSE 8920

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${YUSU_PORT}/api/health" || exit 1

# 单进程：构建/入库任务进度存于进程内存，多 worker 会导致状态查询错乱
CMD ["python", "-m", "yusu_kb.server"]
