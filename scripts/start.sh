#!/usr/bin/env bash
# 语溯开源版（YUSU / TrailRAG）一键启动脚本 —— Linux / macOS
#
# 流程：环境检查 → 后端依赖 → .env 准备与密钥校验 → 前端依赖与构建 → 启动服务
#
# 用法（在仓库任意位置执行均可）：
#   ./scripts/start.sh                  # 完整流程并启动
#   ./scripts/start.sh --port 9000      # 换端口
#   ./scripts/start.sh --skip-frontend  # 跳过前端构建（复用已有 dist）
#   ./scripts/start.sh --skip-install   # 跳过依赖安装（环境已就绪）
#   ./scripts/start.sh --dev            # 开发模式（后端代码热重载）
#   ./scripts/start.sh --no-start       # 只准备环境，不启动服务
#   ./scripts/start.sh --force          # 密钥未填写也强行启动

set -euo pipefail

PORT=8920
BIND_HOST="0.0.0.0"
SKIP_FRONTEND=0
SKIP_INSTALL=0
DEV=0
NO_START=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --host) BIND_HOST="$2"; shift 2 ;;
        --skip-frontend) SKIP_FRONTEND=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --dev) DEV=1; shift ;;
        --no-start) NO_START=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "未知参数: $1（--help 查看用法）" >&2; exit 1 ;;
    esac
done

if [[ -t 1 ]]; then
    C_CYAN='\033[36m'; C_GREEN='\033[32m'; C_YELLOW='\033[33m'
    C_RED='\033[31m'; C_GRAY='\033[90m'; C_MAGENTA='\033[35m'; C_OFF='\033[0m'
else
    C_CYAN=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_GRAY=''; C_MAGENTA=''; C_OFF=''
fi

step() { printf "\n${C_CYAN}== %s${C_OFF}\n" "$1"; }
ok()   { printf "${C_GREEN}   [OK]   %s${C_OFF}\n" "$1"; }
warn() { printf "${C_YELLOW}   [WARN] %s${C_OFF}\n" "$1"; }
err()  { printf "${C_RED}   [FAIL] %s${C_OFF}\n" "$1"; }
has()  { command -v "$1" >/dev/null 2>&1; }

# ---------- 0. 定位仓库根目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
printf "${C_MAGENTA}语溯开源版 YUSU · 一键启动${C_OFF}\n"
echo "仓库根目录: $ROOT"

# ---------- 1. 环境检查 ----------
step "环境检查"

HAS_UV=0
if has uv; then HAS_UV=1; ok "uv 已安装（推荐的依赖管理方式）"
else warn "未找到 uv，将回退 pip；建议安装：https://docs.astral.sh/uv/"; fi

PY_EXE=""
VENV_PY="$ROOT/.venv/bin/python"
if [[ -x "$VENV_PY" ]]; then
    PY_EXE="$VENV_PY"; ok "复用现有虚拟环境 .venv"
elif has python3; then PY_EXE="$(command -v python3)"
elif has python; then PY_EXE="$(command -v python)"
fi

if [[ $HAS_UV -eq 0 && -z "$PY_EXE" ]]; then
    err "未找到 Python 3.11+，也未找到 uv。请先安装 Python 3.11 及以上版本。"
    exit 1
fi

if [[ -n "$PY_EXE" ]]; then
    PY_VER="$("$PY_EXE" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null || echo "")"
    if [[ -n "$PY_VER" ]]; then
        PY_MAJOR="${PY_VER%%.*}"
        PY_REST="${PY_VER#*.}"
        PY_MINOR="${PY_REST%%.*}"
        if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 11) )); then
            err "Python $PY_VER 版本过低，本项目要求 3.11+。"
            exit 1
        fi
        ok "Python $PY_VER"
    fi
fi

HAS_BUN=0; HAS_NPM=0
if has bun; then HAS_BUN=1; ok "bun 已安装（前端构建）"
elif has npm; then HAS_NPM=1; warn "未找到 bun，将使用 npm 构建前端（较慢）"
else warn "未找到 bun / npm，将跳过前端构建"; fi

# ---------- 2. 后端依赖 ----------
if [[ $SKIP_INSTALL -eq 1 ]]; then
    step "后端依赖（已跳过 --skip-install）"
else
    step "安装后端依赖"
    if [[ $HAS_UV -eq 1 ]]; then
        uv sync --extra api
    else
        "$PY_EXE" -m pip install -e ".[api]"
    fi
    ok "后端依赖就绪"
    [[ -x "$VENV_PY" ]] && PY_EXE="$VENV_PY"
fi

# ---------- 3. .env 准备与密钥校验 ----------
step "配置文件 .env"

if [[ ! -f "$ROOT/.env" ]]; then
    if [[ ! -f "$ROOT/env.example" ]]; then
        err "缺少 env.example，无法生成 .env"
        exit 1
    fi
    cp "$ROOT/env.example" "$ROOT/.env"
    ok "已从 env.example 生成 .env"
else
    ok ".env 已存在"
fi

# 读取校验所需的键（不打印任何密钥内容）
env_get() {
    local key="$1"
    sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//p" "$ROOT/.env" \
        | tail -n 1 | sed 's/^"//; s/"$//; s/^'"'"'//; s/'"'"'$//' | tr -d '\r'
}

MISSING=()
for key in YUSU_LLM_API_KEY YUSU_LLM_BASE_URL YUSU_LLM_MODEL; do
    [[ -z "$(env_get "$key")" ]] && MISSING+=("$key")
done
# Embedding 未单独配置时回退 LLM 配置，因此只在显式给了 base_url 却漏了 key 时报错
if [[ -n "$(env_get YUSU_EMBED_BASE_URL)" && -z "$(env_get YUSU_EMBED_API_KEY)" ]]; then
    MISSING+=("YUSU_EMBED_API_KEY")
fi

if (( ${#MISSING[@]} > 0 )); then
    warn "以下必填项尚未填写：${MISSING[*]}"
    printf "${C_YELLOW}   请编辑 %s/.env 后重试（密钥只写入 .env，切勿提交到仓库）。${C_OFF}\n" "$ROOT"
    if [[ $FORCE -eq 0 ]]; then
        err "配置不完整，已终止。确认要继续可加 --force。"
        exit 1
    fi
    warn "--force 已指定，继续启动（模型调用会失败）"
else
    ok "模型配置项完整"
fi

if [[ -n "$(env_get YUSU_RERANK_BASE_URL)" && -z "$(env_get YUSU_RERANK_API_KEY)" ]]; then
    warn "Reranker 配了地址但缺 API Key，重排序将不可用（检索仍可正常工作）"
fi

# ---------- 4. 前端构建 ----------
if [[ $SKIP_FRONTEND -eq 1 ]]; then
    step "前端构建（已跳过 --skip-frontend）"
    [[ -d "$ROOT/yusu_webui/dist" ]] || warn "yusu_webui/dist 不存在，浏览器访问将只有 API，没有界面"
elif [[ $HAS_BUN -eq 1 || $HAS_NPM -eq 1 ]]; then
    step "构建前端"
    pushd "$ROOT/yusu_webui" >/dev/null
    if [[ $HAS_BUN -eq 1 ]]; then
        bun install
        bun run build
    else
        npm install
        npm run build
    fi
    popd >/dev/null
    ok "前端构建完成 → yusu_webui/dist"
else
    step "前端构建（缺少 bun/npm，已跳过）"
    [[ -d "$ROOT/yusu_webui/dist" ]] || warn "yusu_webui/dist 不存在，浏览器访问将只有 API，没有界面"
fi

# ---------- 5. 启动服务 ----------
if [[ $NO_START -eq 1 ]]; then
    step "环境准备完成（--no-start，未启动服务）"
    printf "${C_GRAY}   手动启动：uv run --extra api python -m yusu_kb.server --host %s --port %s${C_OFF}\n" "$BIND_HOST" "$PORT"
    exit 0
fi

step "启动服务"
DISPLAY_HOST="$BIND_HOST"
[[ "$BIND_HOST" == "0.0.0.0" ]] && DISPLAY_HOST="127.0.0.1"
printf "${C_GREEN}   界面     http://%s:%s/${C_OFF}\n" "$DISPLAY_HOST" "$PORT"
printf "${C_GREEN}   API 文档 http://%s:%s/docs${C_OFF}\n" "$DISPLAY_HOST" "$PORT"
printf "${C_GREEN}   自检     http://%s:%s/api/health/models${C_OFF}\n" "$DISPLAY_HOST" "$PORT"
printf "${C_GRAY}   停止服务：Ctrl+C${C_OFF}\n"

SERVER_ARGS=(-m yusu_kb.server --host "$BIND_HOST" --port "$PORT")
[[ $DEV -eq 1 ]] && SERVER_ARGS+=(--reload)

if [[ $HAS_UV -eq 1 && ! -x "$VENV_PY" ]]; then
    exec uv run --extra api python "${SERVER_ARGS[@]}"
else
    exec "$PY_EXE" "${SERVER_ARGS[@]}"
fi
