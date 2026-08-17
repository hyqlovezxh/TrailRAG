# 语溯开源版（TrailRAG / YUSU）一键部署脚本（Windows PowerShell）
# 功能：前置检查 → 安装后端依赖 → 配置 .env → 安装前端依赖 → 构建前端 → 启动后端
# 用法：在仓库根目录执行  .\scripts\start.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot -Resolve
Set-Location ".."

$Root = (Get-Location).Path
Write-Host "== 仓库根目录: $Root" -ForegroundColor Cyan

# ---------- 1. 前置检查 ----------
function Test-Cmd([string]$Name, [string]$Hint) {
    if (Get-Command $Name -ErrorAction SilentlyContinue) {
        Write-Host "[OK] 找到 $Name" -ForegroundColor Green
        return $true
    }
    Write-Host "[跳过] 未找到 $Name —— $Hint" -ForegroundColor Yellow
    return $false
}

$hashUv  = Test-Cmd "uv" "可改用 py -m pip 安装依赖，或先安装 https://docs.astral.sh/uv/"
$hashBun = Test-Cmd "bun" "请先安装 https://bun.sh/（`npm i -g bun` 亦可）"
$hashPy  = Test-Cmd "py" "请安装 Python 3.11+（勾选 Add to PATH）"

if (-not $hashPy) {
    Write-Host "缺少 Python，无法继续。" -ForegroundColor Red
    exit 1
}

# ---------- 2. 后端依赖 ----------
if ($hashUv) {
    Write-Host "== 安装后端依赖 (uv sync --extra api --extra test)" -ForegroundColor Cyan
    uv sync --extra api --extra test
} else {
    Write-Host "== 安装后端依赖 (pip install -e .[api,test])" -ForegroundColor Cyan
    py -m pip install -e ".[api,test]"
}

# ---------- 3. .env 配置 ----------
if (-not (Test-Path ".env")) {
    if (-not (Test-Path "env.example")) {
        Write-Host "缺少 env.example，无法生成 .env。" -ForegroundColor Red
        exit 1
    }
    Copy-Item "env.example" ".env"
    Write-Host "已生成 .env（来自 env.example）。" -ForegroundColor Green
}
Write-Host "请确保 .env 中已填写 YUSU_LLM_API_KEY / YUSU_EMBED_API_KEY 等密钥。" -ForegroundColor Yellow

# ---------- 4. 前端依赖 + 构建 ----------
if ($hashBun) {
    Push-Location "yusu_webui"
    Write-Host "== 安装前端依赖 & 构建 (bun install && bun run build)" -ForegroundColor Cyan
    bun install
    bun run build
    Pop-Location
} else {
    Write-Host "未找到 bun，跳过前端构建；后端将仅提供静态文件（若已有 yusu_webui/dist）。" -ForegroundColor Yellow
}

# ---------- 5. 启动后端 ----------
Write-Host "== 启动后端: http://0.0.0.0:8920  (Ctrl+C 停止)" -ForegroundColor Cyan
if ($hashUv) {
    uv run --extra api python -m yusu_kb.server --host 0.0.0.0 --port 8920
} else {
    py -m yusu_kb.server --host 0.0.0.0 --port 8920
}