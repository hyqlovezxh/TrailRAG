# 语溯开源版（YUSU / TrailRAG）一键启动脚本 —— Windows PowerShell
#
# 流程：环境检查 → 后端依赖 → .env 准备与密钥校验 → 前端依赖与构建 → 启动服务
#
# 用法（在仓库任意位置执行均可）：
#   .\scripts\start.ps1                     # 完整流程并启动
#   .\scripts\start.ps1 -Port 9000          # 换端口
#   .\scripts\start.ps1 -SkipFrontend       # 跳过前端构建（复用已有 dist）
#   .\scripts\start.ps1 -SkipInstall        # 跳过依赖安装（环境已就绪）
#   .\scripts\start.ps1 -Dev                # 开发模式（后端代码热重载）
#   .\scripts\start.ps1 -NoStart            # 只准备环境，不启动服务
#   .\scripts\start.ps1 -Force              # 密钥未填写也强行启动

[CmdletBinding()]
param(
    [int]$Port = 8920,
    [string]$BindHost = "0.0.0.0",
    [switch]$SkipFrontend,
    [switch]$SkipInstall,
    [switch]$Dev,
    [switch]$NoStart,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Text) { Write-Host "`n== $Text" -ForegroundColor Cyan }
function Write-Ok([string]$Text) { Write-Host "   [OK]   $Text" -ForegroundColor Green }
function Write-Warn2([string]$Text) { Write-Host "   [WARN] $Text" -ForegroundColor Yellow }
function Write-Err([string]$Text) { Write-Host "   [FAIL] $Text" -ForegroundColor Red }

function Test-Cmd([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# ---------- 0. 定位仓库根目录 ----------
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "语溯开源版 YUSU · 一键启动" -ForegroundColor Magenta
Write-Host "仓库根目录: $Root"

# ---------- 1. 环境检查 ----------
Write-Step "环境检查"

$hasUv = Test-Cmd "uv"
if ($hasUv) { Write-Ok "uv 已安装（推荐的依赖管理方式）" }
else { Write-Warn2 "未找到 uv，将回退 pip；建议安装：https://docs.astral.sh/uv/" }

$pyExe = $null
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    $pyExe = $venvPy
    Write-Ok "复用现有虚拟环境 .venv"
} elseif (Test-Cmd "py") {
    $pyExe = "py"
} elseif (Test-Cmd "python") {
    $pyExe = "python"
}

if (-not $hasUv -and -not $pyExe) {
    Write-Err "未找到 Python 3.11+，也未找到 uv。请安装 Python（勾选 Add to PATH）后重试。"
    exit 1
}

if ($pyExe) {
    $pyVer = & $pyExe -c "import sys;print('.'.join(map(str,sys.version_info[:3])))" 2>$null
    if ($pyVer) {
        $major, $minor = $pyVer.Split('.')[0..1]
        if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 11)) {
            Write-Err "Python $pyVer 版本过低，本项目要求 3.11+。"
            exit 1
        }
        Write-Ok "Python $pyVer"
    }
}

$hasBun = Test-Cmd "bun"
$hasNpm = Test-Cmd "npm"
if ($hasBun) { Write-Ok "bun 已安装（前端构建）" }
elseif ($hasNpm) { Write-Warn2 "未找到 bun，将使用 npm 构建前端（较慢）" }
else { Write-Warn2 "未找到 bun / npm，将跳过前端构建" }

# ---------- 2. 后端依赖 ----------
if ($SkipInstall) {
    Write-Step "后端依赖（已跳过 -SkipInstall）"
} else {
    Write-Step "安装后端依赖"
    if ($hasUv) {
        uv sync --extra api
        if ($LASTEXITCODE -ne 0) { Write-Err "uv sync 失败"; exit 1 }
    } else {
        & $pyExe -m pip install -e ".[api]"
        if ($LASTEXITCODE -ne 0) { Write-Err "pip install 失败"; exit 1 }
    }
    Write-Ok "后端依赖就绪"
    if (Test-Path $venvPy) { $pyExe = $venvPy }
}

# ---------- 3. .env 准备与密钥校验 ----------
Write-Step "配置文件 .env"

$envPath = Join-Path $Root ".env"
if (-not (Test-Path $envPath)) {
    $examplePath = Join-Path $Root "env.example"
    if (-not (Test-Path $examplePath)) {
        Write-Err "缺少 env.example，无法生成 .env"
        exit 1
    }
    Copy-Item $examplePath $envPath
    Write-Ok "已从 env.example 生成 .env"
} else {
    Write-Ok ".env 已存在"
}

# 逐行解析 .env，仅取本脚本校验所需的键（不打印任何密钥内容）
$envMap = @{}
foreach ($line in Get-Content $envPath -Encoding UTF8) {
    $trimmed = $line.Trim()
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
    $idx = $trimmed.IndexOf("=")
    if ($idx -lt 1) { continue }
    $key = $trimmed.Substring(0, $idx).Trim()
    $val = $trimmed.Substring($idx + 1).Trim().Trim('"').Trim("'")
    $envMap[$key] = $val
}

$missing = @()
foreach ($required in @("YUSU_LLM_API_KEY", "YUSU_LLM_BASE_URL", "YUSU_LLM_MODEL")) {
    if (-not $envMap.ContainsKey($required) -or $envMap[$required] -eq "") { $missing += $required }
}
# Embedding 未单独配置时回退 LLM 配置，因此只在显式给了 base_url 却漏了 key 时报错
if ($envMap["YUSU_EMBED_BASE_URL"] -and -not $envMap["YUSU_EMBED_API_KEY"]) {
    $missing += "YUSU_EMBED_API_KEY"
}

if ($missing.Count -gt 0) {
    Write-Warn2 ("以下必填项尚未填写：" + ($missing -join ", "))
    Write-Host "   请编辑 $envPath 后重试（密钥只写入 .env，切勿提交到仓库）。" -ForegroundColor Yellow
    if (-not $Force) {
        Write-Err "配置不完整，已终止。确认要继续可加 -Force。"
        exit 1
    }
    Write-Warn2 "-Force 已指定，继续启动（模型调用会失败）"
} else {
    Write-Ok "模型配置项完整"
}

if ($envMap["YUSU_RERANK_BASE_URL"] -and -not $envMap["YUSU_RERANK_API_KEY"]) {
    Write-Warn2 "Reranker 配了地址但缺 API Key，重排序将不可用（检索仍可正常工作）"
}

# ---------- 4. 前端构建 ----------
$distPath = Join-Path $Root "yusu_webui\dist"
if ($SkipFrontend) {
    Write-Step "前端构建（已跳过 -SkipFrontend）"
    if (-not (Test-Path $distPath)) { Write-Warn2 "yusu_webui\dist 不存在，浏览器访问将只有 API，没有界面" }
} elseif ($hasBun -or $hasNpm) {
    Write-Step "构建前端"
    Push-Location (Join-Path $Root "yusu_webui")
    try {
        if ($hasBun) {
            bun install
            if ($LASTEXITCODE -ne 0) { throw "bun install 失败" }
            bun run build
            if ($LASTEXITCODE -ne 0) { throw "bun run build 失败" }
        } else {
            npm install
            if ($LASTEXITCODE -ne 0) { throw "npm install 失败" }
            npm run build
            if ($LASTEXITCODE -ne 0) { throw "npm run build 失败" }
        }
    } finally {
        Pop-Location
    }
    Write-Ok "前端构建完成 → yusu_webui\dist"
} else {
    Write-Step "前端构建（缺少 bun/npm，已跳过）"
    if (-not (Test-Path $distPath)) { Write-Warn2 "yusu_webui\dist 不存在，浏览器访问将只有 API，没有界面" }
}

# ---------- 5. 启动服务 ----------
if ($NoStart) {
    Write-Step "环境准备完成（-NoStart，未启动服务）"
    Write-Host "   手动启动：uv run --extra api python -m yusu_kb.server --host $BindHost --port $Port" -ForegroundColor Gray
    exit 0
}

Write-Step "启动服务"
$displayHost = if ($BindHost -eq "0.0.0.0") { "127.0.0.1" } else { $BindHost }
Write-Host "   界面   http://${displayHost}:$Port/" -ForegroundColor Green
Write-Host "   API 文档 http://${displayHost}:$Port/docs" -ForegroundColor Green
Write-Host "   自检   http://${displayHost}:$Port/api/health/models" -ForegroundColor Green
Write-Host "   停止服务：Ctrl+C" -ForegroundColor Gray

$serverArgs = @("-m", "yusu_kb.server", "--host", $BindHost, "--port", "$Port")
if ($Dev) { $serverArgs += "--reload" }

if ($hasUv -and -not (Test-Path $venvPy)) {
    uv run --extra api python @serverArgs
} else {
    & $pyExe @serverArgs
}
