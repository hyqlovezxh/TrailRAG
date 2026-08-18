"""语溯开源版环境自检：Python / 依赖 / .env / 数据目录 / 模型连通性。

用法（仓库根目录执行）：
    uv run --extra api python scripts/doctor.py
    # 或
    .venv/Scripts/python.exe scripts/doctor.py        # Windows
    .venv/bin/python scripts/doctor.py                # Linux / macOS

只做只读探测：不写业务数据、不改配置、不打印任何完整密钥。
退出码 0 = 全部通过；1 = 存在阻断性问题。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def section(title: str) -> None:
    print(f"\n{_c('36', '== ' + title)}")


def ok(text: str) -> None:
    print(_c("32", f"   [OK]   {text}"))


def warn(text: str) -> None:
    print(_c("33", f"   [WARN] {text}"))


def fail(text: str) -> None:
    print(_c("31", f"   [FAIL] {text}"))


def mask(value: str) -> str:
    """Mask a secret so problems stay diagnosable without leaking the key."""
    if not value:
        return "(空)"
    if len(value) <= 8:
        return value[:2] + "*" * (len(value) - 2)
    return f"{value[:4]}...{value[-4:]} (长度 {len(value)})"


ERRORS: list[str] = []
WARNINGS: list[str] = []


def record_fail(text: str) -> None:
    fail(text)
    ERRORS.append(text)


def record_warn(text: str) -> None:
    warn(text)
    WARNINGS.append(text)


def check_python() -> None:
    section("Python 运行时")
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info < (3, 11):  # noqa: UP036 - 脚本在任意用户机器运行，版本检查必要
        record_fail(f"Python {version} 版本过低，要求 3.11+")
    else:
        ok(f"Python {version}")
    print(f"          解释器: {sys.executable}")


def check_dependencies() -> None:
    section("关键依赖")
    required = {
        "fastapi": "API 框架",
        "uvicorn": "ASGI 服务器",
        "httpx": "模型 HTTP 客户端",
        "sqlalchemy": "数据库 ORM",
        "aiosqlite": "SQLite 异步驱动",
        "networkx": "知识图谱存储",
        "scipy": "PPR 多跳检索（缺失会让图谱召回静默降级）",
        "nano_vectordb": "向量库",
        "jieba": "中文分词（BM25 关键词召回）",
        "pymupdf": "PDF 解析",
        "docx2txt": "Word (.docx) 解析",
        "openpyxl": "Excel 解析",
        "bs4": "HTML 解析（pip 包名 beautifulsoup4，导入模块 bs4）",
        "dotenv": ".env 加载（python-dotenv）",
    }
    for module, purpose in required.items():
        try:
            __import__(module)
        except ImportError:
            record_fail(f"缺少 {module} —— {purpose}")
        else:
            ok(f"{module:<14} {purpose}")


def check_env() -> dict[str, str]:
    section("配置文件 .env")
    env_path = ROOT / ".env"
    if not env_path.is_file():
        record_fail(f"未找到 {env_path}，请复制 env.example 为 .env 并填写密钥")
        return {}

    from dotenv import load_dotenv

    load_dotenv(env_path)
    ok(f"已加载 {env_path}")

    values = {
        key: (os.getenv(key) or "").strip()
        for key in (
            "YUSU_LLM_BASE_URL",
            "YUSU_LLM_API_KEY",
            "YUSU_LLM_MODEL",
            "YUSU_EMBED_BASE_URL",
            "YUSU_EMBED_API_KEY",
            "YUSU_EMBED_MODEL",
            "YUSU_EMBED_DIM",
            "YUSU_RERANK_BASE_URL",
            "YUSU_RERANK_API_KEY",
            "YUSU_RERANK_MODEL",
            "YUSU_DATA_DIR",
            "YUSU_API_KEY",
        )
    }

    for key in ("YUSU_LLM_BASE_URL", "YUSU_LLM_MODEL"):
        if values[key]:
            ok(f"{key} = {values[key]}")
        else:
            record_fail(f"{key} 未配置")
    if values["YUSU_LLM_API_KEY"]:
        ok(f"YUSU_LLM_API_KEY = {mask(values['YUSU_LLM_API_KEY'])}")
    else:
        record_fail("YUSU_LLM_API_KEY 未配置")

    if values["YUSU_EMBED_BASE_URL"] or values["YUSU_EMBED_MODEL"]:
        ok(f"YUSU_EMBED_BASE_URL = {values['YUSU_EMBED_BASE_URL'] or '(回退 LLM 配置)'}")
        ok(f"YUSU_EMBED_MODEL = {values['YUSU_EMBED_MODEL'] or '(回退 LLM 配置)'}")
        if values["YUSU_EMBED_BASE_URL"] and not values["YUSU_EMBED_API_KEY"]:
            record_fail("配置了 YUSU_EMBED_BASE_URL 但缺 YUSU_EMBED_API_KEY")
        elif values["YUSU_EMBED_API_KEY"]:
            ok(f"YUSU_EMBED_API_KEY = {mask(values['YUSU_EMBED_API_KEY'])}")
    else:
        record_warn("未单独配置 Embedding，将回退使用 LLM 的 base_url / api_key")

    if values["YUSU_RERANK_MODEL"]:
        ok(f"YUSU_RERANK_MODEL = {values['YUSU_RERANK_MODEL']}")
        if not values["YUSU_RERANK_API_KEY"]:
            record_warn("Reranker 配了模型但缺 API Key，重排序不可用（检索仍可工作）")
    else:
        print("          Reranker 未配置（可选，仅影响重排序精度）")

    print(f"          YUSU_DATA_DIR = {values['YUSU_DATA_DIR'] or './yusu_data (默认)'}")
    print(f"          API 鉴权 = {'启用' if values['YUSU_API_KEY'] else '未启用（本地开发可接受）'}")
    return values


def check_data_dir(values: dict[str, str]) -> None:
    section("数据目录")
    raw = values.get("YUSU_DATA_DIR") or "./yusu_data"
    data_dir = Path(raw) if Path(raw).is_absolute() else ROOT / raw
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".doctor_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        record_fail(f"数据目录不可写：{data_dir} ({exc})")
        return
    ok(f"{data_dir} 可读写")


async def _probe_models() -> None:
    from yusu_kb.models.chat import create_chat_model
    from yusu_kb.models.embed import create_embedding_model
    from yusu_kb.models.rerank import create_reranker

    # --- Embedding ---
    embed = create_embedding_model()
    try:
        embed_ok, message = await embed.test_connection()
        if embed_ok:
            ok(f"Embedding 可用 —— {message}")
        else:
            record_fail(f"Embedding 不可用 —— {message}")
    except Exception as exc:  # noqa: BLE001 - 自检须报告失败而非抛栈
        record_fail(f"Embedding 探测异常 —— {exc}")
    finally:
        await embed.close_async_client()

    # --- Chat ---
    try:
        chat = create_chat_model()
        response = await chat.call([{"role": "user", "content": "回复数字 1"}], stream=False)
        if response and getattr(response, "content", None):
            ok(f"Chat 可用 —— 模型 {getattr(chat, 'model', '?')}")
        else:
            record_fail("Chat 响应为空")
    except Exception as exc:  # noqa: BLE001 - 自检须报告失败而非抛栈
        record_fail(f"Chat 探测异常 —— {exc}")

    # --- Rerank（可选） ---
    reranker = create_reranker()
    if reranker is None:
        print("          Rerank 未配置，跳过探测")
        return
    try:
        rerank_ok, message = await reranker.test_connection()
        if rerank_ok:
            ok(f"Rerank 可用 —— {message}")
        else:
            record_warn(f"Rerank 不可用 —— {message}")
    except Exception as exc:  # noqa: BLE001 - 自检须报告失败而非抛栈
        record_warn(f"Rerank 探测异常 —— {exc}")
    finally:
        await reranker.aclose()


def check_models(values: dict[str, str]) -> None:
    section("模型连通性（真实调用，会产生少量 token 消耗）")
    if not values.get("YUSU_LLM_API_KEY"):
        record_warn("缺少 API Key，跳过模型探测")
        return
    try:
        asyncio.run(_probe_models())
    except Exception as exc:  # noqa: BLE001 - 自检须报告失败而非抛栈
        record_fail(f"模型探测流程异常 —— {exc}")


def check_frontend() -> None:
    section("前端构建产物")
    dist = ROOT / "yusu_webui" / "dist"
    index = dist / "index.html"
    if index.is_file():
        ok(f"{dist} 已就绪（服务启动后可直接访问界面）")
    else:
        record_warn("yusu_webui/dist 缺失，浏览器访问只有 API；执行 bun run build 生成")


def main() -> int:
    print(_c("35", "语溯开源版 YUSU · 环境自检"))
    print(f"仓库根目录: {ROOT}")

    check_python()
    check_dependencies()
    values = check_env()
    check_data_dir(values)
    check_frontend()
    check_models(values)

    section("自检结论")
    if ERRORS:
        fail(f"{len(ERRORS)} 项阻断性问题，需修复后才能正常使用：")
        for item in ERRORS:
            print(f"          - {item}")
    if WARNINGS:
        warn(f"{len(WARNINGS)} 项提醒（不阻断运行）：")
        for item in WARNINGS:
            print(f"          - {item}")
    if not ERRORS and not WARNINGS:
        ok("全部检查通过，可直接启动服务")
    elif not ERRORS:
        ok("无阻断性问题，可启动服务")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    raise SystemExit(main())
