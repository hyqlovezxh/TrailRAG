"""Uvicorn entry point: ``python -m yusu_kb.server [--host H] [--port P] [--reload]``.

Host/port fall back to ``YUSU_HOST`` / ``YUSU_PORT`` so container and systemd
deployments can be driven purely by environment variables.

Single-worker by design: build/ingest task progress lives in process memory
(``GraphService._build_states``), so running multiple uvicorn workers would make
task status invisible to whichever worker serves the polling request. Scale with
concurrency settings, not with worker processes.
"""

from __future__ import annotations

import argparse
import os

import uvicorn


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        # 环境变量拼错不应静默使用默认值——提示后回退，避免端口错位难排查
        print(f"[warn] {name}={raw!r} 不是合法整数，回退为 {default}")
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="语溯开源版 YUSU API 服务")
    parser.add_argument(
        "--host",
        default=_env_str("YUSU_HOST", "0.0.0.0"),
        help="监听地址（默认 0.0.0.0，可用 YUSU_HOST 覆盖）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("YUSU_PORT", 8920),
        help="监听端口（默认 8920，可用 YUSU_PORT 覆盖）",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开发模式：代码变更自动重载（生产环境请勿开启）",
    )
    parser.add_argument(
        "--log-level",
        default=_env_str("YUSU_LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="日志级别（默认 info，可用 YUSU_LOG_LEVEL 覆盖）",
    )
    args = parser.parse_args()
    uvicorn.run(
        "yusu_kb.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
