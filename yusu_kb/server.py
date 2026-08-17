"""Uvicorn entry point: ``python -m yusu_kb.server [--host H] [--port P]``."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="语溯开源版 YUSU API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8920, help="监听端口（默认 8920）")
    args = parser.parse_args()
    uvicorn.run("yusu_kb.api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()