# 语溯开源版（YUSU / TrailRAG）常用任务
# Windows 用户请直接使用 scripts\start.ps1（无需 make）

.PHONY: help install web doctor run dev test lint fix docker-up docker-down docker-logs clean

PY ?= uv run --extra api python
PORT ?= 8920
BIND ?= 0.0.0.0

help: ## 显示可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## 安装后端依赖（含测试依赖）
	uv sync --extra api --extra test

web: ## 安装并构建前端（产物 yusu_webui/dist）
	cd yusu_webui && bun install && bun run build

doctor: ## 环境与模型连通性自检
	$(PY) scripts/doctor.py

run: ## 启动服务（生产模式）
	$(PY) -m yusu_kb.server --host $(BIND) --port $(PORT)

dev: ## 启动服务（开发模式，代码热重载）
	$(PY) -m yusu_kb.server --host 127.0.0.1 --port $(PORT) --reload

test: ## 运行后端单元测试
	uv run --extra api --extra test python -m pytest yusu_kb/tests -q

lint: ## 静态检查
	uv run ruff check yusu_kb

fix: ## 自动修复可修复的 lint 问题
	uv run ruff check --fix yusu_kb

docker-up: ## 构建并启动容器
	docker compose up -d --build

docker-down: ## 停止容器（数据保留在 volume）
	docker compose down

docker-logs: ## 跟踪容器日志
	docker compose logs -f yusu

clean: ## 清理构建缓存（不动 yusu_data 与 .env）
	rm -rf .pytest_cache .ruff_cache **/__pycache__ yusu_kb/**/__pycache__
