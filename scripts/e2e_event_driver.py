"""端到端验证驱动（S8）：case_document 分块 → event 抽取 → 事件图谱 → 连通性门禁。

以确定性 hash embedding 驱动（embedding API 欠费时的验证替代）：
连通性门禁是图结构指标，与向量质量无关；事件抽取走真实 LLM。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import sys
from pathlib import Path

# 端到端语料目录：默认相对仓库根（yusu_data/uploads/<kb>），
# 可用环境变量 YUSU_E2E_WORK 指定任意绝对路径覆盖。
_WORK_DEFAULT = Path(__file__).resolve().parents[1] / "yusu_data" / "uploads" / "kb_zy42oxnykb"
WORK = Path(os.environ.get("YUSU_E2E_WORK", str(_WORK_DEFAULT)))

# 保证 repo root 在 sys.path（scripts/ 下运行）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("YUSU_DATA_DIR", str(Path(__file__).resolve().parents[1] / ".e2e_work"))
DATA_DIR = Path(os.environ["YUSU_DATA_DIR"])
DATA_DIR.mkdir(parents=True, exist_ok=True)

from yusu_kb.api.deps import create_chat_model
from yusu_kb.knowledge.graphs.graph_service import GraphService
from yusu_kb.knowledge.implementations.local_kb import (
    set_default_embedding_func,
)
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.embed import EmbeddingFunc
from yusu_kb.repositories import configure_repositories
from yusu_kb.storage.sqlite.engine import create_engine, init_db

EMBED_DIM = 64


async def _hash_embed(texts: list[str]) -> list[list[float]]:
    """确定性 hash embedding（验证用）：字符 n-gram 哈希叠加 → 64 维向量。"""
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * EMBED_DIM
        norm = "".join(text.split())
        for i in range(max(0, len(norm) - 2), -1, -1):
            gram = norm[i : i + 2]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            vec[h % EMBED_DIM] += 1.0
        total = sum(abs(v) for v in vec) or 1.0
        out.append([v / total for v in vec])
    return out


async def main() -> None:

    fake_embed = EmbeddingFunc(func=_hash_embed, embedding_dim=EMBED_DIM, model_name="e2e-hash-embed")
    set_default_embedding_func(fake_embed)

    engine = create_engine(DATA_DIR / "yusu.db")
    await init_db(engine)
    configure_repositories(engine)

    manager = KnowledgeBaseManager(DATA_DIR)
    await manager.load_all_metadata()

    # 复用已入库 KB（E2E_REUSE_KB=kb_xxx）→ 只重跑图谱构建，避免重复 parse/index
    kb_id = os.environ.get("E2E_REUSE_KB", "").strip()
    if kb_id:
        print(f"[1] reuse KB: {kb_id}")
        # 重跑前清空图状态（保留抽取缓存可复用；E2E_CLEAR_CACHE=1 强制重抽）
        from yusu_kb.knowledge.graphs.graph_service import GraphService as _GS

        _svc0 = _GS.get_instance(
            kb_id=kb_id, work_dir=DATA_DIR, embed_func=fake_embed, chat_model_fn=create_chat_model().call_collect
        )
        await _svc0.reset(kb_id, clear_extraction_result=bool(os.getenv("E2E_CLEAR_CACHE")), clear_config=False)
        print("[1b] graph state reset")
    else:
        kb = await manager.create_database(
            name="e2e-event-direct",
            description="事件驱动直连验证",
            additional_params={"chunk_preset_id": "case_document", "auto_build_graph": True},
        )
        kb_id = kb["kb_id"]
        print(f"[1] KB created: {kb_id}")

        # 上传 5 份 md（模拟 API 上传：写入 uploads 目录 + add_file_record）
        upload_dir = DATA_DIR / "uploads" / kb_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(WORK.glob("*.md")):
            target = upload_dir / src.name
            shutil.copy2(src, target)
            await manager.add_file_record(kb_id, str(target))
        print("[2] files uploaded")

        # parse + index（LLM 抽取/embedding 走 hash，不触发构建）
        from yusu_kb.repositories.knowledge_file_repository import (
            KnowledgeFileRepository,
        )

        files = await KnowledgeFileRepository().list_documents(kb_id=kb_id)
        for f in files:
            await manager.parse_file(kb_id, f.file_id)
        for f in files:
            await manager.index_file(kb_id, f.file_id)
        print(f"[3] parsed + indexed {len(files)} files")

    # 配置事件抽取器并构建图谱
    svc = GraphService.get_instance(
        kb_id=kb_id,
        work_dir=DATA_DIR,
        embed_func=fake_embed,
        chat_model_fn=create_chat_model().call_collect,
    )
    config = await svc.configure(
        kb_id,
        extractor_type="event",
        extractor_options={"concurrency_count": 4, "model_spec": "env:sensenova-6.8-flash-lite"},
    )
    print(f"[4] graph config locked: {config['extractor_type']}")
    await svc.build_pending_chunks(kb_id)
    for _ in range(240):  # 最长等 20 分钟
        await asyncio.sleep(5)
        status = await svc.get_status(kb_id)
        state = status.get("build_task_status")
        if state in ("completed", "failed", "cancelled"):
            break
    print(f"[5] build status: {state}, progress: {status.get('build_task_progress')}%")
    if state != "completed":
        print(f"    message: {status.get('build_task_message') or status.get('message')}")
        sys.exit(1)

    storage = svc.get_storage(kb_id)
    stats = storage.get_stats()
    connectivity = storage.get_connectivity()
    print("[6] graph stats:", stats)
    print("[7] connectivity:", connectivity)

    from yusu_kb.knowledge.graphs.connectivity import evaluate_gate

    gate_passed, gate_failures = evaluate_gate(connectivity)
    print("[8] gate:", {"passed": gate_passed, "failures": gate_failures})
    if not gate_passed:
        print("!! 连通性门禁未通过")
    else:
        print("== 连通性门禁通过 ==")

    # 检索冒烟：aquery 不报错、图通道有召回
    kb_local = manager.get_instance("local")
    await kb_local.refresh_stats(kb_id)
    chunks = await kb_local.aquery(
        kb_id=kb_id,
        query_text="张三与李四是什么关系？",
        top_k=5,
        query_params={"use_graph_retrieval": True},
    )
    print(f"[9] aquery ok, chunks={len(chunks)}")

    # 多跳检索冒烟
    try:
        paths = await svc.search_event_paths(kb_id, "资金往来", top_k=3)
        print(f"[10] search_event_paths ok, paths={len(paths)}")
        if paths:
            print("     首个路径 hops:", [h["event_id"] for h in paths[0]["hops"]])
    except Exception as exc:  # noqa: BLE001 - 多跳检索失败不阻塞验收
        print(f"[10] search_event_paths failed: {exc}")

    await engine.dispose()
    print("== E2E direct driver done ==")


if __name__ == "__main__":
    asyncio.run(main())
