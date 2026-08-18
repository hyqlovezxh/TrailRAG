"""评估基准生成（逐字移植自 YUSU ``yuxi.knowledge.eval.benchmark_generation``）。

两处适配：
- ``select_graph_enhanced_chunks`` 的 MilvusGraphService PPR 换为本库
  ``rank_chunks_by_ppr``（NetworkX 图存储）。
- ``iter_generated_benchmark_items`` 的全局 ``select_model`` 改为注入的
  ``llm_fn``（按 model_spec 返回 LLM 适配器），测试可注入 FakeChat。
"""

import asyncio
import json
import random
import re
from collections.abc import AsyncIterator, Callable
from typing import Any

import json_repair

from yusu_kb.models.chat import create_chat_model
from yusu_kb.utils.logger import logger

DEFAULT_BENCHMARK_GENERATION_CONCURRENCY = 10
MAX_BENCHMARK_GENERATION_CONCURRENCY = 20
DEFAULT_GRAPH_EXPAND_TOP_K = 1
MAX_GRAPH_EXPAND_TOP_K = 3

# EVAL-3: query 去重用的标点/空白清理正则
_QUERY_DEDUP_PUNCT_RE = re.compile(r"[，。！？,.!?；;：:\s]+")


def _normalize_query_for_dedup(query: str) -> str:
    """EVAL-3: 将 query 归一化为去重 key（转小写 + 去标点空白），容忍标点差异。"""
    return _QUERY_DEDUP_PUNCT_RE.sub("", (query or "").lower())


GRAPH_SEED_DECAY = 0.9
GRAPH_PPR_DAMPING = 0.85
GRAPH_PPR_MAX_NODES = 10000


async def collect_kb_chunks(kb_instance: Any, kb_id: str) -> list[dict[str, Any]]:
    del kb_instance

    from yusu_kb.repositories.knowledge_chunk_repository import (
        KnowledgeChunkRepository,
    )

    return [
        {
            "id": chunk.chunk_id,
            "content": chunk.content or "",
            "file_id": chunk.file_id,
            "chunk_index": chunk.chunk_index,
            "graph_indexed": bool(chunk.graph_indexed),
            "ent_ids": chunk.ent_ids or [],
            "tags": chunk.tags or [],
            "extraction_result": chunk.extraction_result,
        }
        for chunk in await KnowledgeChunkRepository().list_by_kb_id(kb_id)
    ]


def clamp_neighbors_count(neighbors_count: int) -> int:
    return min(max(neighbors_count, 0), 10)


def normalize_generation_concurrency_count(value: Any) -> int:
    if value in (None, ""):
        return DEFAULT_BENCHMARK_GENERATION_CONCURRENCY
    return min(max(1, int(value)), MAX_BENCHMARK_GENERATION_CONCURRENCY)


def normalize_graph_expand_top_k(value: Any) -> int:
    if value in (None, ""):
        return DEFAULT_GRAPH_EXPAND_TOP_K
    return min(max(1, int(value)), MAX_GRAPH_EXPAND_TOP_K)


def _chunk_entity_ids(chunk: dict[str, Any]) -> list[str]:
    return [str(entity_id) for entity_id in chunk.get("ent_ids") or [] if entity_id]


def _is_anchor_chunk(candidate: dict[str, Any], anchor_chunk: dict[str, Any]) -> bool:
    metadata = candidate.get("metadata") or {}
    candidate_id = metadata.get("chunk_id")
    if candidate_id is not None and str(candidate_id) == str(anchor_chunk.get("id")):
        return True

    candidate_file_id = metadata.get("file_id")
    candidate_chunk_index = metadata.get("chunk_index")
    return candidate_file_id == anchor_chunk.get("file_id") and candidate_chunk_index == anchor_chunk.get("chunk_index")


async def select_neighbor_chunks_by_kb_query(
    *, kb_instance: Any, kb_id: str, anchor_chunk: dict[str, Any], neighbors_count: int
) -> list[dict[str, Any]]:
    if neighbors_count <= 0:
        return []

    anchor_content = anchor_chunk.get("content", "")
    if not anchor_content:
        return []

    candidates = await kb_instance.aquery(
        anchor_content,
        kb_id,
        search_mode="vector",
        final_top_k=neighbors_count + 3,
        use_reranker=False,
        similarity_threshold=0.0,
    )

    chunks = []
    for candidate in candidates:
        if _is_anchor_chunk(candidate, anchor_chunk):
            continue

        metadata = candidate.get("metadata") or {}
        chunk_id = metadata.get("chunk_id")
        content = candidate.get("content", "")
        if not chunk_id or not content:
            continue

        chunks.append(
            {
                "id": str(chunk_id),
                "content": content,
                "file_id": metadata.get("file_id"),
                "chunk_index": metadata.get("chunk_index"),
            }
        )
        if len(chunks) >= neighbors_count:
            break

    return chunks


async def select_graph_enhanced_chunks(
    *,
    kb_instance: Any,
    kb_id: str,
    anchor_chunk: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    context_count: int,
    graph_expand_top_k: int,
) -> list[dict[str, Any]] | None:
    if context_count <= 1:
        return [anchor_chunk]

    from yusu_kb.knowledge.graphs.graph_service import GraphService
    from yusu_kb.knowledge.graphs.ppr import rank_chunks_by_ppr

    anchor_entity_ids = _chunk_entity_ids(anchor_chunk)
    if not anchor_entity_ids:
        return None

    graph_service = GraphService.get_instance(kb_id=kb_id, work_dir=kb_instance.work_dir)
    storage = graph_service.get_storage(kb_id)
    if not storage.is_built():
        return None
    selected = [anchor_chunk]
    selected_ids = {str(anchor_chunk.get("id"))}
    seed_weights = {entity_id: 1.0 for entity_id in anchor_entity_ids}
    round_index = 1

    while len(selected) < context_count:
        for entity_id in anchor_entity_ids:
            seed_weights[entity_id] = 1.0

        ranked_chunks = rank_chunks_by_ppr(
            storage,
            seed_weights,
            max_nodes=GRAPH_PPR_MAX_NODES,
            top_k=max(context_count * 5, 20),
            damping=GRAPH_PPR_DAMPING,
            directed=False,
            # chunk_count_weight=0.0 使行为与 v1 一致（评估数据集生成不需要 chunk 计数信号）
            chunk_count_weight=0.0,
        )
        if not ranked_chunks:
            return None

        new_chunks = []
        for chunk_id, _ in ranked_chunks:
            chunk_id = str(chunk_id)
            if chunk_id in selected_ids:
                continue
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            new_chunks.append(chunk)
            if len(new_chunks) >= min(graph_expand_top_k, context_count - len(selected)):
                break

        if not new_chunks:
            return None

        new_weight = GRAPH_SEED_DECAY**round_index
        for chunk in new_chunks:
            selected.append(chunk)
            selected_ids.add(str(chunk.get("id")))
            for entity_id in _chunk_entity_ids(chunk):
                seed_weights[entity_id] = max(seed_weights.get(entity_id, 0.0), new_weight)
        round_index += 1

    return selected


def build_benchmark_generation_prompt(ctx_items: list[tuple[str, str]]) -> str:
    context_text = "\n\n".join([f"片段ID={cid}\n{content}" for cid, content in ctx_items])
    return (
        "你将基于以下上下文生成一个可由上下文准确回答的问题与标准答案。"
        "仅返回一个JSON对象，不要包含其他文字。"
        "键为 query、gold_answer、gold_chunk_ids。gold_chunk_ids 必须是上述上下文片段的ID子集。\n\n"
        "问题生成要求（Fix-4 消歧）：\n"
        "1. 问题必须包含至少一个具体实体（人名、地名、时间、数字、机构名等），"
        "使得在多人笔录知识库中可唯一定位到 gold chunk\n"
        "2. 禁止生成过于泛化的问题，例如：\n"
        "   - 禁止：'被讯问人叫什么名字？'（适用于任何笔录，无法消歧）\n"
        "   - 禁止：'案件发生在哪里？'（无具体定位信息）\n"
        "3. 应生成带上下文限定的问题，例如：\n"
        "   - 允许：'在张三的讯问笔录中，被讯问人叫什么名字？'\n"
        "   - 允许：'2024年3月在祥园路903室发生的案件中，固定工位分配给了谁？'\n"
        "4. 问题应能从上下文中直接找到明确答案，无需推理或跨文档关联\n\n"
        "上下文：\n" + context_text + "\n"
    )


# Fix-4: 问句特异性校验 - 确保问句包含至少一个具体实体可在多人笔录 KB 中消歧
# 启发式规则：
# 1. 包含中文姓氏 + 名字特征（2-4 字中文姓名）
# 2. 包含数字（时间、金额、编号等）
# 3. 包含具体地点关键词（路/号/室/村/镇/区/市等）
# 4. 包含具体时间表达（年/月/日/时/分）
_CHINESE_SURNAME_CHARS = (
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张"
    "孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎"
    "鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤"
    "滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟黄"
    "穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞"
    "熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭"
    "梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯"
)
_PLACE_KEYWORDS = ("路", "号", "室", "村", "镇", "区", "市", "县", "街", "楼", "栋", "单元", "层")
_TIME_KEYWORDS = ("年", "月", "日", "时", "分", "点")


def _validate_question_specificity(question: str) -> bool:
    """Fix-4: 校验问句是否包含至少一个具体实体，可在多人笔录 KB 中消歧。

    Args:
        question: 待校验的问句

    Returns:
        True 如果问句包含至少一个具体实体（人名/数字/地点/时间），False 否则
    """
    if not question or not question.strip():
        return False

    q = question.strip()

    # 规则 1: 包含数字（时间、金额、编号等）
    if any(ch.isdigit() for ch in q):
        return True

    # 规则 2: 包含具体地点关键词
    if any(kw in q for kw in _PLACE_KEYWORDS):
        return True

    # 规则 3: 包含具体时间表达
    if any(kw in q for kw in _TIME_KEYWORDS):
        return True

    # 规则 4: 包含中文姓氏 + 后续 1-3 字（疑似人名）
    # 简化启发式：包含姓氏字符即认为可能含人名（recall 优先，避免误拒）
    return any(ch in _CHINESE_SURNAME_CHARS for ch in q)


async def _generate_benchmark_item_once(
    *,
    kb_instance: Any,
    kb_id: str,
    all_chunks: list[dict[str, Any]],
    llm: Any,
    context_count: int,
    generation_mode: str,
    graph_expand_top_k: int,
    chunks_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if generation_mode == "graph_enhanced":
        graph_anchor_chunks = [
            chunk for chunk in all_chunks if chunk.get("graph_indexed") is True and _chunk_entity_ids(chunk)
        ]
        if not graph_anchor_chunks:
            raise ValueError("No graph indexed chunks with entities found in knowledge base")
        anchor_chunk = graph_anchor_chunks[random.randrange(len(graph_anchor_chunks))]
        ctx_chunks = await select_graph_enhanced_chunks(
            kb_instance=kb_instance,
            kb_id=kb_id,
            anchor_chunk=anchor_chunk,
            chunks_by_id=chunks_by_id,
            context_count=context_count,
            graph_expand_top_k=graph_expand_top_k,
        )
        if ctx_chunks is None:
            return None
    else:
        anchor_chunk = all_chunks[random.randrange(len(all_chunks))]
        neighbor_chunks = await select_neighbor_chunks_by_kb_query(
            kb_instance=kb_instance,
            kb_id=kb_id,
            anchor_chunk=anchor_chunk,
            neighbors_count=context_count - 1,
        )
        ctx_chunks = [anchor_chunk] + neighbor_chunks
    ctx_items = [(chunk["id"], chunk["content"]) for chunk in ctx_chunks]
    allowed_ids = {cid for cid, _ in ctx_items}

    # Fix-4: 问句特异性校验 + 重试机制
    # 最多重试 3 次，若 3 次均未通过特异性校验则接受最后一次结果（避免生成失败）
    specificity_max_retries = 3
    last_valid_item: dict[str, Any] | None = None

    for attempt in range(specificity_max_retries):
        try:
            resp = await llm.call(build_benchmark_generation_prompt(ctx_items), False)
            obj = json_repair.loads(resp.content if resp else "")
            if not isinstance(obj, dict):
                logger.warning(f"Generated JSON is not a dict: {type(obj).__name__}: {str(obj)[:200]}")
                return None
            query = obj.get("query")
            answer = obj.get("gold_answer")
            gold_ids = obj.get("gold_chunk_ids")
            if not query or not answer or not isinstance(gold_ids, list):
                logger.warning(f"Generated JSON missing fields or invalid format: {obj}")
                return None

            gold_ids = [str(item) for item in gold_ids if str(item) in allowed_ids]
            if not gold_ids:
                logger.warning("Generated gold_chunk_ids not found in allowed context")
                return None

            item = {"query": query, "gold_chunk_ids": gold_ids, "gold_answer": answer}
            last_valid_item = item

            # Fix-4: 特异性校验
            if _validate_question_specificity(query):
                return item

            logger.warning(
                f"Fix-4: question lacks specificity (attempt {attempt + 1}/{specificity_max_retries}): {query!r}, "
                f"retrying with disambiguation prompt"
            )
            # 校验未通过，继续重试
            continue

        except Exception as e:  # noqa: BLE001 - 单题生成失败返回 None 不抛出，与 YUSU 语义一致
            logger.warning(f"Benchmark generation failed for one item: {e}")
            return None

    # 3 次重试均未通过特异性校验，接受最后一次结果（避免生成失败）
    if last_valid_item is not None:
        logger.warning(
            f"Fix-4: question specificity validation failed after {specificity_max_retries} retries, "
            f"accepting last result: {last_valid_item['query']!r}"
        )
        return last_valid_item

    return None


async def iter_generated_benchmark_items(
    *,
    kb_instance: Any,
    kb_id: str,
    count: int,
    neighbors_count: int,
    llm_model_spec: str | None,
    concurrency_count: int = DEFAULT_BENCHMARK_GENERATION_CONCURRENCY,
    generation_mode: str = "vector",
    graph_expand_top_k: int = DEFAULT_GRAPH_EXPAND_TOP_K,
    progress_cb: Callable[[int, str], Any] | None = None,
    cancel_cb: Callable[[], Any] | None = None,
    llm_fn: Callable[[str], Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    if progress_cb:
        await progress_cb(5, "加载chunks")

    all_chunks = await collect_kb_chunks(kb_instance, kb_id)
    if not all_chunks:
        raise ValueError("No chunks found in knowledge base")
    chunks_by_id = {str(chunk["id"]): chunk for chunk in all_chunks if chunk.get("id") is not None}

    if generation_mode not in {"vector", "graph_enhanced"}:
        raise ValueError("Unsupported benchmark generation mode")
    graph_expand_top_k = normalize_graph_expand_top_k(graph_expand_top_k)

    if progress_cb:
        await progress_cb(15, "准备生成样本")

    if llm_fn is not None:
        llm = llm_fn(llm_model_spec or "")
    elif llm_model_spec:
        llm = create_chat_model(default_spec=llm_model_spec)
    else:
        # .env-only 部署（DB 无 provider 行）：回退 env 路径，与图谱抽取/模型层
        # 其他入口保持一致，避免"llm_model_spec 不能为空"硬卡开源上手。
        llm = create_chat_model()
    context_count = max(clamp_neighbors_count(neighbors_count), 1)
    max_attempts = max(count * 5, 50)
    worker_count = normalize_generation_concurrency_count(concurrency_count)
    actual_worker_count = min(worker_count, max(count, 1), max_attempts)
    generated = 0
    results: list[tuple[int, dict[str, Any]]] = []
    state_lock = asyncio.Lock()
    # EVAL-3 修复：query 文本去重，避免并发 worker 选中同一 anchor chunk 生成语义重复 QA。
    # normalize 后比较（去空白+转小写），容忍标点差异。
    seen_queries: set[str] = set()
    # 额外按 gold_chunk_ids 去重：随机锚点可能指向同一来源 chunk，导致两个
    # 生成项实际考察同一个 chunk（test_eval 中 all_ids 去重后数量不足）。
    seen_chunk_sets: set[frozenset[str]] = set()
    queue: asyncio.Queue[int] = asyncio.Queue()

    for attempt_no in range(max_attempts):
        queue.put_nowait(attempt_no)

    async def worker() -> None:
        nonlocal generated
        while True:
            if cancel_cb:
                await cancel_cb()
            async with state_lock:
                if generated >= count:
                    return
            try:
                attempt_no = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                item = await _generate_benchmark_item_once(
                    kb_instance=kb_instance,
                    kb_id=kb_id,
                    all_chunks=all_chunks,
                    llm=llm,
                    context_count=context_count,
                    generation_mode=generation_mode,
                    graph_expand_top_k=graph_expand_top_k,
                    chunks_by_id=chunks_by_id,
                )
                if item is None:
                    continue
                # EVAL-3: normalize query 文本并去重
                query_text = (item.get("query") or "").strip()
                normalized = _normalize_query_for_dedup(query_text)
                progress = None
                message = None
                async with state_lock:
                    if generated >= count:
                        continue
                    # query 为空或重复则丢弃（不计入 generated，让后续 attempt 重试）
                    if not normalized or normalized in seen_queries:
                        continue
                    # 按来源 chunk 去重：两个不同 query 仍可能指向同一 anchor chunk
                    chunk_key = frozenset(str(c) for c in (item.get("gold_chunk_ids") or []))
                    if chunk_key in seen_chunk_sets:
                        continue
                    seen_queries.add(normalized)
                    seen_chunk_sets.add(chunk_key)
                    generated += 1
                    results.append((attempt_no, item))
                    if progress_cb:
                        progress = int(99 * generated / max(count, 1))
                        message = f"已生成 {generated}/{count}"
                if progress_cb:
                    await progress_cb(progress, message)
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(actual_worker_count)]
    try:
        await asyncio.gather(*workers)
    except asyncio.CancelledError:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    except Exception:
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    for _, item in sorted(results, key=lambda pair: pair[0]):
        yield item


def dump_benchmark_item(item: dict[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"


__all__ = [
    "DEFAULT_BENCHMARK_GENERATION_CONCURRENCY",
    "MAX_BENCHMARK_GENERATION_CONCURRENCY",
    "_validate_question_specificity",
    "build_benchmark_generation_prompt",
    "clamp_neighbors_count",
    "collect_kb_chunks",
    "dump_benchmark_item",
    "iter_generated_benchmark_items",
    "normalize_generation_concurrency_count",
    "normalize_graph_expand_top_k",
    "select_graph_enhanced_chunks",
    "select_neighbor_chunks_by_kb_query",
]
