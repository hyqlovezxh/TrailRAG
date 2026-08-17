"""Token 计数与按 token 预算截断工具。

设计参考 LightRAG `truncate_list_by_token_size`：
- tiktoken cl100k_base 优先（精确）
- 失败回退字符数/4（与 DescriptionMerger 一致）
- 按权重降序排序后再截断，保留高分项
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 模块级 tokenizer 单例（容错初始化）
_tokenizer = None
_tokenizer_init_failed = False


def get_tokenizer():
    """获取 tiktoken tokenizer 单例。失败时返回 None，后续回退字符数估算。"""
    global _tokenizer, _tokenizer_init_failed
    if _tokenizer is None and not _tokenizer_init_failed:
        try:
            import tiktoken

            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:  # noqa: BLE001 - tiktoken 初始化失败属预期回退路径，须兜底捕获
            logger.warning(f"tiktoken 初始化失败，回退字符数/4 估算: {e}")
            _tokenizer_init_failed = True
    return _tokenizer


def count_tokens(text: str) -> int:
    """token 计数。tiktoken 优先，失败回退字符数/4。"""
    if not text:
        return 0
    tokenizer = get_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:  # noqa: BLE001,S110 - encode 失败回退字符数估算，静默降级为预期行为
            pass
    return len(text) // 4


def truncate_list_by_token_size(
    items: list[T],
    key_fn: Callable[[T], str],
    max_token_size: int,
) -> list[T]:
    """按 token 累加截断列表。

    Args:
        items: 待截断的列表（调用方应先按权重降序排序）
        key_fn: item -> str，用于 token 计算
        max_token_size: 最大 token 预算；<= 0 返回空列表

    Returns:
        截断后的列表（保留前 N 项使累计 token 不超预算）
    """
    if max_token_size <= 0 or not items:
        return []
    result: list[T] = []
    tokens = 0
    for item in items:
        item_tokens = count_tokens(key_fn(item))
        if tokens + item_tokens > max_token_size:
            break
        tokens += item_tokens
        result.append(item)

    # GKB-15(b): 首条即超预算时截断首条而非返回空，避免极端边界下完全无结果
    if not result and items:
        first = items[0]
        first_text = key_fn(first)
        if isinstance(first_text, str):
            # 粗略按 4 char/token 截断到预算内
            char_limit = max(max_token_size * 4, 1)
            truncated_text = first_text[:char_limit]
            # 重建 item：对字符串列表直接替换；其他类型保留原样由调用方处理
            if isinstance(first, str):
                return [truncated_text]  # type: ignore[return-value]
        # 非字符串类型无法截断，保留原样
        return [first]
    return result


def truncate_section_context(heading_path: str, max_tokens: int = 256) -> str:
    """折叠章节路径到 token 预算内。

    策略：
    1. 若 token 数 <= max_tokens，直接返回
    2. 若层级 >= 3，折叠为 "first → … → leaf"
    3. 硬截断兜底（应对 token-dense 短路径）
    """
    if not heading_path or max_tokens <= 0:
        return heading_path
    if count_tokens(heading_path) <= max_tokens:
        return heading_path

    levels = heading_path.split(" → ")
    if len(levels) >= 3:
        heading_path = f"{levels[0]} → … → {levels[-1]}"
        if count_tokens(heading_path) <= max_tokens:
            return heading_path

    # 硬截断兜底：二分查找最大保留长度
    while count_tokens(heading_path) > max_tokens and len(heading_path) > 50:
        heading_path = heading_path[: len(heading_path) // 2] + "…"
    return heading_path