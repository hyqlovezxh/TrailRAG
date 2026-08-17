"""Markdown parsing helpers for semantic chunking.

Ported from YUSU ``yuxi.knowledge.chunking.ragflow_like.utils.md_parser_utils``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .semantic_utils import semantic_chunking_with_auto_clusters


def infer_heading_level(title: str) -> int:
    """Infer a heading level (1-6) from a title text:
    1. numeric sequences like "1.", "1.1", "1.2.3" map dots to levels;
    2. Chinese numerals like "一、" map to level 1;
    3. otherwise default to level 1.
    """
    m = re.match(r"^\s*(\d+(?:\.\d+)*)[.)、]?\s*", title)
    if m:
        return max(1, min(len(m.group(1).split(".")), 6))
    m_zh = re.match(r"^\s*[一二三四五六七八九十百千]+[、.]\s*", title)
    if m_zh:
        return 1
    return 1


def get_title_path(stack: list[str]) -> str:
    """Build a title path from the title stack, joined by "|"."""
    return "|".join([t for t in stack if t])


def extract_table_block(tokens: list[Any], i: int, original_lines: list[str]) -> tuple[int, str]:
    """Extract a complete table block from the token stream and original text:
    1. locate the table start line via the current token's ``map``;
    2. scan forward for ``table_close``;
    3. resolve the end line (close token ``map``, next token with ``map``,
       or a heuristic scan of markdown table lines);
    4. return the close token index and the joined original table text.
    """
    token = tokens[i]
    table_start = token.map[0] if token.map else 0
    j = i + 1
    while j < len(tokens) and tokens[j].type != "table_close":
        j += 1
    if j < len(tokens):
        end_token = tokens[j]
        if end_token.map and end_token.map[1] is not None:
            table_end = end_token.map[1]
        else:
            table_end = None
            for k in range(j + 1, len(tokens)):
                if tokens[k].map and tokens[k].map[0] is not None:
                    table_end = tokens[k].map[0]
                    break
            if table_end is None:
                table_end = table_start + 1
                for line_idx in range(table_start, len(original_lines)):
                    line = original_lines[line_idx].strip()
                    if not line or not (line.startswith("|") or "|" in line):
                        table_end = line_idx
                        break
    else:
        table_end = table_start + 1
        for line_idx in range(table_start, len(original_lines)):
            line = original_lines[line_idx].strip()
            if not line or not (line.startswith("|") or "|" in line):
                table_end = line_idx
                break
    return j, "\n".join(original_lines[table_start:table_end])


def split_text_by_length_and_newline(
    text: str, max_length: int, embed_fn: Callable[[list[str]], Any] | None, token_count_fn: Callable[[str], int]
) -> list[str]:
    """Hierarchical text splitting strategy."""
    chunks = []

    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_token_count = token_count_fn(paragraph)

        # Paragraph within the limit becomes its own chunk
        if paragraph_token_count <= max_length:
            chunks.append(paragraph)
            continue

        # Otherwise split the paragraph line by line
        lines = paragraph.split("\n")
        current_chunk_lines = []
        current_chunk_tokens = 0

        for line in lines:
            line = line.strip()
            if not line:  # skip blank lines
                continue

            line_token_count = token_count_fn(line)  # token count of this line
            # A newline token is added for every line after the first
            added_tokens = line_token_count + (1 if current_chunk_lines else 0)
            # A single overlong line becomes its own chunk(s)
            if line_token_count > max_length:
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                    current_chunk_lines = []
                    current_chunk_tokens = 0

                sub_chunks = semantic_chunking_with_auto_clusters(
                    line, embed_fn=embed_fn, token_count_fn=token_count_fn, max_chunk_size=max_length
                )
                chunks.extend(sub_chunks)
            # Merging would overflow: start a new chunk with this line
            elif current_chunk_tokens + added_tokens > max_length:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = [line]
                current_chunk_tokens = line_token_count
            # Safe to append to the current chunk
            else:
                current_chunk_lines.append(line)
                current_chunk_tokens += added_tokens  # update current chunk tokens
        # Flush the trailing lines
        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))

    return chunks