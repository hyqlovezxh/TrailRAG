"""Pydantic schemas for open / find / search tools.

Ported from YUSU ``yuxi.knowledge.schemas`` (demo edition).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchInputSchema(BaseModel):
    query_text: str = Field(description="查询文本")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="相似度阈值")
    additional_params: dict[str, Any] = Field(default_factory=dict, description="附加参数")


class SearchResultSchema(BaseModel):
    id: str = Field(description="结果 ID（通常是 chunk_id）")
    kb_id: str = Field(description="知识库 ID")
    file_id: str = Field(description="文件 ID")
    content: str = Field(description="结果内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class SearchOutputSchema(BaseModel):
    kb_id: str = Field(description="知识库 ID")
    results: list[SearchResultSchema] = Field(default_factory=list, description="搜索结果列表")


class FindInputSchema(BaseModel):
    file_id: str = Field(description="文件 ID")
    patterns: list[str] = Field(description="查找模式列表")
    use_regex: bool = Field(default=False, description="是否使用正则表达式")
    case_sensitive: bool = Field(default=False, description="是否区分大小写")
    max_windows: int = Field(default=5, ge=1, le=20, description="最大窗口数量")
    window_size: int = Field(default=80, ge=1, le=200, description="窗口大小（行数）")


class FindWindowSchema(BaseModel):
    start_line: int = Field(description="起始行号")
    end_line: int = Field(description="结束行号")
    matched_lines: list[int] = Field(default_factory=list, description="命中的行号")
    content: str = Field(description="窗口内容")


class FindOutputSchema(BaseModel):
    kb_id: str = Field(description="知识库 ID")
    file_id: str = Field(description="文件 ID")
    semantic: bool = Field(default=False, description="是否为语义查找")
    match_mode: str = Field(description="匹配模式：keyword/regex")
    total_matches: int = Field(description="总命中数")
    windows: list[FindWindowSchema] = Field(default_factory=list, description="查找窗口列表")


class OpenInputSchema(BaseModel):
    file_id: str = Field(description="文件 ID")
    offset: int = Field(default=0, ge=0, description="起始偏移（行）")
    limit: int = Field(default=800, ge=1, le=2000, description="窗口大小（行）")


class OpenOutputSchema(BaseModel):
    start_line: int = Field(description="起始行号（1 起）")
    end_line: int = Field(description="结束行号")
    total_lines: int = Field(description="文件总行数")
    content: str = Field(description="窗口内容")