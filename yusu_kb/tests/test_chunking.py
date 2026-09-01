"""Tests for the chunking package (nlp, presets, parsers, semantic utils).

API matches the YUSU-ported signatures exactly.
"""

from __future__ import annotations

import numpy as np

from yusu_kb.knowledge.chunking import nlp
from yusu_kb.knowledge.chunking.dispatcher import chunk_file, chunk_markdown
from yusu_kb.knowledge.chunking.presets import (
    DEFAULT_CHUNK_PRESET_ID,
    get_chunk_preset_options,
    normalize_chunk_preset_id,
    resolve_chunk_processing_params,
)
from yusu_kb.knowledge.chunking.utils.semantic_utils import (
    agglomerative_cluster_labels,
    cosine_distance_matrix,
    find_best_num_clusters,
    semantic_chunking_with_auto_clusters,
    silhouette_score,
    split_sentences_chinese,
    split_sentences_english,
)

GENERAL_MD = """# 概述

LightRAG 是一个检索增强生成框架。

## 架构

它由知识图谱与向量检索组成。

### 数据流

文档经过解析、分块、实体抽取后写入图谱。

# 结论

混合检索模式效果最好。
"""


class TestNlp:
    def test_count_tokens(self):
        assert nlp.count_tokens("你好世界") == 4
        assert nlp.count_tokens("hello world") == 2

    def test_hard_split_by_token_limit(self):
        chunks = nlp.hard_split_by_token_limit("01234 56789 abcde f", chunk_token_num=2)
        assert chunks == ["01234 56789", "abcde f"]

    def test_naive_merge(self):
        lines = [f"内容{i:02d}第一段文字" for i in range(20)]
        chunks = nlp.naive_merge(lines, chunk_token_num=10)
        assert len(chunks) >= 5
        assert all(nlp.count_tokens(chunk) <= 17 for chunk in chunks)
        assert "".join(chunk.replace("\n", "") for chunk in chunks) == "".join(lines)

    def test_tree_merge_reduces_count(self):
        sections = [
            ("# 章节一", ""),
            ("内容甲", ""),
            ("# 章节二", ""),
            ("内容乙", ""),
            ("## 小节一", ""),
            ("内容丙", ""),
            ("# 章节三", ""),
            ("内容丁", ""),
        ]
        chunks = nlp.tree_merge(0, sections, depth=2)
        assert chunks
        assert len(chunks) <= len(sections)
        assert all(isinstance(chunk, str) and chunk.strip() for chunk in chunks)

    def test_hierarchical_merge_keeps_heading_titles(self):
        sections = [
            ("# 标题一", ""),
            ("内容甲", ""),
            ("# 标题二", ""),
            ("内容乙", ""),
        ]
        chunks = nlp.hierarchical_merge(0, sections, depth=5)
        assert chunks
        assert all("#" in chunk[0] for chunk in chunks)

    def test_remove_contents_table(self):
        sections = [
            ("目录", ""),
            ("第一章", ""),
            ("第二章", ""),
            ("# 正文", ""),
            ("内容", ""),
        ]
        nlp.remove_contents_table(sections)
        titles = [text for text, _ in sections]
        assert not any("目录" in title for title in titles)
        assert any("正文" in title for title in titles)

    def test_bullets_category(self):
        sections = ["# 章节一", "## 小节一", "# 章节二", "### 子节"]
        assert nlp.bullets_category(sections) >= 0

    def test_is_english(self):
        assert nlp.is_english("hello world foo bar")
        assert not nlp.is_english("你好世界")


class TestPresets:
    def test_default_preset_is_case_document(self):
        # 事件驱动版默认预设为 case_document（案件文档自动检测分块）
        assert DEFAULT_CHUNK_PRESET_ID == "case_document"

    def test_normalize_preset_id(self):
        assert normalize_chunk_preset_id("qa") == "qa"
        assert normalize_chunk_preset_id("unknown_preset") == "case_document"
        assert normalize_chunk_preset_id("") == "case_document"
        assert normalize_chunk_preset_id(None) == "case_document"

    def test_resolve_processing_params_precedence(self):
        params = resolve_chunk_processing_params(
            kb_additional_params={"chunk_preset_id": "qa", "chunk_parser_config": {"chunk_token_num": 200}},
            file_processing_params={},
            request_params={"chunk_preset_id": "book"},
        )
        assert params["chunk_preset_id"] == "book"
        assert params["chunk_engine_version"] == "ragflow_like_v1"
        assert "chunk_parser_config" in params

    def test_get_chunk_preset_options_has_all_presets(self):
        options = get_chunk_preset_options()
        preset_ids = {option["value"] for option in options}
        assert {"general", "qa", "book", "laws", "separator", "semantic"} <= preset_ids


class TestDispatcher:
    def test_chunk_markdown_general_preset(self):
        chunks = chunk_markdown(
            GENERAL_MD,
            file_id="file_abc",
            filename="doc.md",
            processing_params={
                "chunk_preset_id": "general",
                "chunk_parser_config": {"chunk_token_num": 400},
            },
        )
        assert chunks
        assert chunks[0]["chunk_id"] == "file_abc_chunk_0"
        assert chunks[0]["file_id"] == "file_abc"
        assert all(chunk["content"].strip() for chunk in chunks)
        assert all(
            {"id", "content", "file_id", "filename", "chunk_index", "chunk_id"} <= set(chunk)
            for chunk in chunks
        )

    def test_chunk_file_matches_chunk_markdown(self):
        params = {"chunk_preset_id": "general", "chunk_parser_config": {"chunk_token_num": 400}}
        direct = chunk_markdown(GENERAL_MD, file_id="f1", filename="doc.md", processing_params=params)
        via_file = chunk_file(GENERAL_MD, file_id="f1", filename="doc.md", processing_params=params)
        assert direct == via_file

    def test_chunk_markdown_qa_preset(self):
        qa_md = "# 问答\n\n## 问题一\n\n答案内容一\n\n## 问题二\n\n答案内容二"
        chunks = chunk_markdown(
            qa_md,
            file_id="f1",
            filename="qa.md",
            processing_params={"chunk_preset_id": "qa"},
        )
        assert chunks
        assert any("问题一" in chunk["content"] for chunk in chunks)

    def test_chunk_markdown_book_preset(self):
        chunks = chunk_markdown(
            GENERAL_MD,
            file_id="f1",
            filename="book.md",
            processing_params={"chunk_preset_id": "book"},
        )
        assert chunks

    def test_chunk_markdown_laws_preset(self):
        laws_md = "# 第一章 总则\n\n第一条 本法适用于……\n\n第二条 其他……"
        chunks = chunk_markdown(
            laws_md,
            file_id="f1",
            filename="law.md",
            processing_params={"chunk_preset_id": "laws"},
        )
        assert chunks

    def test_chunk_markdown_separator_preset(self):
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = chunk_markdown(
            text,
            file_id="f1",
            filename="sep.md",
            processing_params={
                "chunk_preset_id": "separator",
                "chunk_parser_config": {"chunk_token_num": 100, "overlapped_percent": 10},
            },
        )
        assert chunks

    def test_chunk_markdown_unknown_preset_falls_back_to_general(self):
        chunks = chunk_markdown(
            GENERAL_MD,
            file_id="f1",
            filename="doc.md",
            processing_params={"chunk_preset_id": "bogus"},
        )
        assert chunks


class TestSemanticUtils:
    def test_split_sentences_chinese(self):
        sentences = split_sentences_chinese("第一句。第二句！第三句？")
        assert len(sentences) >= 3

    def test_split_sentences_english(self):
        sentences = split_sentences_english("First sentence. Second one! Third?")
        assert len(sentences) == 3

    def test_cosine_distance_matrix(self):
        matrix = cosine_distance_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 0.0)
        assert 0.0 < matrix[0, 1] <= 2.0

    def test_agglomerative_cluster_labels(self):
        vectors = np.array([[1.0, 0.0], [0.9, 0.05], [0.0, 1.0], [0.05, 0.95]])
        labels = agglomerative_cluster_labels(vectors, n_clusters=2)
        assert len(set(labels.tolist())) == 2

    def test_silhouette_score_is_finite(self):
        vectors = np.array([[1.0, 0.0], [0.9, 0.05], [0.0, 1.0], [0.05, 0.95]])
        labels = agglomerative_cluster_labels(vectors, n_clusters=2)
        score = silhouette_score(vectors, labels)
        assert np.isfinite(score)
        assert -1.0 <= score <= 1.0

    def test_find_best_num_clusters(self):
        vectors = np.array(
            [[1.0, 0.0], [0.9, 0.05], [0.0, 1.0], [0.05, 0.95], [1.0, 0.1], [0.1, 0.9]]
        )
        k = find_best_num_clusters(vectors, min_clusters=2, max_clusters=3)
        assert k in (2, 3)

    def test_semantic_chunking_with_auto_clusters_no_embed(self):
        text = "第一句。第二句。第三句。第四句。"
        chunks = semantic_chunking_with_auto_clusters(text, embed_fn=None, token_count_fn=nlp.count_tokens)
        assert chunks
        assert all(chunk.strip() for chunk in chunks)

    def test_semantic_chunking_with_embed_fn(self):
        def dummy_embed(texts: list[str]) -> np.ndarray:
            rng = np.random.default_rng(0)
            return rng.normal(size=(len(texts), 4))

        text = "第一句。第二句。第三句。第四句。第五句。"
        chunks = semantic_chunking_with_auto_clusters(text, embed_fn=dummy_embed, token_count_fn=nlp.count_tokens)
        assert chunks
        assert all(chunk.strip() for chunk in chunks)


class TestSemanticParserPreset:
    def test_chunk_markdown_semantic_preset_with_dummy_embedding(self):
        from yusu_kb.knowledge.chunking.parsers.semantic import (
            chunk_markdown as semantic_chunk_markdown,
        )

        text = "第一句。\n第二句。\n第三句。\n\n# 标题\n\n第四句。"

        def dummy_embed(texts: list[str]) -> np.ndarray:
            return np.random.default_rng(1).normal(size=(len(texts), 4))

        chunks = semantic_chunk_markdown(text, {"chunk_token_num": 200}, embed_fn=dummy_embed)
        assert chunks

    def test_chunk_markdown_semantic_preset_without_embedding(self):
        from yusu_kb.knowledge.chunking.parsers.semantic import (
            chunk_markdown as semantic_chunk_markdown,
        )

        text = "第一句。\n第二句。\n第三句。"
        chunks = semantic_chunk_markdown(text, {"chunk_token_num": 200}, embed_fn=None)
        assert chunks


class TestMdParserUtils:
    def test_infer_heading_level(self):
        from yusu_kb.knowledge.chunking.utils.md_parser_utils import (
            infer_heading_level,
        )

        assert infer_heading_level("1.2.3 标题") == 3
        assert infer_heading_level("一、总则") == 1
        assert infer_heading_level("普通文本") == 1

    def test_get_title_path(self):
        from yusu_kb.knowledge.chunking.utils.md_parser_utils import get_title_path

        assert get_title_path(["概述", "架构"]) == "概述|架构"
        assert get_title_path(["", "架构", ""]) == "架构"

    def test_split_text_by_length_and_newline(self):
        from yusu_kb.knowledge.chunking.utils.md_parser_utils import (
            split_text_by_length_and_newline,
        )

        sentence = "一二三四五六七八九十。"
        text = sentence * 10
        parts = split_text_by_length_and_newline(
            text, max_length=20, embed_fn=None, token_count_fn=nlp.count_tokens
        )
        assert parts
        assert all(part.strip() for part in parts)
        assert "".join(parts) == text

    def test_extract_table_block(self):
        from markdown_it import MarkdownIt

        from yusu_kb.knowledge.chunking.utils.md_parser_utils import (
            extract_table_block,
        )

        md_text = "| a | b |\n|---|---|\n| 1 | 2 |"
        md = MarkdownIt("commonmark").enable("table")
        tokens = md.parse(md_text)
        table_open_index = next(i for i, token in enumerate(tokens) if token.type == "table_open")
        end_index, table_content = extract_table_block(tokens, table_open_index, md_text.split("\n"))
        assert end_index > table_open_index
        assert "|" in table_content