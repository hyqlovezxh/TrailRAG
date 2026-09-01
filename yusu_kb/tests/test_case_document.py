"""S3 测试：case_document 分块移植 + doc_type 落库。

覆盖案件文档的四种分块路径（笔录问答 / 聊天 30 分钟时间窗 + 时间段锚定 /
CSV 键值对 / 表格表头注入）与 doc_type 检测（双路径路由判据）。
"""

from __future__ import annotations

from yusu_kb.knowledge.chunking.dispatcher import _detect_chunk_doc_type, chunk_markdown
from yusu_kb.knowledge.chunking.parsers.case_document import (
    _chunk_chat_record,
    _chunk_csv_table,
    _chunk_transcript,
    _detect_document_type,
)
from yusu_kb.knowledge.chunking.presets import normalize_chunk_preset_id
from yusu_kb.storage.sqlite.engine import create_engine, init_db

# ---------------------------------------------------------------- doc_type 检测


def test_detect_document_type_transcript():
    assert _detect_document_type("询问笔录.md", "问：你叫什么名字？答：张三。") == "transcript"
    assert _detect_document_type("会议记录.md", "问：昨天去哪了？") == "transcript"
    assert _detect_document_type("未命名.txt", "问：请陈述经过。\n答：好的。") == "transcript"


def test_detect_document_type_chat():
    assert _detect_document_type("私聊记录.md", "随便聊聊") == "chat_record"
    assert _detect_document_type("群聊导出.txt", "随便聊聊") == "chat_record"
    assert (
        _detect_document_type("聊天.txt", "2025-10-05 14:20 张三：在吗？") == "chat_record"
    )
    assert _detect_document_type("聊天.txt", "14:20 张三：在吗？") == "chat_record"


def test_detect_document_type_tables():
    # 表格文件不被文件名关键词劫持（"微信转账流水.csv" 不应判为聊天）
    assert _detect_document_type("微信转账流水.csv", "") == "csv_table"
    assert _detect_document_type("通话记录.xlsx", "") == "spreadsheet"
    assert _detect_document_type("普通文档.md", "这是一段普通文本内容。") == "general"


# ---------------------------------------------------------------- 笔录分块


def test_chunk_transcript_splits_qa_pairs():
    content = (
        "询问时间：2025-10-05\n姓名：张三\n身份证号：330100199001011234\n联系电话：13857906361\n"
        "问：你认识陈锦标吗？\n答：认识，他是我的上线。\n"
        "问：你们怎么联系的？\n答：用微信联系。"
    )
    chunks = _chunk_transcript(content, {})
    assert chunks
    # 元信息头独立成块
    assert "询问时间" in chunks[0]
    # 问答对保留 "问：" 前缀
    assert any("问：你认识陈锦标吗" in c for c in chunks)
    # 命中身份信号（姓名+身份证+电话 ≥2 项）→ 注入笔录头前缀
    assert any("〔笔录头：" in c for c in chunks)


def test_chunk_transcript_falls_back_without_qa():
    chunks = _chunk_transcript("纯叙述文本，没有问答结构。", {})
    assert chunks  # 回退 general 不丢数据


# ---------------------------------------------------------------- 聊天分块


def test_chunk_chat_record_time_window_prefix():
    content = (
        "2025-10-05 14:20 张三：到账了没？\n"
        "2025-10-05 14:25 李四：到了，5 万。\n"
        "2025-10-05 14:28 张三：转给邓家俊。\n"
        "2025-10-05 15:40 张三：下午见个面。\n"  # 与上一条差 70 分钟 > 30 → 断块
    )
    chunks = _chunk_chat_record(content, {"chat_time_gap_minutes": 30})
    assert len(chunks) >= 2  # 时间窗断成 ≥2 块
    # 每块带【时间段】前缀（事件抽取 time_expr 的高置信来源）
    assert all("【时间段" in c for c in chunks)


def test_chunk_chat_record_gap_disabled():
    content = "14:20 张三：在吗？\n14:25 李四：在。"
    chunks = _chunk_chat_record(content, {"chat_time_gap_minutes": 0})
    assert chunks


# ---------------------------------------------------------------- CSV/表格分块


def test_chunk_csv_table_key_value_records():
    content = (
        "| 流水号 | 付款人 | 收款人 | 金额 | 经度 | 纬度 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| FB2026-0001 | 陈锦标 | 邓家俊 | 50000 | 120.1 | 30.2 |\n"
    )
    chunks = _chunk_csv_table(content, {})
    assert chunks
    # 键值对格式 + 表头上下文注入
    assert any("表头字段：" in c for c in chunks)
    assert any("流水号：FB2026-0001" in c for c in chunks)
    # 坐标列被默认裁剪（S2-B1），不进入键值对
    assert not any("经度" in c for c in chunks)


# ---------------------------------------------------------------- dispatcher 落库


def test_dispatcher_chunk_records_include_doc_type():
    records = chunk_markdown(
        "问：你叫什么？\n答：张三。",
        file_id="f1",
        filename="询问笔录.md",
        processing_params={"chunk_preset_id": "case_document"},
    )
    assert records
    assert all(record["doc_type"] == "transcript" for record in records)


def test_detect_chunk_doc_type_helper():
    assert (
        _detect_chunk_doc_type(
            "case_document", "通话记录.csv", "| 通话方 | 时长 |", {}
        )
        == "csv_table"
    )
    assert _detect_chunk_doc_type("general", "普通文档.md", "文本", {}) == "general"
    assert normalize_chunk_preset_id("case_document") == "case_document"


# ---------------------------------------------------------------- init_db 幂等列


async def test_init_db_idempotent_adds_doc_type_column(tmp_path):
    engine = create_engine(db_path=tmp_path / "test.db")
    await init_db(engine)
    await init_db(engine)  # 第二次幂等

    import sqlalchemy as sa

    async with engine.begin() as conn:
        result = await conn.run_sync(
            lambda sync_conn: sync_conn.execute(
                sa.text("PRAGMA table_info(ys_knowledge_chunks)")
            ).mappings()
        )
        rows = list(result)
    names = {row["name"] for row in rows}
    assert "doc_type" in names
    await engine.dispose()
