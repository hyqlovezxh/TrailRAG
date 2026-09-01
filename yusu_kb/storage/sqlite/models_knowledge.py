"""SQLite knowledge models — 13 tables adapted from YUSU ``models_knowledge.py``.

All table names carry the ``ys_`` prefix (adapted from the LightRAG-era
``yushu_`` hosting convention) to avoid collisions. JSON columns use plain
SQLAlchemy ``JSON`` (SQLite serializes to TEXT). ``LargeBinary`` maps to
SQLite BLOB.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)

from ...utils.datetime_utils import utc_now_naive
from .engine import Base

JSON_VALUE = JSON


class KnowledgeBase(Base):
    """知识库模型"""

    __tablename__ = "ys_knowledge_bases"
    __table_args__ = (UniqueConstraint("kb_id", name="uq_ys_knowledge_bases_kb_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    kb_id = Column(String(80), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    kb_type = Column(String(32), nullable=False, index=True)
    embedding_model_spec = Column(String(512))
    llm_model_spec = Column(String(512))
    query_params = Column(JSON_VALUE)
    additional_params = Column(JSON_VALUE)
    share_config = Column(JSON_VALUE)
    created_by = Column(String(64))
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)


class KnowledgeFile(Base):
    """知识文件模型"""

    __tablename__ = "ys_knowledge_files"
    __table_args__ = (UniqueConstraint("file_id", name="uq_ys_knowledge_files_file_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String(64), unique=True, nullable=False, index=True)
    kb_id = Column(
        String(80), ForeignKey("ys_knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_id = Column(String(64), ForeignKey("ys_knowledge_files.file_id", ondelete="SET NULL"), index=True)
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512))
    file_type = Column(String(64))
    path = Column(String(1024))
    local_url = Column(String(1024))
    markdown_file = Column(String(1024))
    status = Column(String(32), default="uploaded", index=True)
    content_hash = Column(String(128), index=True)
    file_size = Column(BigInteger)
    chunk_count = Column(Integer, default=0)
    token_count = Column(BigInteger, default=0)
    content_type = Column(String(64))
    processing_params = Column(JSON_VALUE)
    is_folder = Column(Boolean, default=False)
    error_message = Column(Text)
    created_by = Column(String(64))
    updated_by = Column(String(64))
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)


class KnowledgeChunk(Base):
    """知识库 Chunk 模型"""

    __tablename__ = "ys_knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_ys_knowledge_chunks_chunk_id"),
        Index("ix_ys_knowledge_chunks_file_id", "file_id"),
        Index("ix_ys_knowledge_chunks_kb_id", "kb_id"),
        Index("ix_ys_knowledge_chunks_graph_indexed", "graph_indexed"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(128), nullable=False)
    file_id = Column(
        String(64), ForeignKey("ys_knowledge_files.file_id", ondelete="CASCADE"), nullable=False
    )
    kb_id = Column(
        String(80), ForeignKey("ys_knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    start_char_pos = Column(Integer)
    end_char_pos = Column(Integer)
    start_token_pos = Column(Integer)
    end_token_pos = Column(Integer)
    graph_indexed = Column(Boolean, default=False)
    ent_ids = Column(JSON_VALUE)
    tags = Column(JSON_VALUE)
    extraction_result = Column(JSON_VALUE)
    # 文档类型（事件驱动双路径路由判据，构建与查询读同一份）
    doc_type = Column(String(32))
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)


class KnowledgeGraphEntity(Base):
    """知识图谱实体"""

    __tablename__ = "ys_knowledge_graph_entities"
    __table_args__ = (
        UniqueConstraint("entity_id", name="uq_ys_knowledge_graph_entities_entity_id"),
        UniqueConstraint("kb_id", "normalized_name", "label", name="uq_ys_knowledge_graph_entities_identity"),
        Index("ix_ys_knowledge_graph_entities_kb_id", "kb_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(64), nullable=False)
    kb_id = Column(
        String(80), ForeignKey("ys_knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False
    )
    normalized_name = Column(String(512), nullable=False)
    label = Column(String(128), nullable=False)
    name = Column(String(512), nullable=False)
    attributes = Column(JSON_VALUE)
    description = Column(Text)
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)


class KnowledgeGraphEntityMention(Base):
    """知识图谱实体在 chunk 中的引用"""

    __tablename__ = "ys_knowledge_graph_entity_mentions"
    __table_args__ = (
        UniqueConstraint("entity_id", "chunk_id", name="uq_ys_knowledge_graph_entity_mentions_entity_chunk"),
        Index("ix_ys_knowledge_graph_entity_mentions_kb_id", "kb_id"),
        Index("ix_ys_knowledge_graph_entity_mentions_file_id", "file_id"),
        Index("ix_ys_knowledge_graph_entity_mentions_chunk_id", "chunk_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(
        String(64),
        ForeignKey("ys_knowledge_graph_entities.entity_id", ondelete="CASCADE"),
        nullable=False,
    )
    kb_id = Column(
        String(80), ForeignKey("ys_knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False
    )
    file_id = Column(String(64), ForeignKey("ys_knowledge_files.file_id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(
        String(128), ForeignKey("ys_knowledge_chunks.chunk_id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(DateTime, default=utc_now_naive)


class KnowledgeGraphTriple(Base):
    """知识图谱三元组"""

    __tablename__ = "ys_knowledge_graph_triples"
    __table_args__ = (
        UniqueConstraint("triple_id", name="uq_ys_knowledge_graph_triples_triple_id"),
        Index("ix_ys_knowledge_graph_triples_kb_id", "kb_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    triple_id = Column(String(64), nullable=False)
    kb_id = Column(
        String(80), ForeignKey("ys_knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False
    )
    source_entity_id = Column(
        String(64),
        ForeignKey("ys_knowledge_graph_entities.entity_id", ondelete="CASCADE"),
        nullable=False,
    )
    target_entity_id = Column(
        String(64),
        ForeignKey("ys_knowledge_graph_entities.entity_id", ondelete="CASCADE"),
        nullable=False,
    )
    relation_type = Column(String(256), nullable=False)
    content = Column(Text, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)


class KnowledgeGraphTripleMention(Base):
    """知识图谱三元组在 chunk 中的引用"""

    __tablename__ = "ys_knowledge_graph_triple_mentions"
    __table_args__ = (
        UniqueConstraint("triple_id", "chunk_id", name="uq_ys_knowledge_graph_triple_mentions_triple_chunk"),
        Index("ix_ys_knowledge_graph_triple_mentions_kb_id", "kb_id"),
        Index("ix_ys_knowledge_graph_triple_mentions_file_id", "file_id"),
        Index("ix_ys_knowledge_graph_triple_mentions_chunk_id", "chunk_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    triple_id = Column(
        String(64),
        ForeignKey("ys_knowledge_graph_triples.triple_id", ondelete="CASCADE"),
        nullable=False,
    )
    kb_id = Column(
        String(80), ForeignKey("ys_knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False
    )
    file_id = Column(String(64), ForeignKey("ys_knowledge_files.file_id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(
        String(128), ForeignKey("ys_knowledge_chunks.chunk_id", ondelete="CASCADE"), nullable=False
    )
    text = Column(Text)
    extractor_type = Column(String(128))
    created_at = Column(DateTime, default=utc_now_naive)


class EvaluationDataset(Base):
    """评估数据集模型"""

    __tablename__ = "ys_evaluation_datasets"
    __table_args__ = (UniqueConstraint("dataset_id", name="uq_ys_evaluation_datasets_dataset_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(64), unique=True, nullable=False, index=True)
    kb_id = Column(
        String(80), ForeignKey("ys_knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False, index=True
    )
    name = Column(String(255), nullable=False)
    description = Column(Text)
    item_count = Column(Integer, default=0)
    has_gold_chunks = Column(Boolean, default=False)
    has_gold_answers = Column(Boolean, default=False)
    build_metadata = Column(JSON_VALUE)
    created_by = Column(String(64))
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)


class EvaluationDatasetItem(Base):
    """评估数据集题目模型"""

    __tablename__ = "ys_evaluation_dataset_items"
    __table_args__ = (
        UniqueConstraint("item_id", name="uq_ys_evaluation_dataset_items_item_id"),
        UniqueConstraint("dataset_id", "item_index", name="uq_ys_evaluation_dataset_items_dataset_index"),
        Index("ix_ys_evaluation_dataset_items_dataset_index", "dataset_id", "item_index"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(String(64), unique=True, nullable=False, index=True)
    dataset_id = Column(
        String(64),
        ForeignKey("ys_evaluation_datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kb_id = Column(
        String(80), ForeignKey("ys_knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False, index=True
    )
    item_index = Column(Integer, nullable=False)
    query_text = Column(Text, nullable=False)
    gold_chunk_ids = Column(JSON_VALUE)
    gold_answer = Column(Text)
    created_at = Column(DateTime, default=utc_now_naive)


class EvaluationRun(Base):
    """评估运行模型"""

    __tablename__ = "ys_evaluation_runs"
    __table_args__ = (UniqueConstraint("run_id", name="uq_ys_evaluation_runs_run_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    kb_id = Column(
        String(80), ForeignKey("ys_knowledge_bases.kb_id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id = Column(
        String(64),
        ForeignKey("ys_evaluation_datasets.dataset_id", ondelete="SET NULL"),
        index=True,
    )
    status = Column(String(32), default="running", index=True)
    retrieval_config = Column(JSON_VALUE)
    metrics = Column(JSON_VALUE)
    overall_score = Column(Float)
    total_items = Column(Integer, default=0)
    completed_items = Column(Integer, default=0)
    started_at = Column(DateTime, default=utc_now_naive, index=True)
    completed_at = Column(DateTime)
    created_by = Column(String(64))
    error_message = Column(Text)


class EvaluationRunItem(Base):
    """评估逐题结果模型"""

    __tablename__ = "ys_evaluation_run_items"
    __table_args__ = (
        UniqueConstraint("run_id", "item_index", name="uq_ys_evaluation_run_items_run_index"),
        Index("ix_ys_evaluation_run_items_run_index", "run_id", "item_index"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(
        String(64),
        ForeignKey("ys_evaluation_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_item_id = Column(
        String(64), ForeignKey("ys_evaluation_dataset_items.item_id", ondelete="SET NULL"), index=True
    )
    item_index = Column(Integer, nullable=False)
    query_text = Column(Text, nullable=False)
    gold_chunk_ids = Column(JSON_VALUE)
    gold_answer = Column(Text)
    generated_answer = Column(Text)
    retrieved_chunks = Column(JSON_VALUE)
    metrics = Column(JSON_VALUE)
    is_error = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=utc_now_naive)


class ModelProvider(Base):
    """独立模型供应商配置

    密钥只存环境变量引用（api_key_env），DB 永不存真实 key。
    """

    __tablename__ = "ys_model_providers"
    __table_args__ = (UniqueConstraint("provider_id", name="uq_ys_model_providers_provider_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    provider_type = Column(String(32), default="openai")
    default_protocol = Column(String(32))
    base_url = Column(String(1024), nullable=False)
    embedding_base_url = Column(String(1024))
    rerank_base_url = Column(String(1024))
    models_endpoint = Column(String(1024))
    embedding_models_endpoint = Column(String(1024))
    rerank_models_endpoint = Column(String(1024))
    api_key_env = Column(String(128))
    capabilities = Column(JSON_VALUE)
    enabled_models = Column(JSON_VALUE)
    headers_json = Column(JSON_VALUE)
    extra_json = Column(JSON_VALUE)
    is_enabled = Column(Boolean, default=True)
    is_builtin = Column(Boolean, default=False)
    created_by = Column(String(64))
    updated_by = Column(String(64))
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)


class AppConfig(Base):
    """应用级配置项（key/value JSON，如 default_chat_model_spec）"""

    __tablename__ = "ys_app_config"
    __table_args__ = (UniqueConstraint("config_key", name="uq_ys_app_config_key"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    config_key = Column(String(128), unique=True, nullable=False, index=True)
    config_value = Column(JSON_VALUE)
    updated_by = Column(String(64))
    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)


class LLMCacheEntry(Base):
    """LLM 响应缓存条目（L2 缓存）"""

    __tablename__ = "ys_llm_cache_entries"
    __table_args__ = (
        Index("ix_ys_llm_cache_entries_mode", "mode"),
        Index("ix_ys_llm_cache_entries_cache_type", "cache_type"),
        Index("ix_ys_llm_cache_entries_chunk_id", "chunk_id"),
        Index("ix_ys_llm_cache_entries_model_spec", "model_spec"),
        Index("ix_ys_llm_cache_entries_create_time", "create_time"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(128), unique=True, nullable=False, index=True)
    mode = Column(String(32), nullable=False)
    cache_type = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    original_prompt = Column(Text)
    chunk_id = Column(String(64))
    model_spec = Column(String(128), nullable=False)
    create_time = Column(DateTime, default=utc_now_naive)
    access_time = Column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)
    access_count = Column(Integer, default=0)


class EmbeddingCacheEntry(Base):
    """Embedding 向量缓存条目（L3 缓存）"""

    __tablename__ = "ys_embedding_cache_entries"
    __table_args__ = (
        Index("ix_ys_embedding_cache_entries_content_hash", "content_hash"),
        Index("ix_ys_embedding_cache_entries_model_spec", "model_spec"),
        Index("ix_ys_embedding_cache_entries_create_time", "create_time"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(128), unique=True, nullable=False, index=True)
    content_hash = Column(String(64), nullable=False)
    model_spec = Column(String(128), nullable=False)
    vector = Column(LargeBinary, nullable=False)
    dim = Column(Integer, nullable=False)
    create_time = Column(DateTime, default=utc_now_naive)


__all__ = [
    "AppConfig",
    "EmbeddingCacheEntry",
    "EvaluationDataset",
    "EvaluationDatasetItem",
    "EvaluationRun",
    "EvaluationRunItem",
    "KnowledgeBase",
    "KnowledgeChunk",
    "KnowledgeFile",
    "KnowledgeGraphEntity",
    "KnowledgeGraphEntityMention",
    "KnowledgeGraphTriple",
    "KnowledgeGraphTripleMention",
    "LLMCacheEntry",
    "ModelProvider",
]